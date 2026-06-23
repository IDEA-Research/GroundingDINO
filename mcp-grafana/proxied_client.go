package mcpgrafana

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"sync"

	mcp_client "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

// ProxiedClient represents a connection to a remote MCP server (e.g., Tempo datasource)
type ProxiedClient struct {
	DatasourceUID  string
	DatasourceName string
	DatasourceType string
	Client         *mcp_client.Client
	Tools          []mcp.Tool
	mutex          sync.RWMutex
}

// NewProxiedClient creates a new connection to a remote MCP server
func NewProxiedClient(ctx context.Context, datasourceUID, datasourceName, datasourceType, mcpEndpoint string) (*ProxiedClient, error) {
	// Get Grafana config for authentication
	config := GrafanaConfigFromContext(ctx)

	// Build headers for authentication
	headers := make(map[string]string)
	if config.APIKey != "" {
		headers["Authorization"] = "Bearer " + config.APIKey
	} else if config.BasicAuth != nil {
		auth := config.BasicAuth.String()
		headers["Authorization"] = "Basic " + base64.StdEncoding.EncodeToString([]byte(auth))
	}

	// Add org ID header if configured
	if config.OrgID != 0 {
		headers["X-Grafana-Org-Id"] = fmt.Sprintf("%d", config.OrgID)
	}

	// Create HTTP transport with authentication and org ID headers
	slog.DebugContext(ctx, "connecting to MCP server", "datasource", datasourceUID, "url", mcpEndpoint)
	httpTransport, err := transport.NewStreamableHTTP(
		mcpEndpoint,
		transport.WithHTTPHeaders(headers),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP transport: %w", err)
	}

	// Create MCP client
	mcpClient := mcp_client.NewClient(httpTransport)

	// Initialize the connection
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{
		Name:    "mcp-grafana-proxy",
		Version: Version(),
	}

	_, err = mcpClient.Initialize(ctx, initReq)
	if err != nil {
		_ = mcpClient.Close()
		return nil, fmt.Errorf("failed to initialize MCP client: %w", err)
	}

	// List available tools from the remote server
	listReq := mcp.ListToolsRequest{}
	toolsResult, err := mcpClient.ListTools(ctx, listReq)
	if err != nil {
		_ = mcpClient.Close()
		return nil, fmt.Errorf("failed to list tools from remote MCP server: %w", err)
	}

	slog.DebugContext(ctx, "connected to proxied MCP server",
		"datasource", datasourceUID,
		"type", datasourceType,
		"tools", len(toolsResult.Tools))

	return &ProxiedClient{
		DatasourceUID:  datasourceUID,
		DatasourceName: datasourceName,
		DatasourceType: datasourceType,
		Client:         mcpClient,
		Tools:          toolsResult.Tools,
	}, nil
}

// CallTool forwards a tool call to the remote MCP server
func (pc *ProxiedClient) CallTool(ctx context.Context, toolName string, arguments map[string]any) (*mcp.CallToolResult, error) {
	pc.mutex.RLock()
	defer pc.mutex.RUnlock()

	// Validate the tool exists
	var toolExists bool
	for _, tool := range pc.Tools {
		if tool.Name == toolName {
			toolExists = true
			break
		}
	}
	if !toolExists {
		return nil, fmt.Errorf("tool %s not found in remote MCP server", toolName)
	}

	// Create the call tool request
	req := mcp.CallToolRequest{}
	req.Params.Name = toolName
	req.Params.Arguments = arguments

	// Forward the call to the remote server
	result, err := pc.Client.CallTool(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to call tool on remote MCP server: %w", err)
	}

	return result, nil
}

// ListTools returns the tools available from this remote server
// Note: This method doesn't take a context parameter as the tools are cached locally
func (pc *ProxiedClient) ListTools() []mcp.Tool {
	pc.mutex.RLock()
	defer pc.mutex.RUnlock()

	// Return a copy to prevent external modification
	result := make([]mcp.Tool, len(pc.Tools))
	copy(result, pc.Tools)
	return result
}

// Close closes the connection to the remote MCP server
func (pc *ProxiedClient) Close() error {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()

	if pc.Client != nil {
		if err := pc.Client.Close(); err != nil {
			return fmt.Errorf("failed to close MCP client: %w", err)
		}
	}

	return nil
}
