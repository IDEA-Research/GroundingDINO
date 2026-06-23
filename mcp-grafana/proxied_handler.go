package mcpgrafana

import (
	"context"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// ProxiedToolHandler implements the CallToolHandler interface for proxied tools
type ProxiedToolHandler struct {
	sessionManager *SessionManager
	toolManager    *ToolManager
	toolName       string
}

// NewProxiedToolHandler creates a new handler for a proxied tool
func NewProxiedToolHandler(sm *SessionManager, tm *ToolManager, toolName string) *ProxiedToolHandler {
	return &ProxiedToolHandler{
		sessionManager: sm,
		toolManager:    tm,
		toolName:       toolName,
	}
}

// Handle forwards the tool call to the appropriate remote MCP server
func (h *ProxiedToolHandler) Handle(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// Check if session is in context
	session := server.ClientSessionFromContext(ctx)
	if session == nil {
		return nil, fmt.Errorf("session not found in context")
	}

	// Extract arguments
	args, ok := request.Params.Arguments.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("invalid arguments type")
	}

	// Extract required datasourceUid parameter
	datasourceUidRaw, ok := args["datasourceUid"]
	if !ok {
		return nil, fmt.Errorf("datasourceUid parameter is required")
	}
	datasourceUID, ok := datasourceUidRaw.(string)
	if !ok {
		return nil, fmt.Errorf("datasourceUid must be a string")
	}

	// Parse the tool name to get datasource type and original tool name
	// Format: datasourceType_originalToolName (e.g., "tempo_traceql-search")
	datasourceType, originalToolName, err := parseProxiedToolName(h.toolName)
	if err != nil {
		return nil, fmt.Errorf("failed to parse tool name: %w", err)
	}

	// Get the proxied client for this datasource
	var client *ProxiedClient

	if h.toolManager.serverMode {
		// Server mode (stdio): clients stored at manager level
		client, err = h.toolManager.GetServerClient(datasourceType, datasourceUID)
	} else {
		// Session mode (HTTP/SSE): clients stored per-session
		client, err = h.sessionManager.GetProxiedClient(ctx, datasourceType, datasourceUID)
		if err != nil {
			// Fallback to server-level in case of mixed mode
			client, err = h.toolManager.GetServerClient(datasourceType, datasourceUID)
		}
	}

	if err != nil {
		return nil, fmt.Errorf("datasource '%s' not found or not accessible. Ensure the datasource exists and you have permission to access it", datasourceUID)
	}

	// Remove datasourceUid from args before forwarding to remote server
	forwardArgs := make(map[string]any)
	for k, v := range args {
		if k != "datasourceUid" {
			forwardArgs[k] = v
		}
	}

	// Forward the call to the remote MCP server
	return client.CallTool(ctx, originalToolName, forwardArgs)
}
