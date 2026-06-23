package mcpgrafana

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	// mcpProbeTimeout is the timeout for probing a single datasource's MCP endpoint.
	// This is kept short to avoid slow startup when datasources are unreachable.
	mcpProbeTimeout = 5 * time.Second
)

// MCPDatasourceConfig defines configuration for a datasource type that supports MCP
type MCPDatasourceConfig struct {
	Type         string
	EndpointPath string // e.g., "/api/mcp"
}

// mcpEnabledDatasources is a registry of datasource types that support MCP
var mcpEnabledDatasources = map[string]MCPDatasourceConfig{
	"tempo": {Type: "tempo", EndpointPath: "/api/mcp"},
	// Future: add other datasource types here
}

// DiscoveredDatasource represents a datasource that supports MCP
type DiscoveredDatasource struct {
	UID    string
	Name   string
	Type   string
	MCPURL string // The MCP endpoint URL
}

// discoverMCPDatasources discovers datasources that support MCP
// Returns a list of datasources with MCP endpoints
func discoverMCPDatasources(ctx context.Context) ([]DiscoveredDatasource, error) {
	gc := GrafanaClientFromContext(ctx)
	if gc == nil {
		return nil, fmt.Errorf("grafana client not found in context")
	}

	var discovered []DiscoveredDatasource

	// List all datasources
	resp, err := gc.Datasources.GetDataSources()
	if err != nil {
		return nil, fmt.Errorf("failed to list datasources: %w", err)
	}

	// Get the Grafana base URL from context
	config := GrafanaConfigFromContext(ctx)
	if config.URL == "" {
		return nil, fmt.Errorf("grafana url not found in context")
	}
	grafanaBaseURL := config.URL

	// Filter for datasources that support MCP and collect candidates
	type candidate struct {
		uid      string
		name     string
		dsType   string
		dsConfig MCPDatasourceConfig
	}
	var candidates []candidate
	for _, ds := range resp.Payload {
		// Check if this datasource type supports MCP
		dsConfig, supported := mcpEnabledDatasources[ds.Type]
		if !supported {
			continue
		}
		candidates = append(candidates, candidate{
			uid:      ds.UID,
			name:     ds.Name,
			dsType:   ds.Type,
			dsConfig: dsConfig,
		})
	}

	if len(candidates) == 0 {
		slog.DebugContext(ctx, "no candidate MCP datasources found")
		return nil, nil
	}

	// Probe candidates in parallel with timeout
	type probeResult struct {
		ds      DiscoveredDatasource
		enabled bool
	}
	results := make(chan probeResult, len(candidates))
	var wg sync.WaitGroup

	for _, c := range candidates {
		wg.Add(1)
		go func(c candidate) {
			defer wg.Done()

			// Create a context with timeout for this probe
			probeCtx, cancel := context.WithTimeout(ctx, mcpProbeTimeout)
			defer cancel()

			// Check if the datasource instance has MCP enabled
			// We use a DELETE request to probe the MCP endpoint since:
			// - GET would start an event stream and hang
			// - POST doesn't work with the Grafana OpenAPI client
			// - DELETE returns 200 if MCP is enabled, 404 if not
			//
			// Note: We create a new client with the probe context to respect the timeout.
			// The existing client doesn't pass context to HTTP requests properly.
			probeURL := fmt.Sprintf("%s/api/datasources/proxy/uid/%s%s", grafanaBaseURL, c.uid, c.dsConfig.EndpointPath)
			req, err := http.NewRequestWithContext(probeCtx, http.MethodDelete, probeURL, nil)
			if err != nil {
				slog.DebugContext(ctx, "failed to create probe request", "datasource", c.uid, "error", err)
				return
			}

			// Add authentication headers from the Grafana config
			if config.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+config.APIKey)
			} else if config.BasicAuth != nil {
				password, _ := config.BasicAuth.Password()
				req.SetBasicAuth(config.BasicAuth.Username(), password)
			}

			httpClient := &http.Client{Timeout: mcpProbeTimeout}
			resp, err := httpClient.Do(req)
			if err != nil {
				slog.DebugContext(ctx, "MCP probe failed", "datasource", c.uid, "error", err)
				return
			}
			defer func() { _ = resp.Body.Close() }()

			// MCP is enabled if we get a 200 response
			if resp.StatusCode == http.StatusOK {
				mcpURL := fmt.Sprintf("%s/api/datasources/proxy/uid/%s%s", grafanaBaseURL, c.uid, c.dsConfig.EndpointPath)
				results <- probeResult{
					ds: DiscoveredDatasource{
						UID:    c.uid,
						Name:   c.name,
						Type:   c.dsType,
						MCPURL: mcpURL,
					},
					enabled: true,
				}
			}
		}(c)
	}

	// Wait for all probes to complete and close results channel
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	for result := range results {
		if result.enabled {
			discovered = append(discovered, result.ds)
		}
	}

	slog.DebugContext(ctx, "discovered MCP datasources", "count", len(discovered), "candidates", len(candidates))
	return discovered, nil
}

// addDatasourceUidParameter adds a required datasourceUid parameter to a tool's input schema
func addDatasourceUidParameter(tool mcp.Tool, datasourceType string) mcp.Tool {
	modifiedTool := tool
	// Prefix tool name with datasource type (e.g., "tempo_traceql-search")
	modifiedTool.Name = datasourceType + "_" + tool.Name

	// Add datasourceUid to the input schema
	if modifiedTool.InputSchema.Properties == nil {
		modifiedTool.InputSchema.Properties = make(map[string]any)
	}

	modifiedTool.InputSchema.Properties["datasourceUid"] = map[string]any{
		"type":        "string",
		"description": "UID of the " + datasourceType + " datasource to query",
	}

	// Add to required fields
	modifiedTool.InputSchema.Required = append(modifiedTool.InputSchema.Required, "datasourceUid")

	return modifiedTool
}

// parseProxiedToolName extracts datasource type and original tool name from a proxied tool name
// Format: <datasource_type>_<original_tool_name>
// Returns: datasourceType, originalToolName, error
func parseProxiedToolName(toolName string) (string, string, error) {
	parts := strings.SplitN(toolName, "_", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("invalid proxied tool name format: %s", toolName)
	}
	return parts[0], parts[1], nil
}

// ToolManager manages proxied tools (either per-session or server-wide)
type ToolManager struct {
	sm     *SessionManager
	server *server.MCPServer

	// Whether to enable proxied tools.
	enableProxiedTools bool

	// For stdio transport: store clients at manager level (single-tenant).
	// These will be unused for HTTP/SSE transports.
	serverMode    bool // true if using server-wide tools (stdio), false for per-session (HTTP/SSE)
	serverClients map[string]*ProxiedClient
	clientsMutex  sync.RWMutex
}

// NewToolManager creates a new ToolManager
func NewToolManager(sm *SessionManager, mcpServer *server.MCPServer, opts ...toolManagerOption) *ToolManager {
	tm := &ToolManager{
		sm:            sm,
		server:        mcpServer,
		serverClients: make(map[string]*ProxiedClient),
	}
	for _, opt := range opts {
		opt(tm)
	}
	return tm
}

type toolManagerOption func(*ToolManager)

// WithProxiedTools sets whether proxied tools are enabled
func WithProxiedTools(enabled bool) toolManagerOption {
	return func(tm *ToolManager) {
		tm.enableProxiedTools = enabled
	}
}

// InitializeAndRegisterServerTools discovers datasources and registers tools on the server (for stdio transport)
// This should be called once at server startup for single-tenant stdio servers
func (tm *ToolManager) InitializeAndRegisterServerTools(ctx context.Context) error {
	if !tm.enableProxiedTools {
		return nil
	}

	// Mark as server mode (stdio transport)
	tm.serverMode = true

	// Discover datasources with MCP support
	discovered, err := discoverMCPDatasources(ctx)
	if err != nil {
		return fmt.Errorf("failed to discover MCP datasources: %w", err)
	}

	if len(discovered) == 0 {
		slog.Info("no MCP datasources discovered")
		return nil
	}

	// Connect to each datasource and store in manager
	tm.clientsMutex.Lock()
	for _, ds := range discovered {
		client, err := NewProxiedClient(ctx, ds.UID, ds.Name, ds.Type, ds.MCPURL)
		if err != nil {
			slog.Error("failed to create proxied client", "datasource", ds.UID, "error", err)
			continue
		}
		key := ds.Type + "_" + ds.UID
		tm.serverClients[key] = client
	}
	clientCount := len(tm.serverClients)
	tm.clientsMutex.Unlock()

	if clientCount == 0 {
		slog.Warn("no proxied clients created")
		return nil
	}

	slog.Info("connected to proxied MCP servers", "datasources", clientCount)

	// Collect and register all unique tools
	tm.clientsMutex.RLock()
	toolMap := make(map[string]mcp.Tool)
	for _, client := range tm.serverClients {
		for _, tool := range client.ListTools() {
			toolName := client.DatasourceType + "_" + tool.Name
			if _, exists := toolMap[toolName]; !exists {
				modifiedTool := addDatasourceUidParameter(tool, client.DatasourceType)
				toolMap[toolName] = modifiedTool
			}
		}
	}
	tm.clientsMutex.RUnlock()

	// Register tools on the server (not per-session)
	for toolName, tool := range toolMap {
		handler := NewProxiedToolHandler(tm.sm, tm, toolName)
		tm.server.AddTool(tool, handler.Handle)
	}

	slog.Info("registered proxied tools on server", "tools", len(toolMap))
	return nil
}

// InitializeAndRegisterProxiedTools discovers datasources, creates clients, and registers tools per-session
// This should be called in OnBeforeListTools and OnBeforeCallTool hooks for HTTP/SSE transports
func (tm *ToolManager) InitializeAndRegisterProxiedTools(ctx context.Context, session server.ClientSession) {
	if !tm.enableProxiedTools {
		return
	}

	sessionID := session.SessionID()
	state, exists := tm.sm.GetSession(sessionID)
	if !exists {
		// Session exists in server context but not in our SessionManager yet
		tm.sm.CreateSession(ctx, session)
		state, exists = tm.sm.GetSession(sessionID)
		if !exists {
			slog.Error("failed to create session in SessionManager", "sessionID", sessionID)
			return
		}
	}

	// Step 1: Discover and connect (guaranteed to run exactly once per session)
	state.initOnce.Do(func() {
		// Discover datasources with MCP support
		discovered, err := discoverMCPDatasources(ctx)
		if err != nil {
			slog.Error("failed to discover MCP datasources", "error", err)
			state.mutex.Lock()
			state.proxiedToolsInitialized = true
			state.mutex.Unlock()
			return
		}

		state.mutex.Lock()
		// For each discovered datasource, create a proxied client
		for _, ds := range discovered {
			client, err := NewProxiedClient(ctx, ds.UID, ds.Name, ds.Type, ds.MCPURL)
			if err != nil {
				slog.Error("failed to create proxied client", "datasource", ds.UID, "error", err)
				continue
			}

			// Store the client
			key := ds.Type + "_" + ds.UID
			state.proxiedClients[key] = client
		}
		state.proxiedToolsInitialized = true
		state.mutex.Unlock()

		slog.Info("connected to proxied MCP servers", "session", sessionID, "datasources", len(state.proxiedClients))
	})

	// Step 2: Register tools with the MCP server
	state.mutex.Lock()
	defer state.mutex.Unlock()

	// Check if tools already registered
	if len(state.proxiedTools) > 0 {
		return
	}

	// Check if we have any clients (discovery should have happened above)
	if len(state.proxiedClients) == 0 {
		return
	}

	// First pass: collect all unique tools and track which datasources support them
	toolMap := make(map[string]mcp.Tool) // unique tools by name

	for key, client := range state.proxiedClients {
		remoteTools := client.ListTools()

		for _, tool := range remoteTools {
			// Tool name format: datasourceType_originalToolName (e.g., "tempo_traceql-search")
			toolName := client.DatasourceType + "_" + tool.Name

			// Store the tool if we haven't seen it yet
			if _, exists := toolMap[toolName]; !exists {
				// Add datasourceUid parameter to the tool
				modifiedTool := addDatasourceUidParameter(tool, client.DatasourceType)
				toolMap[toolName] = modifiedTool
			}

			// Track which datasources support this tool
			state.toolToDatasources[toolName] = append(state.toolToDatasources[toolName], key)
		}
	}

	// Second pass: register all unique tools at once (reduces listChanged notifications)
	var serverTools []server.ServerTool
	for toolName, tool := range toolMap {
		handler := NewProxiedToolHandler(tm.sm, tm, toolName)
		serverTools = append(serverTools, server.ServerTool{
			Tool:    tool,
			Handler: handler.Handle,
		})
		state.proxiedTools = append(state.proxiedTools, tool)
	}

	if err := tm.server.AddSessionTools(sessionID, serverTools...); err != nil {
		slog.Warn("failed to add session tools", "session", sessionID, "error", err)
	} else {
		slog.Info("registered proxied tools", "session", sessionID, "tools", len(state.proxiedTools))
	}
}

// GetServerClient retrieves a proxied client from server-level storage (for stdio transport)
func (tm *ToolManager) GetServerClient(datasourceType, datasourceUID string) (*ProxiedClient, error) {
	tm.clientsMutex.RLock()
	defer tm.clientsMutex.RUnlock()

	key := datasourceType + "_" + datasourceUID
	client, exists := tm.serverClients[key]
	if !exists {
		// List available datasources to help with debugging
		var availableUIDs []string
		for _, c := range tm.serverClients {
			if c.DatasourceType == datasourceType {
				availableUIDs = append(availableUIDs, c.DatasourceUID)
			}
		}

		if len(availableUIDs) > 0 {
			return nil, fmt.Errorf("datasource '%s' not found. Available %s datasources: %v", datasourceUID, datasourceType, availableUIDs)
		}
		return nil, fmt.Errorf("datasource '%s' not found. No %s datasources with MCP support are configured", datasourceUID, datasourceType)
	}

	return client, nil
}
