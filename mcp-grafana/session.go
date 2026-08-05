package mcpgrafana

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// SessionState holds the state for a single client session
type SessionState struct {
	// Proxied tools state
	initOnce                sync.Once
	proxiedToolsInitialized bool
	proxiedTools            []mcp.Tool
	proxiedClients          map[string]*ProxiedClient // key: datasourceType_datasourceUID
	toolToDatasources       map[string][]string       // key: toolName, value: list of datasource keys that support it
	mutex                   sync.RWMutex
}

func newSessionState() *SessionState {
	return &SessionState{
		proxiedClients:    make(map[string]*ProxiedClient),
		toolToDatasources: make(map[string][]string),
	}
}

// SessionManager manages client sessions and their state
type SessionManager struct {
	sessions map[string]*SessionState
	mutex    sync.RWMutex
}

func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions: make(map[string]*SessionState),
	}
}

func (sm *SessionManager) CreateSession(ctx context.Context, session server.ClientSession) {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	sessionID := session.SessionID()
	if _, exists := sm.sessions[sessionID]; !exists {
		sm.sessions[sessionID] = newSessionState()
	}
}

func (sm *SessionManager) GetSession(sessionID string) (*SessionState, bool) {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()

	session, exists := sm.sessions[sessionID]
	return session, exists
}

func (sm *SessionManager) RemoveSession(ctx context.Context, session server.ClientSession) {
	sm.mutex.Lock()
	sessionID := session.SessionID()
	state, exists := sm.sessions[sessionID]
	delete(sm.sessions, sessionID)
	sm.mutex.Unlock()

	if !exists {
		return
	}

	// Clean up proxied clients outside of the main lock
	state.mutex.Lock()
	defer state.mutex.Unlock()

	for key, client := range state.proxiedClients {
		if err := client.Close(); err != nil {
			slog.Error("failed to close proxied client", "key", key, "error", err)
		}
	}
}

// GetProxiedClient retrieves a proxied client for the given datasource
func (sm *SessionManager) GetProxiedClient(ctx context.Context, datasourceType, datasourceUID string) (*ProxiedClient, error) {
	session := server.ClientSessionFromContext(ctx)
	if session == nil {
		return nil, fmt.Errorf("session not found in context")
	}

	state, exists := sm.GetSession(session.SessionID())
	if !exists {
		return nil, fmt.Errorf("session not found")
	}

	state.mutex.RLock()
	defer state.mutex.RUnlock()

	key := datasourceType + "_" + datasourceUID
	client, exists := state.proxiedClients[key]
	if !exists {
		// List available datasources to help with debugging
		var availableUIDs []string
		for _, c := range state.proxiedClients {
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
