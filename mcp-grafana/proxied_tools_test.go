package mcpgrafana

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/assert"
)

func TestSessionStateRaceConditions(t *testing.T) {
	t.Run("concurrent initialization with sync.Once is safe", func(t *testing.T) {
		state := newSessionState()

		var initCounter int32
		var wg sync.WaitGroup

		// Launch 100 goroutines that all try to initialize at once
		const numGoroutines = 100
		wg.Add(numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func() {
				defer wg.Done()
				state.initOnce.Do(func() {
					// Simulate initialization work
					atomic.AddInt32(&initCounter, 1)
					time.Sleep(10 * time.Millisecond) // Simulate some work
					state.mutex.Lock()
					state.proxiedToolsInitialized = true
					state.mutex.Unlock()
				})
			}()
		}

		wg.Wait()

		// Verify initialization happened exactly once
		assert.Equal(t, int32(1), atomic.LoadInt32(&initCounter),
			"Initialization should run exactly once despite 100 concurrent calls")
		assert.True(t, state.proxiedToolsInitialized)
	})

	t.Run("concurrent reads and writes with mutex protection", func(t *testing.T) {
		state := newSessionState()
		var wg sync.WaitGroup

		// Writer goroutines
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				state.mutex.Lock()
				key := "tempo_" + string(rune('a'+id))
				state.proxiedClients[key] = &ProxiedClient{
					DatasourceUID:  key,
					DatasourceName: "Test " + key,
					DatasourceType: "tempo",
				}
				state.mutex.Unlock()
			}(i)
		}

		// Reader goroutines
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				state.mutex.RLock()
				_ = len(state.proxiedClients)
				state.mutex.RUnlock()
			}()
		}

		wg.Wait()

		// Verify all writes succeeded
		state.mutex.RLock()
		count := len(state.proxiedClients)
		state.mutex.RUnlock()

		assert.Equal(t, 10, count, "All 10 clients should be stored")
	})

	t.Run("concurrent tool registration is safe", func(t *testing.T) {
		state := newSessionState()
		var wg sync.WaitGroup

		// Multiple goroutines trying to register tools
		const numGoroutines = 50
		wg.Add(numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(id int) {
				defer wg.Done()
				state.mutex.Lock()
				toolName := "tempo_tool-" + string(rune('a'+id%26))
				if state.toolToDatasources[toolName] == nil {
					state.toolToDatasources[toolName] = []string{}
				}
				state.toolToDatasources[toolName] = append(
					state.toolToDatasources[toolName],
					"datasource_"+string(rune('a'+id%26)),
				)
				state.mutex.Unlock()
			}(i)
		}

		wg.Wait()

		// Verify the tool mappings exist
		state.mutex.RLock()
		defer state.mutex.RUnlock()
		assert.Greater(t, len(state.toolToDatasources), 0, "Should have tool mappings")
	})
}

func TestSessionManagerConcurrency(t *testing.T) {
	t.Run("concurrent session creation is safe", func(t *testing.T) {
		sm := NewSessionManager()
		var wg sync.WaitGroup

		// Create many sessions concurrently
		const numSessions = 100
		wg.Add(numSessions)

		for i := 0; i < numSessions; i++ {
			go func(id int) {
				defer wg.Done()
				sessionID := "session-" + string(rune('a'+id%26)) + "-" + string(rune('0'+id/26))
				mockSession := &mockClientSession{id: sessionID}
				sm.CreateSession(context.Background(), mockSession)
			}(i)
		}

		wg.Wait()

		// Verify all sessions were created
		sm.mutex.RLock()
		count := len(sm.sessions)
		sm.mutex.RUnlock()

		assert.Equal(t, numSessions, count, "All sessions should be created")
	})

	t.Run("concurrent get and remove is safe", func(t *testing.T) {
		sm := NewSessionManager()

		// Pre-populate sessions
		for i := 0; i < 50; i++ {
			sessionID := "session-" + string(rune('a'+i%26))
			mockSession := &mockClientSession{id: sessionID}
			sm.CreateSession(context.Background(), mockSession)
		}

		var wg sync.WaitGroup

		// Readers
		for i := 0; i < 50; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				sessionID := "session-" + string(rune('a'+id%26))
				_, _ = sm.GetSession(sessionID)
			}(i)
		}

		// Writers (removers)
		for i := 0; i < 25; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				sessionID := "session-" + string(rune('a'+id%26))
				mockSession := &mockClientSession{id: sessionID}
				sm.RemoveSession(context.Background(), mockSession)
			}(i)
		}

		wg.Wait()

		// Test passed if no race conditions occurred
	})
}

func TestInitOncePattern(t *testing.T) {
	t.Run("verify sync.Once guarantees single execution", func(t *testing.T) {
		var once sync.Once
		var counter int32
		var wg sync.WaitGroup

		// Simulate what happens in InitializeAndRegisterProxiedTools
		initFunc := func() {
			atomic.AddInt32(&counter, 1)
			// Simulate expensive initialization
			time.Sleep(50 * time.Millisecond)
		}

		// Launch many concurrent calls
		for i := 0; i < 1000; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				once.Do(initFunc)
			}()
		}

		wg.Wait()

		assert.Equal(t, int32(1), atomic.LoadInt32(&counter),
			"sync.Once should guarantee function runs exactly once")
	})

	t.Run("sync.Once with different functions only runs first", func(t *testing.T) {
		var once sync.Once
		var result string
		var mu sync.Mutex

		once.Do(func() {
			mu.Lock()
			result = "first"
			mu.Unlock()
		})

		once.Do(func() {
			mu.Lock()
			result = "second"
			mu.Unlock()
		})

		mu.Lock()
		finalResult := result
		mu.Unlock()

		assert.Equal(t, "first", finalResult, "Only first function should execute")
	})
}

func TestProxiedToolsInitializationFlow(t *testing.T) {
	t.Run("initialization state transitions are correct", func(t *testing.T) {
		state := newSessionState()

		// Initial state
		assert.False(t, state.proxiedToolsInitialized)
		assert.Empty(t, state.proxiedClients)
		assert.Empty(t, state.proxiedTools)

		// Simulate initialization
		state.initOnce.Do(func() {
			state.mutex.Lock()
			state.proxiedToolsInitialized = true
			state.proxiedClients["tempo_test"] = &ProxiedClient{
				DatasourceUID:  "test",
				DatasourceName: "Test",
				DatasourceType: "tempo",
			}
			state.mutex.Unlock()
		})

		// Verify state after initialization
		state.mutex.RLock()
		initialized := state.proxiedToolsInitialized
		clientCount := len(state.proxiedClients)
		state.mutex.RUnlock()

		assert.True(t, initialized)
		assert.Equal(t, 1, clientCount)
	})

	t.Run("multiple sessions maintain separate state", func(t *testing.T) {
		sm := NewSessionManager()

		// Create two sessions
		session1 := &mockClientSession{id: "session-1"}
		session2 := &mockClientSession{id: "session-2"}

		sm.CreateSession(context.Background(), session1)
		sm.CreateSession(context.Background(), session2)

		state1, _ := sm.GetSession("session-1")
		state2, _ := sm.GetSession("session-2")

		// Initialize only session1
		state1.initOnce.Do(func() {
			state1.mutex.Lock()
			state1.proxiedToolsInitialized = true
			state1.mutex.Unlock()
		})

		// Verify states are independent
		assert.True(t, state1.proxiedToolsInitialized)
		assert.False(t, state2.proxiedToolsInitialized)
		assert.NotSame(t, state1, state2)
	})
}

func TestRaceConditionDemonstration(t *testing.T) {
	t.Run("old pattern WITHOUT sync.Once would have race condition", func(t *testing.T) {
		// This test demonstrates what WOULD happen with the old mutex-check pattern
		state := newSessionState()

		var discoveryCallCount int32
		var wg sync.WaitGroup

		// Simulate the OLD pattern (mutex check, unlock, then do work)
		oldPatternInitialize := func() {
			state.mutex.Lock()
			// Check if already initialized
			if state.proxiedToolsInitialized {
				state.mutex.Unlock()
				return
			}
			alreadyDiscovered := state.proxiedToolsInitialized
			state.mutex.Unlock() // ❌ OLD PATTERN: Unlock before expensive work

			if !alreadyDiscovered {
				// Simulate discovery work that should only happen once
				atomic.AddInt32(&discoveryCallCount, 1)
				time.Sleep(10 * time.Millisecond) // Simulate expensive operation

				state.mutex.Lock()
				state.proxiedToolsInitialized = true
				state.mutex.Unlock()
			}
		}

		// Launch concurrent initializations
		const numGoroutines = 10
		wg.Add(numGoroutines)
		for i := 0; i < numGoroutines; i++ {
			go func() {
				defer wg.Done()
				oldPatternInitialize()
			}()
		}
		wg.Wait()

		// With the old pattern, multiple goroutines can get past the check
		// and call discovery multiple times
		count := atomic.LoadInt32(&discoveryCallCount)
		if count > 1 {
			t.Logf("OLD PATTERN: Discovery called %d times (race condition!)", count)
		}
		// We can't assert > 1 reliably because timing matters, but this demonstrates the problem
	})

	t.Run("new pattern WITH sync.Once prevents race condition", func(t *testing.T) {
		// This test demonstrates the NEW pattern with sync.Once
		state := newSessionState()

		var discoveryCallCount int32
		var wg sync.WaitGroup

		// NEW pattern: sync.Once guarantees single execution
		newPatternInitialize := func() {
			state.initOnce.Do(func() {
				// Simulate discovery work that should only happen once
				atomic.AddInt32(&discoveryCallCount, 1)
				time.Sleep(10 * time.Millisecond) // Simulate expensive operation

				state.mutex.Lock()
				state.proxiedToolsInitialized = true
				state.mutex.Unlock()
			})
		}

		// Launch concurrent initializations
		const numGoroutines = 10
		wg.Add(numGoroutines)
		for i := 0; i < numGoroutines; i++ {
			go func() {
				defer wg.Done()
				newPatternInitialize()
			}()
		}
		wg.Wait()

		// With sync.Once, discovery is guaranteed to run exactly once
		count := atomic.LoadInt32(&discoveryCallCount)
		assert.Equal(t, int32(1), count, "NEW PATTERN: Discovery must be called exactly once")
	})
}

func TestRaceDetector(t *testing.T) {
	// This test is primarily valuable when run with -race flag
	t.Run("stress test with race detector", func(t *testing.T) {

		sm := NewSessionManager()
		var wg sync.WaitGroup

		// Create a mix of operations happening concurrently
		for i := 0; i < 20; i++ {
			sessionID := "stress-session-" + string(rune('a'+i%10))

			// Create session
			wg.Add(1)
			go func(sid string) {
				defer wg.Done()
				mockSession := &mockClientSession{id: sid}
				sm.CreateSession(context.Background(), mockSession)
			}(sessionID)

			// Initialize session state
			wg.Add(1)
			go func(sid string) {
				defer wg.Done()
				time.Sleep(time.Millisecond) // Let creation happen first
				state, exists := sm.GetSession(sid)
				if exists {
					state.initOnce.Do(func() {
						state.mutex.Lock()
						state.proxiedToolsInitialized = true
						state.mutex.Unlock()
					})
				}
			}(sessionID)

			// Read session state
			wg.Add(1)
			go func(sid string) {
				defer wg.Done()
				time.Sleep(2 * time.Millisecond)
				state, exists := sm.GetSession(sid)
				if exists {
					state.mutex.RLock()
					_ = state.proxiedToolsInitialized
					state.mutex.RUnlock()
				}
			}(sessionID)
		}

		wg.Wait()

		// If we get here without race detector warnings, we're good
		t.Log("Stress test completed without race conditions")
	})
}

// mockClientSession implements server.ClientSession for testing
type mockClientSession struct {
	id            string
	notifChannel  chan mcp.JSONRPCNotification
	isInitialized bool
}

func (m *mockClientSession) SessionID() string {
	return m.id
}

func (m *mockClientSession) NotificationChannel() chan<- mcp.JSONRPCNotification {
	if m.notifChannel == nil {
		m.notifChannel = make(chan mcp.JSONRPCNotification, 10)
	}
	return m.notifChannel
}

func (m *mockClientSession) Initialize() {
	m.isInitialized = true
}

func (m *mockClientSession) Initialized() bool {
	return m.isInitialized
}
