package mcpgrafana

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTLSConfig_CreateTLSConfig(t *testing.T) {
	t.Run("nil config returns nil", func(t *testing.T) {
		var config *TLSConfig
		tlsCfg, err := config.CreateTLSConfig()
		assert.NoError(t, err)
		assert.Nil(t, tlsCfg)
	})

	t.Run("skip verify only", func(t *testing.T) {
		config := &TLSConfig{SkipVerify: true}
		tlsCfg, err := config.CreateTLSConfig()
		assert.NoError(t, err)
		require.NotNil(t, tlsCfg)
		assert.True(t, tlsCfg.InsecureSkipVerify)
		assert.Empty(t, tlsCfg.Certificates)
		assert.Nil(t, tlsCfg.RootCAs)
	})

	t.Run("invalid cert file", func(t *testing.T) {
		config := &TLSConfig{
			CertFile: "nonexistent.pem",
			KeyFile:  "nonexistent.key",
		}
		_, err := config.CreateTLSConfig()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to load client certificate")
	})

	t.Run("invalid CA file", func(t *testing.T) {
		config := &TLSConfig{
			CAFile: "nonexistent-ca.pem",
		}
		_, err := config.CreateTLSConfig()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to read CA certificate")
	})
}

func TestHTTPTransport(t *testing.T) {
	t.Run("nil TLS config", func(t *testing.T) {
		var tlsConfig *TLSConfig
		transport, err := tlsConfig.HTTPTransport(http.DefaultTransport.(*http.Transport))
		assert.NoError(t, err)
		assert.NotNil(t, transport)

		// Should be default transport clone
		httpTransport, ok := transport.(*http.Transport)
		require.True(t, ok)
		assert.NotNil(t, httpTransport)
	})

	t.Run("skip verify config", func(t *testing.T) {
		tlsConfig := &TLSConfig{SkipVerify: true}
		transport, err := tlsConfig.HTTPTransport(http.DefaultTransport.(*http.Transport))
		assert.NoError(t, err)
		require.NotNil(t, transport)

		httpTransport, ok := transport.(*http.Transport)
		require.True(t, ok)
		require.NotNil(t, httpTransport.TLSClientConfig)
		assert.True(t, httpTransport.TLSClientConfig.InsecureSkipVerify)
	})

	t.Run("invalid TLS config", func(t *testing.T) {
		tlsConfig := &TLSConfig{
			CertFile: "nonexistent.pem",
			KeyFile:  "nonexistent.key",
		}
		_, err := tlsConfig.HTTPTransport(http.DefaultTransport.(*http.Transport))
		assert.Error(t, err)
	})
}

// mockRoundTripper is a mock implementation of http.RoundTripper for testing
type mockRoundTripper struct {
	capturedRequest *http.Request
	response        *http.Response
	err             error
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	m.capturedRequest = req
	if m.response != nil {
		return m.response, m.err
	}
	// Return a default successful response
	return &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       http.NoBody,
	}, m.err
}

func TestUserAgentTransport(t *testing.T) {
	tests := []struct {
		name              string
		userAgent         string
		existingUserAgent string
		expectedUserAgent string
	}{
		{
			name:              "sets user agent when not present",
			userAgent:         "mcp-grafana/1.0.0",
			existingUserAgent: "",
			expectedUserAgent: "mcp-grafana/1.0.0",
		},
		{
			name:              "does not override existing user agent",
			userAgent:         "mcp-grafana/1.0.0",
			existingUserAgent: "existing-client/2.0.0",
			expectedUserAgent: "existing-client/2.0.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock round tripper
			mockRT := &mockRoundTripper{}

			// Create user agent transport
			transport := &UserAgentTransport{
				rt:        mockRT,
				UserAgent: tt.userAgent,
			}

			// Create request
			req, err := http.NewRequest("GET", "http://example.com", nil)
			require.NoError(t, err)

			// Set existing user agent if specified
			if tt.existingUserAgent != "" {
				req.Header.Set("User-Agent", tt.existingUserAgent)
			}

			// Make request through transport
			_, err = transport.RoundTrip(req)
			require.NoError(t, err)

			// Verify user agent header
			assert.Equal(t, tt.expectedUserAgent, mockRT.capturedRequest.Header.Get("User-Agent"))
		})
	}
}

func TestVersion(t *testing.T) {
	version := Version()
	assert.NotEmpty(t, version)
	// Version should be either "(devel)" for development builds or a proper version
	assert.True(t, version == "(devel)" || len(version) > 0)
}

func TestUserAgent(t *testing.T) {
	userAgent := UserAgent()
	assert.Contains(t, userAgent, "mcp-grafana/")
	assert.NotEqual(t, "mcp-grafana/", userAgent) // Should have version appended

	// Should match the pattern mcp-grafana/{version}
	version := Version()
	expected := fmt.Sprintf("mcp-grafana/%s", version)
	assert.Equal(t, expected, userAgent)
}

func TestNewUserAgentTransport(t *testing.T) {
	t.Run("with explicit user agent", func(t *testing.T) {
		mockRT := &mockRoundTripper{}
		userAgent := "test-agent/1.0.0"

		transport := NewUserAgentTransport(mockRT, userAgent)

		assert.Equal(t, mockRT, transport.rt)
		assert.Equal(t, userAgent, transport.UserAgent)
	})

	t.Run("with default user agent", func(t *testing.T) {
		mockRT := &mockRoundTripper{}

		transport := NewUserAgentTransport(mockRT)

		assert.Equal(t, mockRT, transport.rt)
		assert.Equal(t, UserAgent(), transport.UserAgent)
		assert.Contains(t, transport.UserAgent, "mcp-grafana/")
	})
}

func TestNewUserAgentTransportWithNilRoundTripper(t *testing.T) {
	t.Run("with explicit user agent", func(t *testing.T) {
		userAgent := "test-agent/1.0.0"

		transport := NewUserAgentTransport(nil, userAgent)

		assert.Equal(t, http.DefaultTransport, transport.rt)
		assert.Equal(t, userAgent, transport.UserAgent)
	})

	t.Run("with default user agent", func(t *testing.T) {
		transport := NewUserAgentTransport(nil)

		assert.Equal(t, http.DefaultTransport, transport.rt)
		assert.Equal(t, UserAgent(), transport.UserAgent)
		assert.Contains(t, transport.UserAgent, "mcp-grafana/")
	})
}
