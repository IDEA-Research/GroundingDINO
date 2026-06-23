package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/grafana/mcp-grafana/tools"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	// Example 1: Basic TLS configuration with skip verify (for testing)
	fmt.Println("Example 1: Basic TLS configuration with skip verify")
	basicTLSExample()

	// Example 2: Full mTLS configuration with client certificates
	fmt.Println("\nExample 2: Full mTLS configuration")
	fullTLSExample()

	// Example 3: Running an MCP server with TLS support
	fmt.Println("\nExample 3: MCP server with TLS support")
	if len(os.Args) > 1 && os.Args[1] == "run-server" {
		runServerWithTLS()
	} else {
		fmt.Println("Use 'go run tls_example.go run-server' to actually start the server")
		showServerExample()
	}
}

func basicTLSExample() {
	// Create a TLS config that skips certificate verification
	// This is useful for testing against self-signed certificates
	tlsConfig := &mcpgrafana.TLSConfig{SkipVerify: true}

	// Create a Grafana config with TLS support
	grafanaConfig := mcpgrafana.GrafanaConfig{
		Debug:     true,
		TLSConfig: tlsConfig,
	}

	// Create a context function that includes TLS configuration
	contextFunc := mcpgrafana.ComposedStdioContextFunc(grafanaConfig)

	// Test the context function
	ctx := contextFunc(context.Background())

	// Verify the configuration is applied
	retrievedConfig := mcpgrafana.GrafanaConfigFromContext(ctx)
	if retrievedConfig.TLSConfig != nil {
		fmt.Printf("✓ TLS configuration applied: SkipVerify=%v\n", retrievedConfig.TLSConfig.SkipVerify)
	}

	fmt.Printf("✓ Debug mode enabled: %v\n", retrievedConfig.Debug)
}

func fullTLSExample() {
	// Example paths for certificate files
	// In a real scenario, these would point to actual certificate files
	certFile := "/path/to/client.crt"
	keyFile := "/path/to/client.key"
	caFile := "/path/to/ca.crt"

	// Create TLS config with client certificates and CA verification
	tlsConfig := &mcpgrafana.TLSConfig{
		CertFile: certFile,
		KeyFile:  keyFile,
		CAFile:   caFile,
	}

	// Create Grafana config with TLS support
	grafanaConfig := mcpgrafana.GrafanaConfig{
		Debug:     false,
		TLSConfig: tlsConfig,
	}

	fmt.Printf("✓ TLS configuration created:\n")
	fmt.Printf("  - Client cert: %s\n", tlsConfig.CertFile)
	fmt.Printf("  - Client key: %s\n", tlsConfig.KeyFile)
	fmt.Printf("  - CA file: %s\n", tlsConfig.CAFile)
	fmt.Printf("  - Skip verify: %v\n", tlsConfig.SkipVerify)
	fmt.Printf("  - Debug mode: %v\n", grafanaConfig.Debug)

	// Create context functions for different transport types
	stdioFunc := mcpgrafana.ComposedStdioContextFunc(grafanaConfig)
	sseFunc := mcpgrafana.ComposedSSEContextFunc(grafanaConfig)
	httpFunc := mcpgrafana.ComposedHTTPContextFunc(grafanaConfig)

	fmt.Printf("✓ Context functions created for all transport types\n")

	_ = stdioFunc
	_ = sseFunc
	_ = httpFunc
}

func showServerExample() {
	fmt.Println("Example MCP server configuration with TLS:")
	fmt.Println(`// Create TLS configuration
tlsConfig := &mcpgrafana.TLSConfig{
    CertFile: "/path/to/client.crt",
    KeyFile:  "/path/to/client.key",
    CAFile:   "/path/to/ca.crt",
}

// Create Grafana configuration
grafanaConfig := mcpgrafana.GrafanaConfig{
    Debug: true,
    TLSConfig: tlsConfig,
}

// Create MCP server
s := server.NewMCPServer("mcp-grafana", "1.0.0")

// Add tools
tools.AddSearchTools(s)
tools.AddDatasourceTools(s)
// ... add other tools as needed

// Create stdio server with TLS support
srv := server.NewStdioServer(s)
srv.SetContextFunc(mcpgrafana.ComposedStdioContextFunc(grafanaConfig))

// Start server
srv.Listen(ctx, os.Stdin, os.Stdout)`)
}

func runServerWithTLS() {
	// Set up environment variables (in practice, these would be set externally)
	if os.Getenv("GRAFANA_URL") == "" {
		if err := os.Setenv("GRAFANA_URL", "https://localhost:3000"); err != nil {
			log.Printf("Failed to set GRAFANA_URL: %v", err)
		}
	}
	// Check for service account token first, then fall back to deprecated API key
	if os.Getenv("GRAFANA_SERVICE_ACCOUNT_TOKEN") == "" {
		if os.Getenv("GRAFANA_API_KEY") == "" {
			fmt.Println("Warning: Neither GRAFANA_SERVICE_ACCOUNT_TOKEN nor GRAFANA_API_KEY is set")
		} else {
			fmt.Println("Warning: GRAFANA_API_KEY is deprecated, please use GRAFANA_SERVICE_ACCOUNT_TOKEN instead")
		}
	}

	// Create TLS configuration that skips verification for demo purposes
	// In production, you would use real certificates
	tlsConfig := &mcpgrafana.TLSConfig{SkipVerify: true}
	grafanaConfig := mcpgrafana.GrafanaConfig{
		Debug:     true,
		TLSConfig: tlsConfig,
	}

	// Create MCP server
	s := server.NewMCPServer("mcp-grafana-tls-example", "1.0.0")

	// Add some basic tools
	tools.AddSearchTools(s)
	tools.AddDatasourceTools(s)
	tools.AddDashboardTools(s, false) // Read-only mode (no write tools)

	// Create stdio server with TLS-enabled context function
	srv := server.NewStdioServer(s)
	srv.SetContextFunc(mcpgrafana.ComposedStdioContextFunc(grafanaConfig))

	fmt.Printf("Starting MCP Grafana server with TLS support...\n")
	fmt.Printf("Grafana URL: %s\n", os.Getenv("GRAFANA_URL"))
	fmt.Printf("TLS Skip Verify: %v\n", tlsConfig.SkipVerify)

	// Start the server
	ctx := context.Background()
	if err := srv.Listen(ctx, os.Stdin, os.Stdout); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

// Example of creating custom HTTP clients with TLS configuration
func customClientExample() { //nolint:unused // Example function for documentation
	ctx := context.Background()

	// Add Grafana configuration to context
	tlsConfig := &mcpgrafana.TLSConfig{
		CertFile: "/path/to/cert.pem",
		KeyFile:  "/path/to/key.pem",
		CAFile:   "/path/to/ca.pem",
	}
	config := mcpgrafana.GrafanaConfig{
		TLSConfig: tlsConfig,
	}
	ctx = mcpgrafana.WithGrafanaConfig(ctx, config)
	_ = ctx // Use ctx to avoid ineffectual assignment warning

	// Create custom HTTP transport with TLS
	transport, err := tlsConfig.HTTPTransport(http.DefaultTransport.(*http.Transport))
	if err != nil {
		log.Fatalf("Failed to create transport: %v", err)
	}

	// Use the transport in your HTTP client
	_ = transport
	fmt.Println("✓ Custom HTTP transport created with TLS configuration")
}
