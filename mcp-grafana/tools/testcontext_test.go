//go:build integration

package tools

import (
	"context"
	"fmt"
	"net/url"
	"os"

	"github.com/go-openapi/strfmt"
	"github.com/grafana/grafana-openapi-client-go/client"
	mcpgrafana "github.com/grafana/mcp-grafana"
)

// newTestContext creates a new context with the Grafana URL and service account token
// from the environment variables GRAFANA_URL and GRAFANA_SERVICE_ACCOUNT_TOKEN (or deprecated GRAFANA_API_KEY).
func newTestContext() context.Context {
	cfg := client.DefaultTransportConfig()
	cfg.Host = "localhost:3000"
	cfg.Schemes = []string{"http"}
	// Extract transport config from env vars, and set it on the context.
	if u, ok := os.LookupEnv("GRAFANA_URL"); ok {
		url, err := url.Parse(u)
		if err != nil {
			panic(fmt.Errorf("invalid %s: %w", "GRAFANA_URL", err))
		}
		cfg.Host = url.Host
		// The Grafana client will always prefer HTTPS even if the URL is HTTP,
		// so we need to limit the schemes to HTTP if the URL is HTTP.
		if url.Scheme == "http" {
			cfg.Schemes = []string{"http"}
		}
	}

	// Check for the new service account token environment variable first
	if apiKey := os.Getenv("GRAFANA_SERVICE_ACCOUNT_TOKEN"); apiKey != "" {
		cfg.APIKey = apiKey
	} else if apiKey := os.Getenv("GRAFANA_API_KEY"); apiKey != "" {
		// Fall back to the deprecated API key environment variable
		cfg.APIKey = apiKey
	} else {
		cfg.BasicAuth = url.UserPassword("admin", "admin")
	}

	client := client.NewHTTPClientWithConfig(strfmt.Default, cfg)

	grafanaCfg := mcpgrafana.GrafanaConfig{
		Debug:     true,
		URL:       "http://localhost:3000",
		APIKey:    cfg.APIKey,
		BasicAuth: cfg.BasicAuth,
	}

	ctx := mcpgrafana.WithGrafanaConfig(context.Background(), grafanaCfg)
	return mcpgrafana.WithGrafanaClient(ctx, client)
}
