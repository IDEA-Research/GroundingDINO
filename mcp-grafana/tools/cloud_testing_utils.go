//go:build cloud
// +build cloud

package tools

import (
	"context"
	"os"
	"strings"
	"testing"

	mcpgrafana "github.com/grafana/mcp-grafana"
)

// createCloudTestContext creates a context with a Grafana URL, Grafana service account token and
// Grafana client for cloud integration tests.
// The test will be skipped if required environment variables are not set.
// testName is used to customize the skip message (e.g. "OnCall", "Sift", "Incident")
// urlEnv and apiKeyEnv specify the environment variable names for the Grafana URL and API key (deprecated).
// The function will automatically try the new SERVICE_ACCOUNT_TOKEN pattern first, then fall back to API_KEY.
func createCloudTestContext(t *testing.T, testName, urlEnv, apiKeyEnv string) context.Context {
	ctx := context.Background()

	grafanaURL := os.Getenv(urlEnv)
	if grafanaURL == "" {
		t.Skipf("%s environment variable not set, skipping cloud %s integration tests", urlEnv, testName)
	}

	// Try the new service account token environment variable first
	serviceAccountTokenEnv := strings.Replace(apiKeyEnv, "API_KEY", "SERVICE_ACCOUNT_TOKEN", 1)
	grafanaApiKey := os.Getenv(serviceAccountTokenEnv)

	if grafanaApiKey == "" {
		// Fall back to the deprecated API key environment variable
		grafanaApiKey = os.Getenv(apiKeyEnv)
		if grafanaApiKey != "" {
			t.Logf("Warning: %s is deprecated, please use %s instead", apiKeyEnv, serviceAccountTokenEnv)
		}
	}

	if grafanaApiKey == "" {
		t.Skipf("Neither %s nor %s environment variables are set, skipping cloud %s integration tests", serviceAccountTokenEnv, apiKeyEnv, testName)
	}

	client := mcpgrafana.NewGrafanaClient(ctx, grafanaURL, grafanaApiKey, nil, 0)

	config := mcpgrafana.GrafanaConfig{
		URL:    grafanaURL,
		APIKey: grafanaApiKey,
	}
	ctx = mcpgrafana.WithGrafanaConfig(ctx, config)
	ctx = mcpgrafana.WithGrafanaClient(ctx, client)

	return ctx
}
