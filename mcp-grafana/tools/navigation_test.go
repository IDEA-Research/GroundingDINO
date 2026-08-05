package tools

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	mcpgrafana "github.com/grafana/mcp-grafana"
)

// Helper function to create string pointers
func stringPtr(s string) *string {
	return &s
}

func TestGenerateDeeplink(t *testing.T) {
	grafanaCfg := mcpgrafana.GrafanaConfig{
		URL: "http://localhost:3000",
	}
	ctx := mcpgrafana.WithGrafanaConfig(context.Background(), grafanaCfg)

	t.Run("Dashboard deeplink", func(t *testing.T) {
		params := GenerateDeeplinkParams{
			ResourceType: "dashboard",
			DashboardUID: stringPtr("abc123"),
		}

		result, err := generateDeeplink(ctx, params)
		require.NoError(t, err)
		assert.Equal(t, "http://localhost:3000/d/abc123", result)
	})

	t.Run("Panel deeplink", func(t *testing.T) {
		panelID := 5
		params := GenerateDeeplinkParams{
			ResourceType: "panel",
			DashboardUID: stringPtr("dash-123"),
			PanelID:      &panelID,
		}

		result, err := generateDeeplink(ctx, params)
		require.NoError(t, err)
		assert.Equal(t, "http://localhost:3000/d/dash-123?viewPanel=5", result)
	})

	t.Run("Explore deeplink", func(t *testing.T) {
		params := GenerateDeeplinkParams{
			ResourceType:  "explore",
			DatasourceUID: stringPtr("prometheus-uid"),
		}

		result, err := generateDeeplink(ctx, params)
		require.NoError(t, err)
		assert.Contains(t, result, "http://localhost:3000/explore?left=")
		assert.Contains(t, result, "prometheus-uid")
	})

	t.Run("With time range", func(t *testing.T) {
		params := GenerateDeeplinkParams{
			ResourceType: "dashboard",
			DashboardUID: stringPtr("abc123"),
			TimeRange: &TimeRange{
				From: "now-1h",
				To:   "now",
			},
		}

		result, err := generateDeeplink(ctx, params)
		require.NoError(t, err)
		assert.Contains(t, result, "http://localhost:3000/d/abc123")
		assert.Contains(t, result, "from=now-1h")
		assert.Contains(t, result, "to=now")
	})

	t.Run("With additional query params", func(t *testing.T) {
		params := GenerateDeeplinkParams{
			ResourceType: "dashboard",
			DashboardUID: stringPtr("abc123"),
			QueryParams: map[string]string{
				"var-datasource": "prometheus",
				"refresh":        "30s",
			},
		}

		result, err := generateDeeplink(ctx, params)
		require.NoError(t, err)
		assert.Contains(t, result, "http://localhost:3000/d/abc123")
		assert.Contains(t, result, "var-datasource=prometheus")
		assert.Contains(t, result, "refresh=30s")
	})

	t.Run("Error cases", func(t *testing.T) {
		emptyGrafanaCfg := mcpgrafana.GrafanaConfig{
			URL: "",
		}
		emptyCtx := mcpgrafana.WithGrafanaConfig(context.Background(), emptyGrafanaCfg)
		params := GenerateDeeplinkParams{
			ResourceType: "dashboard",
			DashboardUID: stringPtr("abc123"),
		}
		_, err := generateDeeplink(emptyCtx, params)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "grafana url not configured")

		params.ResourceType = "unsupported"
		_, err = generateDeeplink(ctx, params)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported resource type")

		// Test missing dashboardUid for dashboard
		params = GenerateDeeplinkParams{
			ResourceType: "dashboard",
		}
		_, err = generateDeeplink(ctx, params)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "dashboardUid is required")

		// Test missing dashboardUid for panel
		params = GenerateDeeplinkParams{
			ResourceType: "panel",
		}
		_, err = generateDeeplink(ctx, params)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "dashboardUid is required")

		// Test missing panelId for panel
		params = GenerateDeeplinkParams{
			ResourceType: "panel",
			DashboardUID: stringPtr("dash-123"),
		}
		_, err = generateDeeplink(ctx, params)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "panelId is required")

		// Test missing datasourceUid for explore
		params = GenerateDeeplinkParams{
			ResourceType: "explore",
		}
		_, err = generateDeeplink(ctx, params)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "datasourceUid is required")
	})
}
