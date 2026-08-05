// Requires a Cloud or other Grafana instance with Grafana Incident available,
// with a Prometheus datasource provisioned.
//go:build cloud
// +build cloud

// This file contains cloud integration tests that run against a dedicated test instance
// at mcptests.grafana-dev.net. This instance is configured with a minimal setup on the Incident side
// with two incidents created, one minor and one major, and both of them resolved.
// These tests expect this configuration to exist and will skip if the required
// environment variables (GRAFANA_URL, GRAFANA_SERVICE_ACCOUNT_TOKEN or GRAFANA_API_KEY) are not set.
// The GRAFANA_API_KEY variable is deprecated.

package tools

import (
	"testing"

	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCloudIncidentTools(t *testing.T) {
	t.Run("list incidents", func(t *testing.T) {
		ctx := createCloudTestContext(t, "Incident", "GRAFANA_URL", "GRAFANA_API_KEY")
		ctx = mcpgrafana.ExtractIncidentClientFromEnv(ctx)

		result, err := listIncidents(ctx, ListIncidentsParams{
			Limit: 1,
		})
		require.NoError(t, err)
		assert.NotNil(t, result, "Result should not be nil")
		assert.NotNil(t, result.IncidentPreviews, "IncidentPreviews should not be nil")
		assert.LessOrEqual(t, len(result.IncidentPreviews), 1, "Should not return more incidents than the limit")
	})

	t.Run("get incident by ID", func(t *testing.T) {
		ctx := createCloudTestContext(t, "Incident", "GRAFANA_URL", "GRAFANA_API_KEY")
		ctx = mcpgrafana.ExtractIncidentClientFromEnv(ctx)
		result, err := getIncident(ctx, GetIncidentParams{
			ID: "1",
		})
		require.NoError(t, err)
		assert.NotNil(t, result, "Result should not be nil")
		assert.Equal(t, "1", result.IncidentID, "Should return the requested incident ID")
		assert.NotEmpty(t, result.Title, "Incident should have a title")
		assert.NotEmpty(t, result.Status, "Incident should have a status")
	})
}
