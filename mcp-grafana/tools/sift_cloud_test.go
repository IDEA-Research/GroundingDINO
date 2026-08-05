//go:build cloud
// +build cloud

// This file contains cloud integration tests that run against a dedicated test instance
// at mcptests.grafana-dev.net. This instance is configured with a minimal setup on the Sift side:
//   - 2 test investigations
// These tests expect this configuration to exist and will skip if the required
// environment variables (GRAFANA_URL, GRAFANA_SERVICE_ACCOUNT_TOKEN or GRAFANA_API_KEY) are not set.
// The GRAFANA_API_KEY variable is deprecated.

package tools

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCloudSiftInvestigations(t *testing.T) {
	ctx := createCloudTestContext(t, "Sift", "GRAFANA_URL", "GRAFANA_API_KEY")

	// Test listing all investigations
	t.Run("list all investigations", func(t *testing.T) {
		result, err := listSiftInvestigations(ctx, ListSiftInvestigationsParams{})
		require.NoError(t, err, "Should not error when listing investigations")
		assert.NotNil(t, result, "Result should not be nil")
		assert.GreaterOrEqual(t, len(result), 1, "Should have at least one investigation")
	})

	// Test listing investigations with a limit
	t.Run("list investigations with limit", func(t *testing.T) {
		// Get the client
		client, err := siftClientFromContext(ctx)
		require.NoError(t, err, "Should not error when getting Sift client")

		// List investigations with a limit of 1
		investigations, err := client.listSiftInvestigations(ctx, 1)
		require.NoError(t, err, "Should not error when listing investigations with limit")
		assert.NotNil(t, investigations, "Investigations should not be nil")
		assert.LessOrEqual(t, len(investigations), 1, "Should have at most one investigation")

		// If there are investigations, verify their structure
		if len(investigations) > 0 {
			investigation := investigations[0]
			assert.NotEmpty(t, investigation.ID, "Investigation should have an ID")
			assert.NotEmpty(t, investigation.Name, "Investigation should have a name")
			assert.NotEmpty(t, investigation.TenantID, "Investigation should have a tenant ID")
		}
	})

	// Get an investigation ID from the list to test getting a specific investigation
	investigations, err := listSiftInvestigations(ctx, ListSiftInvestigationsParams{Limit: 10})
	require.NoError(t, err, "Should not error when listing investigations")
	require.NotEmpty(t, investigations, "Should have at least one investigation to test with")

	// Find an investigation with at least one analysis.
	var investigationID string
	for _, investigation := range investigations {
		if len(investigation.Analyses.Items) > 0 {
			investigationID = investigation.ID.String()
			break
		}
	}
	require.NotEmpty(t, investigationID, "Should have at least one investigation with at least one analysis")

	// Test getting a specific investigation
	t.Run("get specific investigation", func(t *testing.T) {
		result, err := getSiftInvestigation(ctx, GetSiftInvestigationParams{
			ID: investigationID,
		})
		require.NoError(t, err, "Should not error when getting specific investigation")
		assert.NotNil(t, result, "Result should not be nil")
		assert.Equal(t, investigationID, result.ID.String(), "Should return the correct investigation")

		// Verify all required fields are present
		assert.NotEmpty(t, result.Name, "Investigation should have a name")
		assert.NotEmpty(t, result.TenantID, "Investigation should have a tenant ID")
		assert.NotNil(t, result.GrafanaURL, "Investigation should have a Grafana URL")
		assert.NotNil(t, result.Status, "Investigation should have a status")
		assert.NotNil(t, result.FailureReason, "Investigation should have a failure reason")
	})

	// Test getting a non-existent investigation
	t.Run("get non-existent investigation", func(t *testing.T) {
		_, err := getSiftInvestigation(ctx, GetSiftInvestigationParams{
			ID: "00000000-0000-0000-0000-000000000000",
		})
		assert.NoError(t, err, "Should not error when getting non-existent investigation")
	})

	// Test getting analyses for an investigation
	t.Run("get analyses for investigation", func(t *testing.T) {
		// Get the investigation
		result, err := getSiftInvestigation(ctx, GetSiftInvestigationParams{
			ID: investigationID,
		})
		require.NoError(t, err, "Should not error when getting specific investigation")
		assert.NotNil(t, result, "Result should not be nil")

		// Get an analysis ID
		analysisID := result.Analyses.Items[0].ID

		// Get the analysis
		analysis, err := getSiftAnalysis(ctx, GetSiftAnalysisParams{
			InvestigationID: investigationID,
			AnalysisID:      analysisID.String(),
		})
		require.NoError(t, err, "Should not error when getting specific analysis")
		assert.NotNil(t, analysis, "Analysis should not be nil")

		// Verify all required fields are present
		assert.NotEmpty(t, analysis.Name, "Analysis should have a name")
		assert.NotEmpty(t, analysis.InvestigationID, "Analysis should have an investigation ID")
		assert.NotNil(t, analysis.Result, "Analysis should have a result")
	})

	t.Run("find error patterns", func(t *testing.T) {
		// Find error patterns
		analysis, err := findErrorPatternLogs(ctx, FindErrorPatternLogsParams{
			Name: "Test Sift",
			Labels: map[string]string{
				"namespace": "hosted-grafana",
				"cluster":   "dev-eu-west-2",
				"slug":      "mcptests",
			},
			Start: time.Now().Add(-5 * time.Minute),
			End:   time.Now(),
		})
		require.NoError(t, err, "Should not error when finding error patterns")
		assert.NotNil(t, analysis, "Result should not be nil")

		// Verify all required fields are present
		assert.NotEmpty(t, analysis.Name, "Analysis should have a name")
		assert.NotEmpty(t, analysis.InvestigationID, "Analysis should have an investigation ID")
		assert.NotEmpty(t, analysis.Result.Message, "Analysis  should have a message")
	})
}
