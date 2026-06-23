//go:build cloud
// +build cloud

// This file contains cloud integration tests that run against a dedicated test instance
// connected to a Grafana instance at (ASSERTS_GRAFANA_URL, ASSERTS_GRAFANA_SERVICE_ACCOUNT_TOKEN or ASSERTS_GRAFANA_API_KEY).
// These tests expect this configuration to exist and will skip if the required
// environment variables are not set. The ASSERTS_GRAFANA_API_KEY variable is deprecated.

package tools

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAssertsCloudIntegration(t *testing.T) {
	ctx := createCloudTestContext(t, "Asserts", "ASSERTS_GRAFANA_URL", "ASSERTS_GRAFANA_API_KEY")

	t.Run("get assertions", func(t *testing.T) {
		// Set up time range for the last hour
		endTime := time.Now()
		startTime := endTime.Add(-24 * time.Hour)

		// Test parameters for a known service in the environment
		params := GetAssertionsParams{
			StartTime:  startTime,
			EndTime:    endTime,
			EntityType: "Service", // Adjust these values based on your actual environment
			EntityName: "model-builder",
			Env:        "dev-us-central-0",
			Namespace:  "asserts",
		}

		// Get assertions from the real Grafana instance
		result, err := getAssertions(ctx, params)
		require.NoError(t, err, "Failed to get assertions from Grafana")
		assert.NotEmpty(t, result, "Expected non-empty assertions result")

		// Basic validation of the response structure
		assert.Contains(t, result, "summaries", "Response should contain a summaries field")
	})
}
