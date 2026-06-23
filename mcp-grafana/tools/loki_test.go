//go:build integration

package tools

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLokiTools(t *testing.T) {
	t.Run("list loki label names", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listLokiLabelNames(ctx, ListLokiLabelNamesParams{
			DatasourceUID: "loki",
		})
		require.NoError(t, err)
		assert.Len(t, result, 3)
	})

	t.Run("get loki label values", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listLokiLabelValues(ctx, ListLokiLabelValuesParams{
			DatasourceUID: "loki",
			LabelName:     "container",
		})
		require.NoError(t, err)
		assert.NotEmpty(t, result, "Should have at least one container label value")
	})

	t.Run("query loki stats", func(t *testing.T) {
		ctx := newTestContext()
		result, err := queryLokiStats(ctx, QueryLokiStatsParams{
			DatasourceUID: "loki",
			LogQL:         `{container="grafana"}`,
		})
		require.NoError(t, err)
		assert.NotNil(t, result, "Should return a result")

		// We can't assert on specific values as they will vary,
		// but we can check that the structure is correct
		assert.GreaterOrEqual(t, result.Streams, 0, "Should have a valid streams count")
		assert.GreaterOrEqual(t, result.Chunks, 0, "Should have a valid chunks count")
		assert.GreaterOrEqual(t, result.Entries, 0, "Should have a valid entries count")
		assert.GreaterOrEqual(t, result.Bytes, 0, "Should have a valid bytes count")
	})

	t.Run("query loki logs", func(t *testing.T) {
		ctx := newTestContext()
		result, err := queryLokiLogs(ctx, QueryLokiLogsParams{
			DatasourceUID: "loki",
			LogQL:         `{container=~".+"}`,
			Limit:         10,
		})
		require.NoError(t, err)

		// We can't assert on specific log content as it will vary,
		// but we can check that the structure is correct
		// If we got logs, check that they have the expected structure
		for _, entry := range result {
			assert.NotEmpty(t, entry.Timestamp, "Log entry should have a timestamp")
			assert.NotNil(t, entry.Labels, "Log entry should have labels")
		}
	})

	t.Run("query loki logs with no results", func(t *testing.T) {
		ctx := newTestContext()
		// Use a query that's unlikely to match any logs
		result, err := queryLokiLogs(ctx, QueryLokiLogsParams{
			DatasourceUID: "loki",
			LogQL:         `{container="non-existent-container-name-123456789"}`,
			Limit:         10,
		})
		require.NoError(t, err)

		// Should return an empty slice, not nil
		assert.NotNil(t, result, "Empty results should be an empty slice, not nil")
		assert.Equal(t, 0, len(result), "Empty results should have length 0")
	})
}
