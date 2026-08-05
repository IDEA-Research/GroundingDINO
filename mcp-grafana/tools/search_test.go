// Requires a Grafana instance running on localhost:3000,
// with a dashboard named "Demo" provisioned.
// Run with `go test -tags integration`.
//go:build integration

package tools

import (
	"testing"

	"github.com/grafana/grafana-openapi-client-go/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSearchTools(t *testing.T) {
	t.Run("search dashboards", func(t *testing.T) {
		ctx := newTestContext()
		result, err := searchDashboards(ctx, SearchDashboardsParams{
			Query: "Demo",
		})
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, models.HitType("dash-db"), result[0].Type)
	})

	t.Run("search folders", func(t *testing.T) {
		ctx := newTestContext()
		result, err := searchFolders(ctx, SearchFoldersParams{
			Query: "Tests",
		})
		require.NoError(t, err)
		assert.NotEmpty(t, result)
		assert.Equal(t, models.HitType("dash-folder"), result[0].Type)
	})
}
