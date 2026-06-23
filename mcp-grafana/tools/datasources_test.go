// Requires a Grafana instance running on localhost:3000,
// with a Prometheus datasource provisioned.
// Run with `go test -tags integration`.
//go:build integration

package tools

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDatasourcesTools(t *testing.T) {
	t.Run("list datasources", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listDatasources(ctx, ListDatasourcesParams{})
		require.NoError(t, err)
		// Seven datasources are provisioned in the test environment (Prometheus, Prometheus Demo, Loki, Pyroscope, Tempo, Tempo Secondary and Alertmanager).
		assert.Len(t, result, 7)
	})

	t.Run("list datasources for type", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listDatasources(ctx, ListDatasourcesParams{Type: "Prometheus"})
		require.NoError(t, err)
		// Only two Prometheus datasources are provisioned in the test environment.
		assert.Len(t, result, 2)
	})

	t.Run("get datasource by uid", func(t *testing.T) {
		ctx := newTestContext()
		result, err := getDatasourceByUID(ctx, GetDatasourceByUIDParams{
			UID: "prometheus",
		})
		require.NoError(t, err)
		assert.Equal(t, "Prometheus", result.Name)
	})

	t.Run("get datasource by uid - not found", func(t *testing.T) {
		ctx := newTestContext()
		result, err := getDatasourceByUID(ctx, GetDatasourceByUIDParams{
			UID: "non-existent-datasource",
		})
		require.Error(t, err)
		require.Nil(t, result)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("get datasource by name", func(t *testing.T) {
		ctx := newTestContext()
		result, err := getDatasourceByName(ctx, GetDatasourceByNameParams{
			Name: "Prometheus",
		})
		require.NoError(t, err)
		assert.Equal(t, "Prometheus", result.Name)
	})
}
