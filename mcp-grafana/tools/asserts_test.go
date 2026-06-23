//go:build unit
// +build unit

package tools

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupMockAssertsServer(handler http.HandlerFunc) (*httptest.Server, context.Context) {
	server := httptest.NewServer(handler)
	config := mcpgrafana.GrafanaConfig{
		URL:    server.URL,
		APIKey: "test-api-key",
	}
	ctx := mcpgrafana.WithGrafanaConfig(context.Background(), config)
	return server, ctx
}

func TestAssertTools(t *testing.T) {
	t.Run("get assertions", func(t *testing.T) {
		startTime := time.Date(2025, 4, 23, 10, 0, 0, 0, time.UTC)
		endTime := time.Date(2025, 4, 23, 11, 0, 0, 0, time.UTC)
		server, ctx := setupMockAssertsServer(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/api/plugins/grafana-asserts-app/resources/asserts/api-server/v1/assertions/llm-summary", r.URL.Path)
			require.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))

			var requestBody map[string]interface{}
			err := json.NewDecoder(r.Body).Decode(&requestBody)
			require.NoError(t, err)

			expectedBody := map[string]interface{}{
				"startTime": float64(startTime.UnixMilli()),
				"endTime":   float64(endTime.UnixMilli()),
				"entityKeys": []interface{}{
					map[string]interface{}{
						"type": "Service",
						"name": "mongodb",
						"scope": map[string]interface{}{
							"env":       "asserts-demo",
							"site":      "app",
							"namespace": "robot-shop",
						},
					},
				},
				"suggestionSrcEntities": []interface{}{},
				"alertCategories":       []interface{}{"saturation", "amend", "anomaly", "failure", "error"},
			}
			require.Equal(t, expectedBody, requestBody)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, err = w.Write([]byte(`{"summary": "test summary"}`))
			require.NoError(t, err)
		})
		defer server.Close()

		result, err := getAssertions(ctx, GetAssertionsParams{
			StartTime:  startTime,
			EndTime:    endTime,
			EntityType: "Service",
			EntityName: "mongodb",
			Env:        "asserts-demo",
			Site:       "app",
			Namespace:  "robot-shop",
		})
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, `{"summary": "test summary"}`, result)
	})

	t.Run("get assertions with no site and namespace", func(t *testing.T) {
		startTime := time.Date(2025, 4, 23, 10, 0, 0, 0, time.UTC)
		endTime := time.Date(2025, 4, 23, 11, 0, 0, 0, time.UTC)
		server, ctx := setupMockAssertsServer(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/api/plugins/grafana-asserts-app/resources/asserts/api-server/v1/assertions/llm-summary", r.URL.Path)
			require.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))

			var requestBody map[string]interface{}
			err := json.NewDecoder(r.Body).Decode(&requestBody)
			require.NoError(t, err)

			expectedBody := map[string]interface{}{
				"startTime": float64(startTime.UnixMilli()),
				"endTime":   float64(endTime.UnixMilli()),
				"entityKeys": []interface{}{
					map[string]interface{}{
						"type": "Service",
						"name": "mongodb",
						"scope": map[string]interface{}{
							"env": "asserts-demo",
						},
					},
				},
				"suggestionSrcEntities": []interface{}{},
				"alertCategories":       []interface{}{"saturation", "amend", "anomaly", "failure", "error"},
			}
			require.Equal(t, expectedBody, requestBody)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, err = w.Write([]byte(`{"summary": "test summary"}`))
			require.NoError(t, err)
		})
		defer server.Close()

		result, err := getAssertions(ctx, GetAssertionsParams{
			StartTime:  startTime,
			EndTime:    endTime,
			EntityType: "Service",
			EntityName: "mongodb",
			Env:        "asserts-demo",
		})
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, `{"summary": "test summary"}`, result)
	})
}
