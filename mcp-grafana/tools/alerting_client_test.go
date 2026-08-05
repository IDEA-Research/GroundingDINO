package tools

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/prometheus/prometheus/model/labels"
	"github.com/stretchr/testify/require"

	mcpgrafana "github.com/grafana/mcp-grafana"
)

var (
	fakeruleGroup = ruleGroup{
		Name:      "TestGroup",
		FolderUID: "test-folder",
		Rules: []alertingRule{
			{
				State:     "firing",
				Name:      "Test Alert Rule",
				UID:       "test-rule-uid",
				FolderUID: "test-folder",
				Labels:    labels.New(labels.Label{Name: "severity", Value: "critical"}),
				Alerts: []alert{
					{
						Labels:      labels.New(labels.Label{Name: "instance", Value: "test-instance"}),
						Annotations: labels.New(labels.Label{Name: "summary", Value: "Test alert firing"}),
						State:       "firing",
						Value:       "1",
					},
				},
			},
		},
	}
)

func setupMockServer(handler http.HandlerFunc) (*httptest.Server, *alertingClient) {
	server := httptest.NewServer(handler)
	baseURL, _ := url.Parse(server.URL)
	client := &alertingClient{
		baseURL:    baseURL,
		apiKey:     "test-api-key",
		httpClient: &http.Client{},
	}
	return server, client
}

func mockrulesResponse() rulesResponse {
	resp := rulesResponse{}
	resp.Data.RuleGroups = []ruleGroup{fakeruleGroup}
	return resp
}

func TestAlertingClient_GetRules(t *testing.T) {
	server, client := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/prometheus/grafana/api/v1/rules", r.URL.Path)
		require.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))

		resp := mockrulesResponse()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		err := json.NewEncoder(w).Encode(resp)
		require.NoError(t, err)
	})
	defer server.Close()

	rules, err := client.GetRules(context.Background())
	require.NoError(t, err)
	require.NotNil(t, rules)
	require.ElementsMatch(t, rules.Data.RuleGroups, []ruleGroup{fakeruleGroup})
}

func TestAlertingClient_GetRules_Error(t *testing.T) {
	t.Run("internal server error", func(t *testing.T) {
		server, client := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, err := w.Write([]byte("internal server error"))
			require.NoError(t, err)
		})
		defer server.Close()

		rules, err := client.GetRules(context.Background())
		require.Error(t, err)
		require.Nil(t, rules)
		require.ErrorContains(t, err, "grafana API returned status code 500: internal server error")
	})

	t.Run("network error", func(t *testing.T) {
		server, client := setupMockServer(func(w http.ResponseWriter, r *http.Request) {})
		server.Close()

		rules, err := client.GetRules(context.Background())

		require.Error(t, err)
		require.Nil(t, rules)
		require.ErrorContains(t, err, "failed to execute request")
	})
}

func TestNewAlertingClientFromContext(t *testing.T) {
	config := mcpgrafana.GrafanaConfig{
		URL:    "http://localhost:3000/",
		APIKey: "test-api-key",
	}
	ctx := mcpgrafana.WithGrafanaConfig(context.Background(), config)

	client, err := newAlertingClientFromContext(ctx)
	require.NoError(t, err)

	require.Equal(t, "http://localhost:3000", client.baseURL.String())
	require.Equal(t, "test-api-key", client.apiKey)
	require.NotNil(t, client.httpClient)
}
