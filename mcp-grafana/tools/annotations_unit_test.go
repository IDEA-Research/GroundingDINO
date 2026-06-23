//go:build unit

package tools

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/grafana/grafana-openapi-client-go/client"
	"github.com/grafana/grafana-openapi-client-go/models"
	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func mockCtxWithClient(server *httptest.Server) context.Context {
	u, _ := url.Parse(server.URL)
	cfg := client.DefaultTransportConfig()
	cfg.Host = u.Host
	cfg.Schemes = []string{"http"}
	cfg.APIKey = "test"

	c := client.NewHTTPClientWithConfig(nil, cfg)
	return mcpgrafana.WithGrafanaClient(context.Background(), c)
}

func TestGetAnnotations_UsesCorrectQueryParams(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/annotations", r.URL.Path)

		q := r.URL.Query()
		assert.Equal(t, "50", q.Get("limit"))
		assert.Equal(t, "dash-1", q.Get("dashboardUID"))
		assert.Equal(t, "true", q.Get("matchAny"))
		assert.Equal(t, "tagA", q["tags"][0])
		assert.Equal(t, "tagB", q["tags"][1])

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode([]interface{}{})
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)
	limit := int64(50)
	uid := "dash-1"
	matchAny := true

	_, err := getAnnotations(ctx, GetAnnotationsInput{
		Limit:        &limit,
		DashboardUID: &uid,
		MatchAny:     &matchAny,
		Tags:         []string{"tagA", "tagB"},
	})
	require.NoError(t, err)
}

func TestGetAnnotations_PropagatesError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`oops`))
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)

	_, err := getAnnotations(ctx, GetAnnotationsInput{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "get annotations:")
}

func TestCreateAnnotationGraphiteFormat_SendsCorrectBody_Minimal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/annotations/graphite", r.URL.Path)
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "Bearer test", r.Header.Get("Authorization"))

		var body models.PostGraphiteAnnotationsCmd
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Equal(t, "deploy", body.What)
		assert.Equal(t, int64(1710000000000), body.When)
		assert.Nil(t, body.Tags)
		assert.Empty(t, body.Data)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"message":"annotation created"}`))
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)

	_, err := createAnnotationGraphiteFormat(ctx, CreateGraphiteAnnotationInput{
		What: "deploy",
		When: 1710000000000,
	})
	require.NoError(t, err)
}

func TestCreateAnnotationGraphiteFormat_SendsCorrectBody_WithTagsAndData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/annotations/graphite", r.URL.Path)
		assert.Equal(t, "POST", r.Method)

		var body map[string]interface{}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))

		assert.Equal(t, "incident", body["what"])
		assert.Equal(t, float64(1720000000000), body["when"])
		assert.ElementsMatch(t, []interface{}{"sev1", "network"}, body["tags"].([]interface{}))
		assert.Equal(t, "context", body["data"])

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"message":"ok"}`))
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)

	_, err := createAnnotationGraphiteFormat(ctx, CreateGraphiteAnnotationInput{
		What: "incident",
		When: 1720000000000,
		Tags: []string{"sev1", "network"},
		Data: "context",
	})
	require.NoError(t, err)
}

func TestCreateAnnotation_SendsCorrectBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/annotations", r.URL.Path)
		assert.Equal(t, "POST", r.Method)

		var body models.PostAnnotationsCmd
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))

		assert.Equal(t, int64(7), body.PanelID)
		assert.Equal(t, "hello", *body.Text)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id": 1}`))
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)

	_, err := createAnnotation(ctx, CreateAnnotationInput{
		PanelID: 7,
		Text:    "hello",
	})
	require.NoError(t, err)
}

func TestCreateAnnotation_ErrorWrapped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)

	_, err := createAnnotation(ctx, CreateAnnotationInput{Text: "t"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "create annotation:")
}

func TestCreateAnnotationGraphiteFormat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`internal error`))
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)

	_, err := createAnnotationGraphiteFormat(ctx, CreateGraphiteAnnotationInput{
		What: "bad",
		When: 1700000000000,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "create graphite annotation")
}

func TestUpdateAnnotation_SendsCorrectBodyAndPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/annotations/"+strconv.Itoa(55), r.URL.Path)
		assert.Equal(t, "PUT", r.Method)

		var body models.UpdateAnnotationsCmd
		_ = json.NewDecoder(r.Body).Decode(&body)

		assert.Equal(t, int64(111), body.Time)
		assert.Equal(t, int64(222), body.TimeEnd)
		assert.Equal(t, "hello", body.Text)
		assert.Equal(t, []string{"a", "b"}, body.Tags)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)

	_, err := updateAnnotation(ctx, UpdateAnnotationInput{
		ID:      55,
		Time:    111,
		TimeEnd: 222,
		Text:    "hello",
		Tags:    []string{"a", "b"},
	})
	require.NoError(t, err)
}

func TestPatchAnnotation_SendsOnlyProvidedFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/annotations/"+strconv.Itoa(9), r.URL.Path)
		assert.Equal(t, "PATCH", r.Method)

		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)

		assert.Equal(t, "patched", body["text"])
		assert.ElementsMatch(t, []interface{}{"x"}, body["tags"].([]interface{}))
		assert.Nil(t, body["time"])
		assert.Nil(t, body["timeEnd"])

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)
	text := "patched"

	_, err := patchAnnotation(ctx, PatchAnnotationInput{
		ID:   9,
		Text: &text,
		Tags: []string{"x"},
	})
	require.NoError(t, err)
}

func TestGetAnnotationTags_UsesCorrectQueryParams(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/annotations/tags", r.URL.Path)

		q := r.URL.Query()
		assert.Equal(t, "error", q.Get("tag"))
		assert.Equal(t, "50", q.Get("limit"))

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"result":{"tags":[]}}`))
	}))
	defer server.Close()

	ctx := mockCtxWithClient(server)
	tag := "error"
	limit := "50"

	_, err := getAnnotationTags(ctx, GetAnnotationTagsInput{
		Tag:   &tag,
		Limit: &limit,
	})
	require.NoError(t, err)
}
