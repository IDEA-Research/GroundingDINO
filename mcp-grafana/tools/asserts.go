package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func newAssertsClient(ctx context.Context) (*Client, error) {
	cfg := mcpgrafana.GrafanaConfigFromContext(ctx)
	url := fmt.Sprintf("%s/api/plugins/grafana-asserts-app/resources/asserts/api-server", strings.TrimRight(cfg.URL, "/"))

	// Create custom transport with TLS configuration if available
	var transport = http.DefaultTransport
	if tlsConfig := cfg.TLSConfig; tlsConfig != nil {
		var err error
		transport, err = tlsConfig.HTTPTransport(transport.(*http.Transport))
		if err != nil {
			return nil, fmt.Errorf("failed to create custom transport: %w", err)
		}
	}

	transport = NewAuthRoundTripper(transport, cfg.AccessToken, cfg.IDToken, cfg.APIKey, cfg.BasicAuth)
	transport = mcpgrafana.NewOrgIDRoundTripper(transport, cfg.OrgID)

	client := &http.Client{
		Transport: mcpgrafana.NewUserAgentTransport(
			transport,
		),
	}

	return &Client{
		httpClient: client,
		baseURL:    url,
	}, nil
}

type GetAssertionsParams struct {
	StartTime  time.Time `json:"startTime" jsonschema:"required,description=The start time in RFC3339 format"`
	EndTime    time.Time `json:"endTime" jsonschema:"required,description=The end time in RFC3339 format"`
	EntityType string    `json:"entityType" jsonschema:"description=The type of the entity to list (e.g. Service\\, Node\\, Pod\\, etc.)"`
	EntityName string    `json:"entityName" jsonschema:"description=The name of the entity to list"`
	Env        string    `json:"env,omitempty" jsonschema:"description=The env of the entity to list"`
	Site       string    `json:"site,omitempty" jsonschema:"description=The site of the entity to list"`
	Namespace  string    `json:"namespace,omitempty" jsonschema:"description=The namespace of the entity to list"`
}

type scope struct {
	Env       string `json:"env,omitempty"`
	Site      string `json:"site,omitempty"`
	Namespace string `json:"namespace,omitempty"`
}

type entity struct {
	Name  string `json:"name"`
	Type  string `json:"type"`
	Scope scope  `json:"scope"`
}

type requestBody struct {
	StartTime             int64    `json:"startTime"`
	EndTime               int64    `json:"endTime"`
	EntityKeys            []entity `json:"entityKeys"`
	SuggestionSrcEntities []entity `json:"suggestionSrcEntities"`
	AlertCategories       []string `json:"alertCategories"`
}

func (c *Client) fetchAssertsData(ctx context.Context, urlPath string, method string, reqBody any) (string, error) {
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+urlPath, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to execute request: %w", err)
	}
	defer func() {
		_ = resp.Body.Close() //nolint:errcheck
	}()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	return string(body), nil
}

func getAssertions(ctx context.Context, args GetAssertionsParams) (string, error) {
	client, err := newAssertsClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create Asserts client: %w", err)
	}

	// Create request body
	reqBody := requestBody{
		StartTime: args.StartTime.UnixMilli(),
		EndTime:   args.EndTime.UnixMilli(),
		EntityKeys: []entity{
			{
				Name:  args.EntityName,
				Type:  args.EntityType,
				Scope: scope{},
			},
		},
		SuggestionSrcEntities: []entity{},
		AlertCategories:       []string{"saturation", "amend", "anomaly", "failure", "error"},
	}

	if args.Env != "" {
		reqBody.EntityKeys[0].Scope.Env = args.Env
	}
	if args.Site != "" {
		reqBody.EntityKeys[0].Scope.Site = args.Site
	}
	if args.Namespace != "" {
		reqBody.EntityKeys[0].Scope.Namespace = args.Namespace
	}

	data, err := client.fetchAssertsData(ctx, "/v1/assertions/llm-summary", "POST", reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to fetch data: %w", err)
	}

	return data, nil
}

var GetAssertions = mcpgrafana.MustTool(
	"get_assertions",
	"Get assertion summary for a given entity with its type, name, env, site, namespace, and a time range",
	getAssertions,
	mcp.WithTitleAnnotation("Get assertions summary"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

func AddAssertsTools(mcp *server.MCPServer) {
	GetAssertions.Register(mcp)
}
