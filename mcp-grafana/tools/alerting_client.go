package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/alertmanager/api/v2/models"
	"github.com/prometheus/alertmanager/config"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/prometheus/model/labels"
	"gopkg.in/yaml.v3"

	"github.com/grafana/grafana-openapi-client-go/client"
	mcpgrafana "github.com/grafana/mcp-grafana"
)

const (
	defaultTimeout    = 30 * time.Second
	rulesEndpointPath = "/api/prometheus/grafana/api/v1/rules"
)

type alertingClient struct {
	baseURL     *url.URL
	accessToken string
	idToken     string
	apiKey      string
	basicAuth   *url.Userinfo
	orgID       int64
	httpClient  *http.Client
}

func newAlertingClientFromContext(ctx context.Context) (*alertingClient, error) {
	cfg := mcpgrafana.GrafanaConfigFromContext(ctx)
	baseURL := strings.TrimRight(cfg.URL, "/")
	parsedBaseURL, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid Grafana base URL %q: %w", baseURL, err)
	}

	client := &alertingClient{
		baseURL:     parsedBaseURL,
		accessToken: cfg.AccessToken,
		idToken:     cfg.IDToken,
		apiKey:      cfg.APIKey,
		basicAuth:   cfg.BasicAuth,
		orgID:       cfg.OrgID,
		httpClient: &http.Client{
			Timeout: defaultTimeout,
		},
	}

	// Create custom transport with TLS configuration if available
	if tlsConfig := mcpgrafana.GrafanaConfigFromContext(ctx).TLSConfig; tlsConfig != nil {
		client.httpClient.Transport, err = tlsConfig.HTTPTransport(http.DefaultTransport.(*http.Transport))
		if err != nil {
			return nil, fmt.Errorf("failed to create custom transport: %w", err)
		}
		// Wrap with user agent
		client.httpClient.Transport = mcpgrafana.NewUserAgentTransport(
			client.httpClient.Transport,
		)
	} else {
		// No custom TLS, but still add user agent
		client.httpClient.Transport = mcpgrafana.NewUserAgentTransport(
			http.DefaultTransport,
		)
	}

	return client, nil
}

func (c *alertingClient) makeRequest(ctx context.Context, path string) (*http.Response, error) {
	p := c.baseURL.JoinPath(path).String()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request to %s: %w", p, err)
	}

	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")

	// If accessToken is set we use that first and fall back to normal Authorization.
	if c.accessToken != "" && c.idToken != "" {
		req.Header.Set("X-Access-Token", c.accessToken)
		req.Header.Set("X-Grafana-Id", c.idToken)
	} else if c.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	} else if c.basicAuth != nil {
		password, _ := c.basicAuth.Password()
		req.SetBasicAuth(c.basicAuth.Username(), password)
	}

	// Add org ID header for multi-org support
	if c.orgID > 0 {
		req.Header.Set(client.OrgIDHeader, strconv.FormatInt(c.orgID, 10))
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request to %s: %w", p, err)
	}
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close() //nolint:errcheck
		return nil, fmt.Errorf("grafana API returned status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	return resp, nil
}

func (c *alertingClient) GetRules(ctx context.Context) (*rulesResponse, error) {
	resp, err := c.makeRequest(ctx, rulesEndpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get alert rules from Grafana API: %w", err)
	}
	defer func() {
		_ = resp.Body.Close() //nolint:errcheck
	}()

	var rulesResponse rulesResponse
	decoder := json.NewDecoder(resp.Body)
	if err := decoder.Decode(&rulesResponse); err != nil {
		return nil, fmt.Errorf("failed to decode rules response from %s: %w", rulesEndpointPath, err)
	}

	return &rulesResponse, nil
}

type rulesResponse struct {
	Data struct {
		RuleGroups []ruleGroup      `json:"groups"`
		NextToken  string           `json:"groupNextToken,omitempty"`
		Totals     map[string]int64 `json:"totals,omitempty"`
	} `json:"data"`
}

type ruleGroup struct {
	Name           string         `json:"name"`
	FolderUID      string         `json:"folderUid"`
	Rules          []alertingRule `json:"rules"`
	Interval       float64        `json:"interval"`
	LastEvaluation time.Time      `json:"lastEvaluation"`
	EvaluationTime float64        `json:"evaluationTime"`
}

type alertingRule struct {
	State          string           `json:"state,omitempty"`
	Name           string           `json:"name,omitempty"`
	Query          string           `json:"query,omitempty"`
	Duration       float64          `json:"duration,omitempty"`
	KeepFiringFor  float64          `json:"keepFiringFor,omitempty"`
	Annotations    labels.Labels    `json:"annotations,omitempty"`
	ActiveAt       *time.Time       `json:"activeAt,omitempty"`
	Alerts         []alert          `json:"alerts,omitempty"`
	Totals         map[string]int64 `json:"totals,omitempty"`
	TotalsFiltered map[string]int64 `json:"totalsFiltered,omitempty"`
	UID            string           `json:"uid"`
	FolderUID      string           `json:"folderUid"`
	Labels         labels.Labels    `json:"labels,omitempty"`
	Health         string           `json:"health"`
	LastError      string           `json:"lastError,omitempty"`
	Type           string           `json:"type"`
	LastEvaluation time.Time        `json:"lastEvaluation"`
	EvaluationTime float64          `json:"evaluationTime"`
}

type alert struct {
	Labels      labels.Labels `json:"labels"`
	Annotations labels.Labels `json:"annotations"`
	State       string        `json:"state"`
	ActiveAt    *time.Time    `json:"activeAt"`
	Value       string        `json:"value"`
}

// GetDatasourceRules queries a datasource's Prometheus ruler API
func (c *alertingClient) GetDatasourceRules(ctx context.Context, datasourceUID string) (*v1.RulesResult, error) {
	// use the Grafana unified endpoint - maybe we need to use the datasource proxy endpoint in the future as this
	// is an api for internal use
	path := fmt.Sprintf("/api/prometheus/%s/api/v1/rules", datasourceUID)
	resp, err := c.makeRequest(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("failed to get datasource rules: %w", err)
	}
	defer func() {
		_ = resp.Body.Close() //nolint:errcheck
	}()

	var response struct {
		Status string         `json:"status"`
		Data   v1.RulesResult `json:"data"`
	}
	decoder := json.NewDecoder(resp.Body)
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode datasource rules response: %w", err)
	}

	if response.Status != "success" {
		return nil, fmt.Errorf("datasource rules API returned status: %s", response.Status)
	}

	return &response.Data, nil
}

// GetAlertmanagerConfig queries an Alertmanager datasource for its configuration
// The implementation type determines the API path:
// - prometheus: /api/v2/status (returns upstream AlertmanagerStatus with YAML config)
// - mimir/cortex: /api/v1/alerts (returns YAML with nested alertmanager_config)
func (c *alertingClient) GetAlertmanagerConfig(ctx context.Context, datasourceUID, implementation string) (*config.Config, error) {
	// determine the API path based on implementation type
	var apiPath string
	var isPrometheusV2 bool
	switch strings.ToLower(implementation) {
	case "prometheus":
		apiPath = "/api/v2/status"
		isPrometheusV2 = true
	case "mimir", "cortex":
		apiPath = "/api/v1/alerts"
	default:
		// default to prometheus
		apiPath = "/api/v2/status"
		isPrometheusV2 = true
	}

	path := fmt.Sprintf("/api/datasources/proxy/uid/%s%s", datasourceUID, apiPath)
	resp, err := c.makeRequest(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("failed to get Alertmanager config: %w", err)
	}
	defer func() {
		_ = resp.Body.Close() //nolint:errcheck
	}()

	if isPrometheusV2 {
		var statusResp models.AlertmanagerStatus
		decoder := json.NewDecoder(resp.Body)
		if err := decoder.Decode(&statusResp); err != nil {
			return nil, fmt.Errorf("failed to decode Alertmanager status response: %w", err)
		}

		var cfg config.Config
		if statusResp.Config != nil && statusResp.Config.Original != nil && *statusResp.Config.Original != "" {
			if err := yaml.Unmarshal([]byte(*statusResp.Config.Original), &cfg); err != nil {
				return nil, fmt.Errorf("failed to parse Alertmanager YAML config: %w", err)
			}
		}

		return &cfg, nil
	}

	// Mimir/Cortex /api/v1/alerts returns YAML with alertmanager_config field
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read Alertmanager config response: %w", err)
	}

	var mimirResp struct {
		TemplateFiles      any    `yaml:"template_files"`
		AlertmanagerConfig string `yaml:"alertmanager_config"` // Nested YAML string
	}
	if err := yaml.Unmarshal(bodyBytes, &mimirResp); err != nil {
		return nil, fmt.Errorf("failed to decode Mimir alertmanager response: %w", err)
	}

	// Parse the nested alertmanager_config YAML string
	var cfg config.Config
	if err := yaml.Unmarshal([]byte(mimirResp.AlertmanagerConfig), &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse Mimir alertmanager_config YAML: %w", err)
	}

	return &cfg, nil
}
