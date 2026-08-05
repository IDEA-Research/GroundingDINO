package tools

import (
	"context"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/backend/gtime"
	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/prometheus/client_golang/api"
	promv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/config"
	"github.com/prometheus/common/model"
	"github.com/prometheus/prometheus/model/labels"
)

var (
	matchTypeMap = map[string]labels.MatchType{
		"":   labels.MatchEqual,
		"=":  labels.MatchEqual,
		"!=": labels.MatchNotEqual,
		"=~": labels.MatchRegexp,
		"!~": labels.MatchNotRegexp,
	}
)

func promClientFromContext(ctx context.Context, uid string) (promv1.API, error) {
	// First check if the datasource exists
	_, err := getDatasourceByUID(ctx, GetDatasourceByUIDParams{UID: uid})
	if err != nil {
		return nil, err
	}

	cfg := mcpgrafana.GrafanaConfigFromContext(ctx)
	url := fmt.Sprintf("%s/api/datasources/proxy/uid/%s", strings.TrimRight(cfg.URL, "/"), uid)

	// Create custom transport with TLS configuration if available
	rt := api.DefaultRoundTripper
	if tlsConfig := cfg.TLSConfig; tlsConfig != nil {
		customTransport, err := tlsConfig.HTTPTransport(rt.(*http.Transport))
		if err != nil {
			return nil, fmt.Errorf("failed to create custom transport: %w", err)
		}
		rt = customTransport
	}

	if cfg.AccessToken != "" && cfg.IDToken != "" {
		rt = config.NewHeadersRoundTripper(&config.Headers{
			Headers: map[string]config.Header{
				"X-Access-Token": {
					Secrets: []config.Secret{config.Secret(cfg.AccessToken)},
				},
				"X-Grafana-Id": {
					Secrets: []config.Secret{config.Secret(cfg.IDToken)},
				},
			},
		}, rt)
	} else if cfg.APIKey != "" {
		rt = config.NewAuthorizationCredentialsRoundTripper(
			"Bearer", config.NewInlineSecret(cfg.APIKey), rt,
		)
	} else if cfg.BasicAuth != nil {
		password, _ := cfg.BasicAuth.Password()
		rt = config.NewBasicAuthRoundTripper(config.NewInlineSecret(cfg.BasicAuth.Username()), config.NewInlineSecret(password), rt)
	}

	// Wrap with org ID support
	rt = mcpgrafana.NewOrgIDRoundTripper(rt, cfg.OrgID)

	c, err := api.NewClient(api.Config{
		Address:      url,
		RoundTripper: rt,
	})
	if err != nil {
		return nil, fmt.Errorf("creating Prometheus client: %w", err)
	}

	return promv1.NewAPI(c), nil
}

type ListPrometheusMetricMetadataParams struct {
	DatasourceUID  string `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	Limit          int    `json:"limit" jsonschema:"default=10,description=The maximum number of metrics to return"`
	LimitPerMetric int    `json:"limitPerMetric" jsonschema:"description=The maximum number of metrics to return per metric"`
	Metric         string `json:"metric" jsonschema:"description=The metric to query"`
}

func listPrometheusMetricMetadata(ctx context.Context, args ListPrometheusMetricMetadataParams) (map[string][]promv1.Metadata, error) {
	promClient, err := promClientFromContext(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("getting Prometheus client: %w", err)
	}

	limit := args.Limit
	if limit == 0 {
		limit = 10
	}

	metadata, err := promClient.Metadata(ctx, args.Metric, fmt.Sprintf("%d", limit))
	if err != nil {
		return nil, fmt.Errorf("listing Prometheus metric metadata: %w", err)
	}
	return metadata, nil
}

var ListPrometheusMetricMetadata = mcpgrafana.MustTool(
	"list_prometheus_metric_metadata",
	"List Prometheus metric metadata. Returns metadata about metrics currently scraped from targets. Note: This endpoint is experimental.",
	listPrometheusMetricMetadata,
	mcp.WithTitleAnnotation("List Prometheus metric metadata"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type QueryPrometheusParams struct {
	DatasourceUID string `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	Expr          string `json:"expr" jsonschema:"required,description=The PromQL expression to query"`
	StartTime     string `json:"startTime" jsonschema:"required,description=The start time. Supported formats are RFC3339 or relative to now (e.g. 'now'\\, 'now-1.5h'\\, 'now-2h45m'). Valid time units are 'ns'\\, 'us' (or 'µs')\\, 'ms'\\, 's'\\, 'm'\\, 'h'\\, 'd'."`
	EndTime       string `json:"endTime,omitempty" jsonschema:"description=The end time. Required if queryType is 'range'\\, ignored if queryType is 'instant' Supported formats are RFC3339 or relative to now (e.g. 'now'\\, 'now-1.5h'\\, 'now-2h45m'). Valid time units are 'ns'\\, 'us' (or 'µs')\\, 'ms'\\, 's'\\, 'm'\\, 'h'\\, 'd'."`
	StepSeconds   int    `json:"stepSeconds,omitempty" jsonschema:"description=The time series step size in seconds. Required if queryType is 'range'\\, ignored if queryType is 'instant'"`
	QueryType     string `json:"queryType,omitempty" jsonschema:"description=The type of query to use. Either 'range' or 'instant'"`
}

func parseTime(timeStr string) (time.Time, error) {
	tr := gtime.TimeRange{
		From: timeStr,
		Now:  time.Now(),
	}
	return tr.ParseFrom()
}

func queryPrometheus(ctx context.Context, args QueryPrometheusParams) (model.Value, error) {
	promClient, err := promClientFromContext(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("getting Prometheus client: %w", err)
	}

	queryType := args.QueryType
	if queryType == "" {
		queryType = "range"
	}

	var startTime time.Time
	startTime, err = parseTime(args.StartTime)
	if err != nil {
		return nil, fmt.Errorf("parsing start time: %w", err)
	}

	switch queryType {
	case "range":
		if args.StepSeconds == 0 {
			return nil, fmt.Errorf("stepSeconds must be provided when queryType is 'range'")
		}

		var endTime time.Time
		endTime, err = parseTime(args.EndTime)
		if err != nil {
			return nil, fmt.Errorf("parsing end time: %w", err)
		}

		step := time.Duration(args.StepSeconds) * time.Second
		result, _, err := promClient.QueryRange(ctx, args.Expr, promv1.Range{
			Start: startTime,
			End:   endTime,
			Step:  step,
		})
		if err != nil {
			return nil, fmt.Errorf("querying Prometheus range: %w", err)
		}
		return result, nil
	case "instant":
		result, _, err := promClient.Query(ctx, args.Expr, startTime)
		if err != nil {
			return nil, fmt.Errorf("querying Prometheus instant: %w", err)
		}
		return result, nil
	}

	return nil, fmt.Errorf("invalid query type: %s", queryType)
}

var QueryPrometheus = mcpgrafana.MustTool(
	"query_prometheus",
	"Query Prometheus using a PromQL expression. Supports both instant queries (at a single point in time) and range queries (over a time range). Time can be specified either in RFC3339 format or as relative time expressions like 'now', 'now-1h', 'now-30m', etc.",
	queryPrometheus,
	mcp.WithTitleAnnotation("Query Prometheus metrics"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type ListPrometheusMetricNamesParams struct {
	DatasourceUID string `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	Regex         string `json:"regex" jsonschema:"description=The regex to match against the metric names"`
	Limit         int    `json:"limit,omitempty" jsonschema:"default=10,description=The maximum number of results to return"`
	Page          int    `json:"page,omitempty" jsonschema:"default=1,description=The page number to return"`
}

func listPrometheusMetricNames(ctx context.Context, args ListPrometheusMetricNamesParams) ([]string, error) {
	promClient, err := promClientFromContext(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("getting Prometheus client: %w", err)
	}

	limit := args.Limit
	if limit == 0 {
		limit = 10
	}

	page := args.Page
	if page == 0 {
		page = 1
	}

	// Get all metric names by querying for __name__ label values
	labelValues, _, err := promClient.LabelValues(ctx, "__name__", nil, time.Time{}, time.Time{})
	if err != nil {
		return nil, fmt.Errorf("listing Prometheus metric names: %w", err)
	}

	// Filter by regex if provided
	matches := []string{}
	if args.Regex != "" {
		re, err := regexp.Compile(args.Regex)
		if err != nil {
			return nil, fmt.Errorf("compiling regex: %w", err)
		}
		for _, val := range labelValues {
			if re.MatchString(string(val)) {
				matches = append(matches, string(val))
			}
		}
	} else {
		for _, val := range labelValues {
			matches = append(matches, string(val))
		}
	}

	// Apply pagination
	start := (page - 1) * limit
	end := start + limit
	if start >= len(matches) {
		matches = []string{}
	} else if end > len(matches) {
		matches = matches[start:]
	} else {
		matches = matches[start:end]
	}

	return matches, nil
}

var ListPrometheusMetricNames = mcpgrafana.MustTool(
	"list_prometheus_metric_names",
	"List metric names in a Prometheus datasource. Retrieves all metric names and then filters them locally using the provided regex. Supports pagination.",
	listPrometheusMetricNames,
	mcp.WithTitleAnnotation("List Prometheus metric names"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type LabelMatcher struct {
	Name  string `json:"name" jsonschema:"required,description=The name of the label to match against"`
	Value string `json:"value" jsonschema:"required,description=The value to match against"`
	Type  string `json:"type" jsonschema:"required,description=One of the '=' or '!=' or '=~' or '!~'"`
}

type Selector struct {
	Filters []LabelMatcher `json:"filters"`
}

func (s Selector) String() string {
	b := strings.Builder{}
	b.WriteRune('{')
	for i, f := range s.Filters {
		if f.Type == "" {
			f.Type = "="
		}
		b.WriteString(fmt.Sprintf(`%s%s'%s'`, f.Name, f.Type, f.Value))
		if i < len(s.Filters)-1 {
			b.WriteString(", ")
		}
	}
	b.WriteRune('}')
	return b.String()
}

// Matches runs the matchers against the given labels and returns whether they match the selector.
func (s Selector) Matches(lbls labels.Labels) (bool, error) {
	matchers := make(labels.Selector, 0, len(s.Filters))

	for _, filter := range s.Filters {
		matchType, ok := matchTypeMap[filter.Type]
		if !ok {
			return false, fmt.Errorf("invalid matcher type: %s", filter.Type)
		}

		matcher, err := labels.NewMatcher(matchType, filter.Name, filter.Value)
		if err != nil {
			return false, fmt.Errorf("creating matcher: %w", err)
		}

		matchers = append(matchers, matcher)
	}

	return matchers.Matches(lbls), nil
}

type ListPrometheusLabelNamesParams struct {
	DatasourceUID string     `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	Matches       []Selector `json:"matches,omitempty" jsonschema:"description=Optionally\\, a list of label matchers to filter the results by"`
	StartRFC3339  string     `json:"startRfc3339,omitempty" jsonschema:"description=Optionally\\, the start time of the time range to filter the results by"`
	EndRFC3339    string     `json:"endRfc3339,omitempty" jsonschema:"description=Optionally\\, the end time of the time range to filter the results by"`
	Limit         int        `json:"limit,omitempty" jsonschema:"default=100,description=Optionally\\, the maximum number of results to return"`
}

func listPrometheusLabelNames(ctx context.Context, args ListPrometheusLabelNamesParams) ([]string, error) {
	promClient, err := promClientFromContext(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("getting Prometheus client: %w", err)
	}

	limit := args.Limit
	if limit == 0 {
		limit = 100
	}

	var startTime, endTime time.Time
	if args.StartRFC3339 != "" {
		if startTime, err = time.Parse(time.RFC3339, args.StartRFC3339); err != nil {
			return nil, fmt.Errorf("parsing start time: %w", err)
		}
	}
	if args.EndRFC3339 != "" {
		if endTime, err = time.Parse(time.RFC3339, args.EndRFC3339); err != nil {
			return nil, fmt.Errorf("parsing end time: %w", err)
		}
	}

	var matchers []string
	for _, m := range args.Matches {
		matchers = append(matchers, m.String())
	}

	labelNames, _, err := promClient.LabelNames(ctx, matchers, startTime, endTime)
	if err != nil {
		return nil, fmt.Errorf("listing Prometheus label names: %w", err)
	}

	// Apply limit
	if len(labelNames) > limit {
		labelNames = labelNames[:limit]
	}

	return labelNames, nil
}

var ListPrometheusLabelNames = mcpgrafana.MustTool(
	"list_prometheus_label_names",
	"List label names in a Prometheus datasource. Allows filtering by series selectors and time range.",
	listPrometheusLabelNames,
	mcp.WithTitleAnnotation("List Prometheus label names"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type ListPrometheusLabelValuesParams struct {
	DatasourceUID string     `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	LabelName     string     `json:"labelName" jsonschema:"required,description=The name of the label to query"`
	Matches       []Selector `json:"matches,omitempty" jsonschema:"description=Optionally\\, a list of selectors to filter the results by"`
	StartRFC3339  string     `json:"startRfc3339,omitempty" jsonschema:"description=Optionally\\, the start time of the query"`
	EndRFC3339    string     `json:"endRfc3339,omitempty" jsonschema:"description=Optionally\\, the end time of the query"`
	Limit         int        `json:"limit,omitempty" jsonschema:"default=100,description=Optionally\\, the maximum number of results to return"`
}

func listPrometheusLabelValues(ctx context.Context, args ListPrometheusLabelValuesParams) (model.LabelValues, error) {
	promClient, err := promClientFromContext(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("getting Prometheus client: %w", err)
	}

	limit := args.Limit
	if limit == 0 {
		limit = 100
	}

	var startTime, endTime time.Time
	if args.StartRFC3339 != "" {
		if startTime, err = time.Parse(time.RFC3339, args.StartRFC3339); err != nil {
			return nil, fmt.Errorf("parsing start time: %w", err)
		}
	}
	if args.EndRFC3339 != "" {
		if endTime, err = time.Parse(time.RFC3339, args.EndRFC3339); err != nil {
			return nil, fmt.Errorf("parsing end time: %w", err)
		}
	}

	var matchers []string
	for _, m := range args.Matches {
		matchers = append(matchers, m.String())
	}

	labelValues, _, err := promClient.LabelValues(ctx, args.LabelName, matchers, startTime, endTime)
	if err != nil {
		return nil, fmt.Errorf("listing Prometheus label values: %w", err)
	}

	// Apply limit
	if len(labelValues) > limit {
		labelValues = labelValues[:limit]
	}

	return labelValues, nil
}

var ListPrometheusLabelValues = mcpgrafana.MustTool(
	"list_prometheus_label_values",
	"Get the values for a specific label name in Prometheus. Allows filtering by series selectors and time range.",
	listPrometheusLabelValues,
	mcp.WithTitleAnnotation("List Prometheus label values"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

func AddPrometheusTools(mcp *server.MCPServer) {
	ListPrometheusMetricMetadata.Register(mcp)
	QueryPrometheus.Register(mcp)
	ListPrometheusMetricNames.Register(mcp)
	ListPrometheusLabelNames.Register(mcp)
	ListPrometheusLabelValues.Register(mcp)
}
