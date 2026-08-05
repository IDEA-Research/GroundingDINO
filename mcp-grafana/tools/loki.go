package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	// DefaultLokiLogLimit is the default number of log lines to return if not specified
	DefaultLokiLogLimit = 10

	// MaxLokiLogLimit is the maximum number of log lines that can be requested
	MaxLokiLogLimit = 100
)

type Client struct {
	httpClient *http.Client
	baseURL    string
}

// LabelResponse represents the http json response to a label query
type LabelResponse struct {
	Status string   `json:"status"`
	Data   []string `json:"data,omitempty"`
}

// Stats represents the statistics returned by Loki's index/stats endpoint
type Stats struct {
	Streams int `json:"streams"`
	Chunks  int `json:"chunks"`
	Entries int `json:"entries"`
	Bytes   int `json:"bytes"`
}

func newLokiClient(ctx context.Context, uid string) (*Client, error) {
	// First check if the datasource exists
	_, err := getDatasourceByUID(ctx, GetDatasourceByUIDParams{UID: uid})
	if err != nil {
		return nil, err
	}

	cfg := mcpgrafana.GrafanaConfigFromContext(ctx)
	url := fmt.Sprintf("%s/api/datasources/proxy/uid/%s", strings.TrimRight(cfg.URL, "/"), uid)

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

// buildURL constructs a full URL for a Loki API endpoint
func (c *Client) buildURL(urlPath string) string {
	fullURL := c.baseURL
	if !strings.HasSuffix(fullURL, "/") && !strings.HasPrefix(urlPath, "/") {
		fullURL += "/"
	} else if strings.HasSuffix(fullURL, "/") && strings.HasPrefix(urlPath, "/") {
		// Remove the leading slash from urlPath to avoid double slash
		urlPath = strings.TrimPrefix(urlPath, "/")
	}
	return fullURL + urlPath
}

// makeRequest makes an HTTP request to the Loki API and returns the response body
func (c *Client) makeRequest(ctx context.Context, method, urlPath string, params url.Values) ([]byte, error) {
	fullURL := c.buildURL(urlPath)

	u, err := url.Parse(fullURL)
	if err != nil {
		return nil, fmt.Errorf("parsing URL: %w", err)
	}

	if params != nil {
		u.RawQuery = params.Encode()
	}

	req, err := http.NewRequestWithContext(ctx, method, u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() {
		_ = resp.Body.Close() //nolint:errcheck
	}()

	// Check for non-200 status code
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("loki API returned status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Read the response body with a limit to prevent memory issues
	body := io.LimitReader(resp.Body, 1024*1024*48)
	bodyBytes, err := io.ReadAll(body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}

	// Check if the response is empty
	if len(bodyBytes) == 0 {
		return nil, fmt.Errorf("empty response from Loki API")
	}

	// Trim any whitespace that might cause JSON parsing issues
	return bytes.TrimSpace(bodyBytes), nil
}

// fetchData is a generic method to fetch data from Loki API
func (c *Client) fetchData(ctx context.Context, urlPath string, startRFC3339, endRFC3339 string) ([]string, error) {
	params := url.Values{}
	if startRFC3339 != "" {
		params.Add("start", startRFC3339)
	}
	if endRFC3339 != "" {
		params.Add("end", endRFC3339)
	}

	bodyBytes, err := c.makeRequest(ctx, "GET", urlPath, params)
	if err != nil {
		return nil, err
	}

	var labelResponse LabelResponse
	err = json.Unmarshal(bodyBytes, &labelResponse)
	if err != nil {
		return nil, fmt.Errorf("unmarshalling response (content: %s): %w", string(bodyBytes), err)
	}

	if labelResponse.Status != "success" {
		return nil, fmt.Errorf("loki API returned unexpected response format: %s", string(bodyBytes))
	}

	// Check if Data is nil or empty and handle it explicitly
	if labelResponse.Data == nil {
		// Return empty slice instead of nil to avoid potential nil pointer issues
		return []string{}, nil
	}

	if len(labelResponse.Data) == 0 {
		return []string{}, nil
	}

	return labelResponse.Data, nil
}

func NewAuthRoundTripper(rt http.RoundTripper, accessToken, idToken, apiKey string, basicAuth *url.Userinfo) *authRoundTripper {
	return &authRoundTripper{
		accessToken: accessToken,
		idToken:     idToken,
		apiKey:      apiKey,
		basicAuth:   basicAuth,
		underlying:  rt,
	}
}

type authRoundTripper struct {
	accessToken string
	idToken     string
	apiKey      string
	basicAuth   *url.Userinfo
	underlying  http.RoundTripper
}

func (rt *authRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if rt.accessToken != "" && rt.idToken != "" {
		req.Header.Set("X-Access-Token", rt.accessToken)
		req.Header.Set("X-Grafana-Id", rt.idToken)
	} else if rt.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+rt.apiKey)
	} else if rt.basicAuth != nil {
		password, _ := rt.basicAuth.Password()
		req.SetBasicAuth(rt.basicAuth.Username(), password)
	}

	resp, err := rt.underlying.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// ListLokiLabelNamesParams defines the parameters for listing Loki label names
type ListLokiLabelNamesParams struct {
	DatasourceUID string `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	StartRFC3339  string `json:"startRfc3339,omitempty" jsonschema:"description=Optionally\\, the start time of the query in RFC3339 format (defaults to 1 hour ago)"`
	EndRFC3339    string `json:"endRfc3339,omitempty" jsonschema:"description=Optionally\\, the end time of the query in RFC3339 format (defaults to now)"`
}

// listLokiLabelNames lists all label names in a Loki datasource
func listLokiLabelNames(ctx context.Context, args ListLokiLabelNamesParams) ([]string, error) {
	client, err := newLokiClient(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("creating Loki client: %w", err)
	}

	result, err := client.fetchData(ctx, "/loki/api/v1/labels", args.StartRFC3339, args.EndRFC3339)
	if err != nil {
		return nil, err
	}

	if len(result) == 0 {
		return []string{}, nil
	}

	return result, nil
}

// ListLokiLabelNames is a tool for listing Loki label names
var ListLokiLabelNames = mcpgrafana.MustTool(
	"list_loki_label_names",
	"Lists all available label names (keys) found in logs within a specified Loki datasource and time range. Returns a list of unique label strings (e.g., `[\"app\", \"env\", \"pod\"]`). If the time range is not provided, it defaults to the last hour.",
	listLokiLabelNames,
	mcp.WithTitleAnnotation("List Loki label names"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// ListLokiLabelValuesParams defines the parameters for listing Loki label values
type ListLokiLabelValuesParams struct {
	DatasourceUID string `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	LabelName     string `json:"labelName" jsonschema:"required,description=The name of the label to retrieve values for (e.g. 'app'\\, 'env'\\, 'pod')"`
	StartRFC3339  string `json:"startRfc3339,omitempty" jsonschema:"description=Optionally\\, the start time of the query in RFC3339 format (defaults to 1 hour ago)"`
	EndRFC3339    string `json:"endRfc3339,omitempty" jsonschema:"description=Optionally\\, the end time of the query in RFC3339 format (defaults to now)"`
}

// listLokiLabelValues lists all values for a specific label in a Loki datasource
func listLokiLabelValues(ctx context.Context, args ListLokiLabelValuesParams) ([]string, error) {
	client, err := newLokiClient(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("creating Loki client: %w", err)
	}

	// Use the client's fetchData method
	urlPath := fmt.Sprintf("/loki/api/v1/label/%s/values", args.LabelName)

	result, err := client.fetchData(ctx, urlPath, args.StartRFC3339, args.EndRFC3339)
	if err != nil {
		return nil, err
	}

	if len(result) == 0 {
		// Return empty slice instead of nil
		return []string{}, nil
	}

	return result, nil
}

// ListLokiLabelValues is a tool for listing Loki label values
var ListLokiLabelValues = mcpgrafana.MustTool(
	"list_loki_label_values",
	"Retrieves all unique values associated with a specific `labelName` within a Loki datasource and time range. Returns a list of string values (e.g., for `labelName=\"env\"`, might return `[\"prod\", \"staging\", \"dev\"]`). Useful for discovering filter options. Defaults to the last hour if the time range is omitted.",
	listLokiLabelValues,
	mcp.WithTitleAnnotation("List Loki label values"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// LogStream represents a stream of log entries from Loki
type LogStream struct {
	Stream map[string]string   `json:"stream"`
	Values [][]json.RawMessage `json:"values"` // [timestamp, value] where value can be string or number
}

// QueryRangeResponse represents the response from Loki's query_range API
type QueryRangeResponse struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string      `json:"resultType"`
		Result     []LogStream `json:"result"`
	} `json:"data"`
}

// addTimeRangeParams adds start and end time parameters to the URL values
// It handles conversion from RFC3339 to Unix nanoseconds
func addTimeRangeParams(params url.Values, startRFC3339, endRFC3339 string) error {
	if startRFC3339 != "" {
		startTime, err := time.Parse(time.RFC3339, startRFC3339)
		if err != nil {
			return fmt.Errorf("parsing start time: %w", err)
		}
		params.Add("start", fmt.Sprintf("%d", startTime.UnixNano()))
	}

	if endRFC3339 != "" {
		endTime, err := time.Parse(time.RFC3339, endRFC3339)
		if err != nil {
			return fmt.Errorf("parsing end time: %w", err)
		}
		params.Add("end", fmt.Sprintf("%d", endTime.UnixNano()))
	}

	return nil
}

// getDefaultTimeRange returns default start and end times if not provided
// Returns start time (1 hour ago) and end time (now) in RFC3339 format
func getDefaultTimeRange(startRFC3339, endRFC3339 string) (string, string) {
	if startRFC3339 == "" {
		// Default to 1 hour ago if not specified
		startRFC3339 = time.Now().Add(-1 * time.Hour).Format(time.RFC3339)
	}
	if endRFC3339 == "" {
		// Default to now if not specified
		endRFC3339 = time.Now().Format(time.RFC3339)
	}
	return startRFC3339, endRFC3339
}

// fetchLogs is a method to fetch logs from Loki API
func (c *Client) fetchLogs(ctx context.Context, query, startRFC3339, endRFC3339 string, limit int, direction string) ([]LogStream, error) {
	params := url.Values{}
	params.Add("query", query)

	// Add time range parameters
	if err := addTimeRangeParams(params, startRFC3339, endRFC3339); err != nil {
		return nil, err
	}

	if limit > 0 {
		params.Add("limit", fmt.Sprintf("%d", limit))
	}

	if direction != "" {
		params.Add("direction", direction)
	}

	bodyBytes, err := c.makeRequest(ctx, "GET", "/loki/api/v1/query_range", params)
	if err != nil {
		return nil, err
	}

	var queryResponse QueryRangeResponse
	err = json.Unmarshal(bodyBytes, &queryResponse)
	if err != nil {
		return nil, fmt.Errorf("unmarshalling response (content: %s): %w", string(bodyBytes), err)
	}

	if queryResponse.Status != "success" {
		return nil, fmt.Errorf("loki API returned unexpected response format: %s", string(bodyBytes))
	}

	return queryResponse.Data.Result, nil
}

// QueryLokiLogsParams defines the parameters for querying Loki logs
type QueryLokiLogsParams struct {
	DatasourceUID string `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	LogQL         string `json:"logql" jsonschema:"required,description=The LogQL query to execute against Loki. This can be a simple label matcher or a complex query with filters\\, parsers\\, and expressions. Supports full LogQL syntax including label matchers\\, filter operators\\, pattern expressions\\, and pipeline operations."`
	StartRFC3339  string `json:"startRfc3339,omitempty" jsonschema:"description=Optionally\\, the start time of the query in RFC3339 format"`
	EndRFC3339    string `json:"endRfc3339,omitempty" jsonschema:"description=Optionally\\, the end time of the query in RFC3339 format"`
	Limit         int    `json:"limit,omitempty" jsonschema:"default=10,description=Optionally\\, the maximum number of log lines to return (max: 100)"`
	Direction     string `json:"direction,omitempty" jsonschema:"description=Optionally\\, the direction of the query: 'forward' (oldest first) or 'backward' (newest first\\, default)"`
}

// LogEntry represents a single log entry or metric sample with metadata
type LogEntry struct {
	Timestamp string            `json:"timestamp"`
	Line      string            `json:"line,omitempty"`  // For log queries
	Value     *float64          `json:"value,omitempty"` // For metric queries
	Labels    map[string]string `json:"labels"`
}

// enforceLogLimit ensures a log limit value is within acceptable bounds
func enforceLogLimit(requestedLimit int) int {
	if requestedLimit <= 0 {
		return DefaultLokiLogLimit
	}
	if requestedLimit > MaxLokiLogLimit {
		return MaxLokiLogLimit
	}
	return requestedLimit
}

// queryLokiLogs queries logs from a Loki datasource using LogQL
func queryLokiLogs(ctx context.Context, args QueryLokiLogsParams) ([]LogEntry, error) {
	client, err := newLokiClient(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("creating Loki client: %w", err)
	}

	// Get default time range if not provided
	startTime, endTime := getDefaultTimeRange(args.StartRFC3339, args.EndRFC3339)

	// Apply limit constraints
	limit := enforceLogLimit(args.Limit)

	// Set default direction if not provided
	direction := args.Direction
	if direction == "" {
		direction = "backward" // Most recent logs first
	}

	streams, err := client.fetchLogs(ctx, args.LogQL, startTime, endTime, limit, direction)
	if err != nil {
		return nil, err
	}

	// Handle empty results
	if len(streams) == 0 {
		return []LogEntry{}, nil
	}

	// Convert the streams to a flat list of log entries
	var entries []LogEntry
	for _, stream := range streams {
		for _, value := range stream.Values {
			if len(value) >= 2 {
				entry := LogEntry{
					Timestamp: string(value[0]),
					Labels:    stream.Stream,
				}

				// Handle metric queries (numeric values) vs log queries
				if stream.Stream["__type__"] == "metrics" {
					// For metric queries, parse the value as a number
					var numStr string
					if err := json.Unmarshal(value[1], &numStr); err == nil {
						if v, err := strconv.ParseFloat(numStr, 64); err == nil {
							entry.Value = &v
						} else {
							// Skip invalid numeric values
							continue
						}
					} else {
						// Try direct number parsing if string parsing fails
						var v float64
						if err := json.Unmarshal(value[1], &v); err == nil {
							entry.Value = &v
						} else {
							// Skip invalid values
							continue
						}
					}
				} else {
					// For log queries, parse the value as a string
					var logLine string
					if err := json.Unmarshal(value[1], &logLine); err == nil {
						entry.Line = logLine
					} else {
						// Skip invalid log lines
						continue
					}
				}

				entries = append(entries, entry)
			}
		}
	}

	// If we processed all streams but still have no entries, return an empty slice
	if len(entries) == 0 {
		return []LogEntry{}, nil
	}

	return entries, nil
}

// QueryLokiLogs is a tool for querying logs from Loki
var QueryLokiLogs = mcpgrafana.MustTool(
	"query_loki_logs",
	"Executes a LogQL query against a Loki datasource to retrieve log entries or metric values. Returns a list of results, each containing a timestamp, labels, and either a log line (`line`) or a numeric metric value (`value`). Defaults to the last hour, a limit of 10 entries, and 'backward' direction (newest first). Supports full LogQL syntax for log and metric queries (e.g., `{app=\"foo\"} |= \"error\"`, `rate({app=\"bar\"}[1m])`). Prefer using `query_loki_stats` first to check stream size and `list_loki_label_names` and `list_loki_label_values` to verify labels exist.",
	queryLokiLogs,
	mcp.WithTitleAnnotation("Query Loki logs"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// fetchStats is a method to fetch stats data from Loki API
func (c *Client) fetchStats(ctx context.Context, query, startRFC3339, endRFC3339 string) (*Stats, error) {
	params := url.Values{}
	params.Add("query", query)

	// Add time range parameters
	if err := addTimeRangeParams(params, startRFC3339, endRFC3339); err != nil {
		return nil, err
	}

	bodyBytes, err := c.makeRequest(ctx, "GET", "/loki/api/v1/index/stats", params)
	if err != nil {
		return nil, err
	}

	var stats Stats
	err = json.Unmarshal(bodyBytes, &stats)
	if err != nil {
		return nil, fmt.Errorf("unmarshalling response (content: %s): %w", string(bodyBytes), err)
	}

	return &stats, nil
}

// QueryLokiStatsParams defines the parameters for querying Loki stats
type QueryLokiStatsParams struct {
	DatasourceUID string `json:"datasourceUid" jsonschema:"required,description=The UID of the datasource to query"`
	LogQL         string `json:"logql" jsonschema:"required,description=The LogQL matcher expression to execute. This parameter only accepts label matcher expressions and does not support full LogQL queries. Line filters\\, pattern operations\\, and metric aggregations are not supported by the stats API endpoint. Only simple label selectors can be used here."`
	StartRFC3339  string `json:"startRfc3339,omitempty" jsonschema:"description=Optionally\\, the start time of the query in RFC3339 format"`
	EndRFC3339    string `json:"endRfc3339,omitempty" jsonschema:"description=Optionally\\, the end time of the query in RFC3339 format"`
}

// queryLokiStats queries stats from a Loki datasource using LogQL
func queryLokiStats(ctx context.Context, args QueryLokiStatsParams) (*Stats, error) {
	client, err := newLokiClient(ctx, args.DatasourceUID)
	if err != nil {
		return nil, fmt.Errorf("creating Loki client: %w", err)
	}

	// Get default time range if not provided
	startTime, endTime := getDefaultTimeRange(args.StartRFC3339, args.EndRFC3339)

	stats, err := client.fetchStats(ctx, args.LogQL, startTime, endTime)
	if err != nil {
		return nil, err
	}

	return stats, nil
}

// QueryLokiStats is a tool for querying stats from Loki
var QueryLokiStats = mcpgrafana.MustTool(
	"query_loki_stats",
	"Retrieves statistics about log streams matching a given LogQL *selector* within a Loki datasource and time range. Returns an object containing the count of streams, chunks, entries, and total bytes (e.g., `{\"streams\": 5, \"chunks\": 50, \"entries\": 10000, \"bytes\": 512000}`). The `logql` parameter **must** be a simple label selector (e.g., `{app=\"nginx\", env=\"prod\"}`) and does not support line filters, parsers, or aggregations. Defaults to the last hour if the time range is omitted.",
	queryLokiStats,
	mcp.WithTitleAnnotation("Get Loki log statistics"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// AddLokiTools registers all Loki tools with the MCP server
func AddLokiTools(mcp *server.MCPServer) {
	ListLokiLabelNames.Register(mcp)
	ListLokiLabelValues.Register(mcp)
	QueryLokiStats.Register(mcp)
	QueryLokiLogs.Register(mcp)
}
