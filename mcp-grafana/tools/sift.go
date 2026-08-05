package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

type investigationStatus string

const (
	investigationStatusPending  investigationStatus = "pending"
	investigationStatusRunning  investigationStatus = "running"
	investigationStatusFinished investigationStatus = "finished"
	investigationStatusFailed   investigationStatus = "failed"
)

// errorPatternLogExampleLimit controls how many log examples are fetched per error pattern.
const errorPatternLogExampleLimit = 3

type analysisStatus string

type investigationRequest struct {
	AlertLabels map[string]string `json:"alertLabels,omitempty"`
	Labels      map[string]string `json:"labels"`

	Start time.Time `json:"start"`
	End   time.Time `json:"end"`

	QueryURL string `json:"queryUrl"`

	Checks []string `json:"checks"`
}

// Interesting: The analysis complete with results that indicate a probable cause for failure.
type analysisResult struct {
	Successful  bool           `json:"successful"`
	Interesting bool           `json:"interesting"`
	Message     string         `json:"message"`
	Details     map[string]any `json:"details"`
}

type analysisMeta struct {
	Items []analysis `json:"items"`
}

// An analysis struct provides the status and results
// of running a specific type of check.
type analysis struct {
	ID        uuid.UUID `json:"id"`
	CreatedAt time.Time `json:"created"`
	UpdatedAt time.Time `json:"modified"`

	Status    analysisStatus `json:"status"`
	StartedAt *time.Time     `json:"started"`

	// Foreign key to the Investigation that created this Analysis.
	InvestigationID uuid.UUID `json:"investigationId"`

	// Name is the name of the check that this analysis represents.
	Name   string         `json:"name"`
	Title  string         `json:"title"`
	Result analysisResult `json:"result"`
}

type InvestigationDatasources struct {
	LokiDatasource struct {
		UID string `json:"uid"`
	} `json:"lokiDatasource"`
}

type Investigation struct {
	ID        uuid.UUID `json:"id"`
	CreatedAt time.Time `json:"created"`
	UpdatedAt time.Time `json:"modified"`

	TenantID string `json:"tenantId"`

	Name string `json:"name"`

	// GrafanaURL is the Grafana URL to be used for datasource queries
	// for this investigation.
	GrafanaURL string `json:"grafanaUrl"`

	// Status describes the state of the investigation (pending, running, failed, or finished).
	Status investigationStatus `json:"status"`

	// FailureReason is a short human-friendly string that explains the reason that the
	// investigation failed.
	FailureReason string `json:"failureReason,omitempty"`

	Analyses analysisMeta `json:"analyses"`

	Datasources InvestigationDatasources `json:"datasources"`
}

// siftClient represents a client for interacting with the Sift API.
type siftClient struct {
	client *http.Client
	url    string
}

func newSiftClient(cfg mcpgrafana.GrafanaConfig) (*siftClient, error) {
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
		Transport: transport,
	}
	return &siftClient{
		client: client,
		url:    cfg.URL,
	}, nil
}

func siftClientFromContext(ctx context.Context) (*siftClient, error) {
	// Get the standard Grafana URL and API key
	cfg := mcpgrafana.GrafanaConfigFromContext(ctx)
	client, err := newSiftClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating Sift client: %w", err)
	}
	return client, nil
}

// checkType represents the type of analysis check to perform.
type checkType string

const (
	checkTypeErrorPatternLogs checkType = "ErrorPatternLogs"
	checkTypeSlowRequests     checkType = "SlowRequests"
)

// GetSiftInvestigationParams defines the parameters for retrieving an investigation
type GetSiftInvestigationParams struct {
	ID string `json:"id" jsonschema:"required,description=The UUID of the investigation as a string (e.g. '02adab7c-bf5b-45f2-9459-d71a2c29e11b')"`
}

// getSiftInvestigation retrieves an existing investigation
func getSiftInvestigation(ctx context.Context, args GetSiftInvestigationParams) (*Investigation, error) {
	client, err := siftClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating Sift client: %w", err)
	}

	// Parse the UUID string
	id, err := uuid.Parse(args.ID)
	if err != nil {
		return nil, fmt.Errorf("invalid investigation ID format: %w", err)
	}

	investigation, err := client.getSiftInvestigation(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("getting investigation: %w", err)
	}

	return investigation, nil
}

// GetSiftInvestigation is a tool for retrieving an existing investigation
var GetSiftInvestigation = mcpgrafana.MustTool(
	"get_sift_investigation",
	"Retrieves an existing Sift investigation by its UUID. The ID should be provided as a string in UUID format (e.g. '02adab7c-bf5b-45f2-9459-d71a2c29e11b').",
	getSiftInvestigation,
	mcp.WithTitleAnnotation("Get Sift investigation"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// GetSiftAnalysisParams defines the parameters for retrieving a specific analysis
type GetSiftAnalysisParams struct {
	InvestigationID string `json:"investigationId" jsonschema:"required,description=The UUID of the investigation as a string (e.g. '02adab7c-bf5b-45f2-9459-d71a2c29e11b')"`
	AnalysisID      string `json:"analysisId" jsonschema:"required,description=The UUID of the specific analysis to retrieve"`
}

// getSiftAnalysis retrieves a specific analysis from an investigation
func getSiftAnalysis(ctx context.Context, args GetSiftAnalysisParams) (*analysis, error) {
	client, err := siftClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating Sift client: %w", err)
	}

	// Parse the UUID strings
	investigationID, err := uuid.Parse(args.InvestigationID)
	if err != nil {
		return nil, fmt.Errorf("invalid investigation ID format: %w", err)
	}

	analysisID, err := uuid.Parse(args.AnalysisID)
	if err != nil {
		return nil, fmt.Errorf("invalid analysis ID format: %w", err)
	}

	analysis, err := client.getSiftAnalysis(ctx, investigationID, analysisID)
	if err != nil {
		return nil, fmt.Errorf("getting analysis: %w", err)
	}

	return analysis, nil
}

// GetSiftAnalysis is a tool for retrieving a specific analysis from an investigation
var GetSiftAnalysis = mcpgrafana.MustTool(
	"get_sift_analysis",
	"Retrieves a specific analysis from an investigation by its UUID. The investigation ID and analysis ID should be provided as strings in UUID format.",
	getSiftAnalysis,
	mcp.WithTitleAnnotation("Get Sift analysis"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// ListSiftInvestigationsParams defines the parameters for retrieving investigations
type ListSiftInvestigationsParams struct {
	Limit int `json:"limit,omitempty" jsonschema:"default=10,description=Maximum number of investigations to return"`
}

// listSiftInvestigations retrieves a list of investigations with an optional limit
func listSiftInvestigations(ctx context.Context, args ListSiftInvestigationsParams) ([]Investigation, error) {
	client, err := siftClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating Sift client: %w", err)
	}

	// Set default limit if not provided
	if args.Limit <= 0 {
		args.Limit = 10
	}

	investigations, err := client.listSiftInvestigations(ctx, args.Limit)
	if err != nil {
		return nil, fmt.Errorf("getting investigations: %w", err)
	}

	return investigations, nil
}

// ListSiftInvestigations is a tool for retrieving a list of investigations
var ListSiftInvestigations = mcpgrafana.MustTool(
	"list_sift_investigations",
	"Retrieves a list of Sift investigations with an optional limit. If no limit is specified, defaults to 10 investigations.",
	listSiftInvestigations,
	mcp.WithTitleAnnotation("List Sift investigations"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// FindErrorPatternLogsParams defines the parameters for running an ErrorPatternLogs check
type FindErrorPatternLogsParams struct {
	Name   string            `json:"name" jsonschema:"required,description=The name of the investigation"`
	Labels map[string]string `json:"labels" jsonschema:"required,description=Labels to scope the analysis"`
	Start  time.Time         `json:"start,omitempty" jsonschema:"description=Start time for the investigation. Defaults to 30 minutes ago if not specified."`
	End    time.Time         `json:"end,omitempty" jsonschema:"description=End time for the investigation. Defaults to now if not specified."`
}

// findErrorPatternLogs creates an investigation with ErrorPatternLogs check, waits for it to complete, and returns the analysis
func findErrorPatternLogs(ctx context.Context, args FindErrorPatternLogsParams) (*analysis, error) {
	client, err := siftClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating Sift client: %w", err)
	}

	// Create the investigation request with ErrorPatternLogs check
	requestData := investigationRequest{
		Labels: args.Labels,
		Start:  args.Start,
		End:    args.End,
		Checks: []string{string(checkTypeErrorPatternLogs)},
	}

	investigation := &Investigation{
		Name:       args.Name,
		GrafanaURL: client.url,
		Status:     investigationStatusPending,
	}

	// Create the investigation and wait for it to complete
	completedInvestigation, err := client.createSiftInvestigation(ctx, investigation, requestData)
	if err != nil {
		return nil, fmt.Errorf("creating investigation: %w", err)
	}

	// Get all analyses from the completed investigation
	slog.Debug("Getting analyses", "investigation_id", completedInvestigation.ID)
	analyses, err := client.getSiftAnalyses(ctx, completedInvestigation.ID)
	if err != nil {
		return nil, fmt.Errorf("getting analyses: %w", err)
	}

	// Find the ErrorPatternLogs analysis
	var errorPatternLogsAnalysis *analysis
	for i := range analyses {
		if analyses[i].Name == string(checkTypeErrorPatternLogs) {
			errorPatternLogsAnalysis = &analyses[i]
			break
		}
	}

	if errorPatternLogsAnalysis == nil {
		return nil, fmt.Errorf("ErrorPatternLogs analysis not found in investigation %s", completedInvestigation.ID)
	}
	slog.Debug("Found ErrorPatternLogs analysis", "analysis_id", errorPatternLogsAnalysis.ID)

	datasourceUID := completedInvestigation.Datasources.LokiDatasource.UID

	if errorPatternLogsAnalysis.Result.Details == nil {
		// No patterns found, return the analysis without examples
		return errorPatternLogsAnalysis, nil
	}
	for _, pattern := range errorPatternLogsAnalysis.Result.Details["patterns"].([]any) {
		patternMap, ok := pattern.(map[string]any)
		if !ok {
			continue
		}
		examples, err := fetchErrorPatternLogExamples(ctx, patternMap, datasourceUID)
		if err != nil {
			return nil, err
		}
		patternMap["examples"] = examples
	}

	return errorPatternLogsAnalysis, nil
}

// FindErrorPatternLogs is a tool for running an ErrorPatternLogs check
var FindErrorPatternLogs = mcpgrafana.MustTool(
	"find_error_pattern_logs",
	"Searches Loki logs for elevated error patterns compared to the last day's average, waits for the analysis to complete, and returns the results including any patterns found.",
	findErrorPatternLogs,
	mcp.WithTitleAnnotation("Find error patterns in logs"),
	mcp.WithReadOnlyHintAnnotation(true),
)

// FindSlowRequestsParams defines the parameters for running an SlowRequests check
type FindSlowRequestsParams struct {
	Name   string            `json:"name" jsonschema:"required,description=The name of the investigation"`
	Labels map[string]string `json:"labels" jsonschema:"required,description=Labels to scope the analysis"`
	Start  time.Time         `json:"start,omitempty" jsonschema:"description=Start time for the investigation. Defaults to 30 minutes ago if not specified."`
	End    time.Time         `json:"end,omitempty" jsonschema:"description=End time for the investigation. Defaults to now if not specified."`
}

// findSlowRequests creates an investigation with SlowRequests check, waits for it to complete, and returns the analysis
func findSlowRequests(ctx context.Context, args FindSlowRequestsParams) (*analysis, error) {
	client, err := siftClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating Sift client: %w", err)
	}

	// Create the investigation request with SlowRequests check
	requestData := investigationRequest{
		Labels: args.Labels,
		Start:  args.Start,
		End:    args.End,
		Checks: []string{string(checkTypeSlowRequests)},
	}

	investigation := &Investigation{
		Name:       args.Name,
		GrafanaURL: client.url,
		Status:     investigationStatusPending,
	}

	// Create the investigation and wait for it to complete
	completedInvestigation, err := client.createSiftInvestigation(ctx, investigation, requestData)
	if err != nil {
		return nil, fmt.Errorf("creating investigation: %w", err)
	}

	// Get all analyses from the completed investigation
	analyses, err := client.getSiftAnalyses(ctx, completedInvestigation.ID)
	if err != nil {
		return nil, fmt.Errorf("getting analyses: %w", err)
	}

	// Find the SlowRequests analysis
	var slowRequestsAnalysis *analysis
	for i := range analyses {
		if analyses[i].Name == string(checkTypeSlowRequests) {
			slowRequestsAnalysis = &analyses[i]
			break
		}
	}

	if slowRequestsAnalysis == nil {
		return nil, fmt.Errorf("SlowRequests analysis not found in investigation %s", completedInvestigation.ID)
	}

	return slowRequestsAnalysis, nil
}

// FindSlowRequests is a tool for running an SlowRequests check
var FindSlowRequests = mcpgrafana.MustTool(
	"find_slow_requests",
	"Searches relevant Tempo datasources for slow requests, waits for the analysis to complete, and returns the results.",
	findSlowRequests,
	mcp.WithTitleAnnotation("Find slow requests"),
	mcp.WithReadOnlyHintAnnotation(true),
)

// AddSiftTools registers all Sift tools with the MCP server
func AddSiftTools(mcp *server.MCPServer, enableWriteTools bool) {
	GetSiftInvestigation.Register(mcp)
	GetSiftAnalysis.Register(mcp)
	ListSiftInvestigations.Register(mcp)
	if enableWriteTools {
		FindErrorPatternLogs.Register(mcp)
		FindSlowRequests.Register(mcp)
	}
}

// makeRequest is a helper method to make HTTP requests and handle common response patterns
func (c *siftClient) makeRequest(ctx context.Context, method, path string, body []byte) ([]byte, error) {
	var req *http.Request
	var err error

	if body != nil {
		req, err = http.NewRequestWithContext(ctx, method, c.url+path, bytes.NewBuffer(body))
		if err != nil {
			return nil, fmt.Errorf("creating request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
	} else {
		req, err = http.NewRequestWithContext(ctx, method, c.url+path, nil)
		if err != nil {
			return nil, fmt.Errorf("creating request: %w", err)
		}
	}

	response, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() {
		_ = response.Body.Close() //nolint:errcheck
	}()

	// Check for non-200 status code (matching Loki client's logic)
	if response.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(response.Body) // Read full body on error
		return nil, fmt.Errorf("API request returned status code %d: %s", response.StatusCode, string(bodyBytes))
	}

	// Read the response body with a limit to prevent memory issues
	reader := io.LimitReader(response.Body, 1024*1024*48) // 48MB limit
	buf, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check if the response is empty (matching Loki client's logic)
	if len(buf) == 0 {
		return nil, fmt.Errorf("empty response from API")
	}

	// Trim any whitespace that might cause JSON parsing issues (matching Loki client's logic)
	return bytes.TrimSpace(buf), nil
}

// getSiftInvestigation is a helper method to get the current status of an investigation
func (c *siftClient) getSiftInvestigation(ctx context.Context, id uuid.UUID) (*Investigation, error) {
	buf, err := c.makeRequest(ctx, "GET", fmt.Sprintf("/api/plugins/grafana-ml-app/resources/sift/api/v1/investigations/%s", id), nil)
	if err != nil {
		return nil, err
	}

	investigationResponse := struct {
		Status string        `json:"status"`
		Data   Investigation `json:"data"`
	}{}

	if err := json.Unmarshal(buf, &investigationResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response body: %w. body: %s", err, buf)
	}

	return &investigationResponse.Data, nil
}

func (c *siftClient) createSiftInvestigation(ctx context.Context, investigation *Investigation, requestData investigationRequest) (*Investigation, error) {
	// Set default time range to last 30 minutes if not provided
	if requestData.Start.IsZero() {
		requestData.Start = time.Now().Add(-30 * time.Minute)
	}
	if requestData.End.IsZero() {
		requestData.End = time.Now()
	}

	// Create the payload including the necessary fields for the API
	payload := struct {
		Investigation
		RequestData investigationRequest `json:"requestData"`
	}{
		Investigation: *investigation,
		RequestData:   requestData,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshaling investigation: %w", err)
	}

	slog.Debug("Creating investigation", "payload", string(jsonData))
	buf, err := c.makeRequest(ctx, "POST", "/api/plugins/grafana-ml-app/resources/sift/api/v1/investigations", jsonData)
	if err != nil {
		return nil, err
	}
	slog.Debug("Investigation created", "response", string(buf))

	investigationResponse := struct {
		Status string        `json:"status"`
		Data   Investigation `json:"data"`
	}{}

	if err := json.Unmarshal(buf, &investigationResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response body: %w. body: %s", err, buf)
	}

	// Poll for investigation completion
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	timeout := time.After(5 * time.Minute)

	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled while waiting for investigation completion")
		case <-timeout:
			return nil, fmt.Errorf("timeout waiting for investigation completion after 5 minutes")
		case <-ticker.C:
			slog.Debug("Polling investigation status", "investigation_id", investigationResponse.Data.ID)
			investigation, err := c.getSiftInvestigation(ctx, investigationResponse.Data.ID)
			if err != nil {
				return nil, err
			}

			if investigation.Status == investigationStatusFailed {
				return nil, fmt.Errorf("investigation failed: %s", investigation.FailureReason)
			}

			if investigation.Status == investigationStatusFinished {
				return investigation, nil
			}
		}
	}
}

// getSiftAnalyses is a helper method to get all analyses from an investigation
func (c *siftClient) getSiftAnalyses(ctx context.Context, investigationID uuid.UUID) ([]analysis, error) {
	path := fmt.Sprintf("/api/plugins/grafana-ml-app/resources/sift/api/v1/investigations/%s/analyses", investigationID)
	buf, err := c.makeRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}

	var response struct {
		Status string     `json:"status"`
		Data   []analysis `json:"data"`
	}

	if err := json.Unmarshal(buf, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response body: %w. body: %s", err, buf)
	}

	return response.Data, nil
}

// getSiftAnalysis is a helper method to get a specific analysis from an investigation
func (c *siftClient) getSiftAnalysis(ctx context.Context, investigationID, analysisID uuid.UUID) (*analysis, error) {
	// First get all analyses to verify the analysis exists
	analyses, err := c.getSiftAnalyses(ctx, investigationID)
	if err != nil {
		return nil, fmt.Errorf("getting analyses: %w", err)
	}

	// Find the specific analysis
	var targetAnalysis *analysis
	for _, analysis := range analyses {
		if analysis.ID == analysisID {
			targetAnalysis = &analysis
			break
		}
	}

	if targetAnalysis == nil {
		return nil, fmt.Errorf("analysis with ID %s not found in investigation %s", analysisID, investigationID)
	}

	return targetAnalysis, nil
}

// listSiftInvestigations is a helper method to get a list of investigations
func (c *siftClient) listSiftInvestigations(ctx context.Context, limit int) ([]Investigation, error) {
	path := fmt.Sprintf("/api/plugins/grafana-ml-app/resources/sift/api/v1/investigations?limit=%d", limit)
	buf, err := c.makeRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}

	var response struct {
		Status string          `json:"status"`
		Data   []Investigation `json:"data"`
	}

	if err := json.Unmarshal(buf, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response body: %w. body: %s", err, buf)
	}

	return response.Data, nil
}

func fetchErrorPatternLogExamples(ctx context.Context, patternMap map[string]any, datasourceUID string) ([]string, error) {
	query, _ := patternMap["query"].(string)
	logEntries, err := queryLokiLogs(ctx, QueryLokiLogsParams{
		DatasourceUID: datasourceUID,
		LogQL:         query,
		Limit:         errorPatternLogExampleLimit,
	})
	if err != nil {
		return nil, fmt.Errorf("querying Loki: %w", err)
	}
	var examples []string
	for _, entry := range logEntries {
		if entry.Line != "" {
			examples = append(examples, entry.Line)
		}
	}
	return examples, nil
}
