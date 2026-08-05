package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strconv"
	"strings"

	aapi "github.com/grafana/amixr-api-go-client"
	"github.com/grafana/grafana-openapi-client-go/client"
	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// getOnCallURLFromSettings retrieves the OnCall API URL from the Grafana settings endpoint.
// It makes a GET request to <grafana-url>/api/plugins/grafana-irm-app/settings and extracts
// the OnCall URL from the jsonData.onCallApiUrl field in the response.
// Returns the OnCall URL if found, or an error if the URL cannot be retrieved.
func getOnCallURLFromSettings(ctx context.Context, cfg mcpgrafana.GrafanaConfig) (string, error) {
	settingsURL := fmt.Sprintf("%s/api/plugins/grafana-irm-app/settings", strings.TrimRight(cfg.URL, "/"))

	req, err := http.NewRequestWithContext(ctx, "GET", settingsURL, nil)
	if err != nil {
		return "", fmt.Errorf("creating settings request: %w", err)
	}

	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	} else if cfg.BasicAuth != nil {
		password, _ := cfg.BasicAuth.Password()
		req.SetBasicAuth(cfg.BasicAuth.Username(), password)
	}

	// Add org ID header for multi-org support
	if cfg.OrgID > 0 {
		req.Header.Set(client.OrgIDHeader, strconv.FormatInt(cfg.OrgID, 10))
	}

	// Add user agent for tracking
	req.Header.Set("User-Agent", mcpgrafana.UserAgent())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetching settings: %w", err)
	}
	defer func() {
		_ = resp.Body.Close() //nolint:errcheck
	}()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status code from settings API: %d", resp.StatusCode)
	}

	var settings struct {
		JSONData struct {
			OnCallAPIURL string `json:"onCallApiUrl"`
		} `json:"jsonData"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&settings); err != nil {
		return "", fmt.Errorf("decoding settings response: %w", err)
	}

	if settings.JSONData.OnCallAPIURL == "" {
		return "", fmt.Errorf("OnCall API URL is not set in settings")
	}

	return settings.JSONData.OnCallAPIURL, nil
}

func oncallClientFromContext(ctx context.Context) (*aapi.Client, error) {
	// Get the standard Grafana URL and API key
	cfg := mcpgrafana.GrafanaConfigFromContext(ctx)

	// Try to get OnCall URL from settings endpoint
	grafanaOnCallURL, err := getOnCallURLFromSettings(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall URL from settings: %w", err)
	}

	grafanaOnCallURL = strings.TrimRight(grafanaOnCallURL, "/")

	// TODO: Allow access to OnCall using an access token instead of an API key.
	client, err := aapi.NewWithGrafanaURL(grafanaOnCallURL, cfg.APIKey, cfg.URL)
	if err != nil {
		return nil, fmt.Errorf("creating OnCall client: %w", err)
	}

	// Try to customize the HTTP client with user agent using reflection
	// since the OnCall client doesn't expose its HTTP client directly
	clientValue := reflect.ValueOf(client)
	if clientValue.Kind() == reflect.Ptr && !clientValue.IsNil() {
		clientValue = clientValue.Elem()
		if clientValue.Kind() == reflect.Struct {
			httpClientField := clientValue.FieldByName("HTTPClient")
			if !httpClientField.IsValid() {
				// Try alternative field names
				httpClientField = clientValue.FieldByName("HttpClient")
			}
			if !httpClientField.IsValid() {
				httpClientField = clientValue.FieldByName("Client")
			}
			if httpClientField.IsValid() && httpClientField.CanSet() {
				if httpClient, ok := httpClientField.Interface().(*http.Client); ok {
					// Wrap the transport with user agent
					if httpClient.Transport == nil {
						httpClient.Transport = http.DefaultTransport
					}
					httpClient.Transport = mcpgrafana.NewUserAgentTransport(
						httpClient.Transport,
					)
				}
			}
		}
	}

	return client, nil
}

// getUserServiceFromContext creates a new UserService using the OnCall client from the context
func getUserServiceFromContext(ctx context.Context) (*aapi.UserService, error) {
	client, err := oncallClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall client: %w", err)
	}

	return aapi.NewUserService(client), nil
}

// getScheduleServiceFromContext creates a new ScheduleService using the OnCall client from the context
func getScheduleServiceFromContext(ctx context.Context) (*aapi.ScheduleService, error) {
	client, err := oncallClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall client: %w", err)
	}

	return aapi.NewScheduleService(client), nil
}

// getTeamServiceFromContext creates a new TeamService using the OnCall client from the context
func getTeamServiceFromContext(ctx context.Context) (*aapi.TeamService, error) {
	client, err := oncallClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall client: %w", err)
	}

	return aapi.NewTeamService(client), nil
}

// getOnCallShiftServiceFromContext creates a new OnCallShiftService using the OnCall client from the context
func getOnCallShiftServiceFromContext(ctx context.Context) (*aapi.OnCallShiftService, error) {
	client, err := oncallClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall client: %w", err)
	}

	return aapi.NewOnCallShiftService(client), nil
}

type ListOnCallSchedulesParams struct {
	TeamID     string `json:"teamId,omitempty" jsonschema:"description=The ID of the team to list schedules for"`
	ScheduleID string `json:"scheduleId,omitempty" jsonschema:"description=The ID of the schedule to get details for. If provided\\, returns only that schedule's details"`
	Page       int    `json:"page,omitempty" jsonschema:"description=The page number to return (1-based)"`
}

// ScheduleSummary represents a simplified view of an OnCall schedule
type ScheduleSummary struct {
	ID       string   `json:"id" jsonschema:"description=The unique identifier of the schedule"`
	Name     string   `json:"name" jsonschema:"description=The name of the schedule"`
	TeamID   string   `json:"teamId" jsonschema:"description=The ID of the team this schedule belongs to"`
	Timezone string   `json:"timezone" jsonschema:"description=The timezone for this schedule"`
	Shifts   []string `json:"shifts" jsonschema:"description=List of shift IDs in this schedule"`
}

func listOnCallSchedules(ctx context.Context, args ListOnCallSchedulesParams) ([]*ScheduleSummary, error) {
	scheduleService, err := getScheduleServiceFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall schedule service: %w", err)
	}

	if args.ScheduleID != "" {
		schedule, _, err := scheduleService.GetSchedule(args.ScheduleID, &aapi.GetScheduleOptions{})
		if err != nil {
			return nil, fmt.Errorf("getting OnCall schedule %s: %w", args.ScheduleID, err)
		}
		summary := &ScheduleSummary{
			ID:       schedule.ID,
			Name:     schedule.Name,
			TeamID:   schedule.TeamId,
			Timezone: schedule.TimeZone,
		}
		if schedule.Shifts != nil {
			summary.Shifts = *schedule.Shifts
		}
		return []*ScheduleSummary{summary}, nil
	}

	listOptions := &aapi.ListScheduleOptions{}
	if args.Page > 0 {
		listOptions.Page = args.Page
	}
	if args.TeamID != "" {
		listOptions.TeamID = args.TeamID
	}

	response, _, err := scheduleService.ListSchedules(listOptions)
	if err != nil {
		return nil, fmt.Errorf("listing OnCall schedules: %w", err)
	}

	// Convert schedules to summaries
	summaries := make([]*ScheduleSummary, 0, len(response.Schedules))
	for _, schedule := range response.Schedules {
		summary := &ScheduleSummary{
			ID:       schedule.ID,
			Name:     schedule.Name,
			TeamID:   schedule.TeamId,
			Timezone: schedule.TimeZone,
		}
		if schedule.Shifts != nil {
			summary.Shifts = *schedule.Shifts
		}
		summaries = append(summaries, summary)
	}

	return summaries, nil
}

var ListOnCallSchedules = mcpgrafana.MustTool(
	"list_oncall_schedules",
	"List Grafana OnCall schedules, optionally filtering by team ID. If a specific schedule ID is provided, retrieves details for only that schedule. Returns a list of schedule summaries including ID, name, team ID, timezone, and shift IDs. Supports pagination.",
	listOnCallSchedules,
	mcp.WithTitleAnnotation("List OnCall schedules"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type GetOnCallShiftParams struct {
	ShiftID string `json:"shiftId" jsonschema:"required,description=The ID of the shift to get details for"`
}

func getOnCallShift(ctx context.Context, args GetOnCallShiftParams) (*aapi.OnCallShift, error) {
	shiftService, err := getOnCallShiftServiceFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall shift service: %w", err)
	}

	shift, _, err := shiftService.GetOnCallShift(args.ShiftID, &aapi.GetOnCallShiftOptions{})
	if err != nil {
		return nil, fmt.Errorf("getting OnCall shift %s: %w", args.ShiftID, err)
	}

	return shift, nil
}

var GetOnCallShift = mcpgrafana.MustTool(
	"get_oncall_shift",
	"Get detailed information for a specific Grafana OnCall shift using its ID. A shift represents a designated time period within a schedule when users are actively on-call. Returns the full shift details.",
	getOnCallShift,
	mcp.WithTitleAnnotation("Get OnCall shift"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// CurrentOnCallUsers represents the currently on-call users for a schedule
type CurrentOnCallUsers struct {
	ScheduleID   string       `json:"scheduleId" jsonschema:"description=The ID of the schedule"`
	ScheduleName string       `json:"scheduleName" jsonschema:"description=The name of the schedule"`
	Users        []*aapi.User `json:"users" jsonschema:"description=List of users currently on call"`
}

type GetCurrentOnCallUsersParams struct {
	ScheduleID string `json:"scheduleId" jsonschema:"required,description=The ID of the schedule to get current on-call users for"`
}

func getCurrentOnCallUsers(ctx context.Context, args GetCurrentOnCallUsersParams) (*CurrentOnCallUsers, error) {
	scheduleService, err := getScheduleServiceFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall schedule service: %w", err)
	}

	schedule, _, err := scheduleService.GetSchedule(args.ScheduleID, &aapi.GetScheduleOptions{})
	if err != nil {
		return nil, fmt.Errorf("getting schedule %s: %w", args.ScheduleID, err)
	}

	// Create the result with the schedule info
	result := &CurrentOnCallUsers{
		ScheduleID:   schedule.ID,
		ScheduleName: schedule.Name,
		Users:        make([]*aapi.User, 0, len(schedule.OnCallNow)),
	}

	// If there are no users on call, return early
	if len(schedule.OnCallNow) == 0 {
		return result, nil
	}

	// Get the user service to fetch user details
	userService, err := getUserServiceFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall user service: %w", err)
	}

	// Fetch details for each user currently on call
	for _, userID := range schedule.OnCallNow {
		user, _, err := userService.GetUser(userID, &aapi.GetUserOptions{})
		if err != nil {
			// Log the error but continue with other users
			fmt.Printf("Error fetching user %s: %v\n", userID, err)
			continue
		}
		result.Users = append(result.Users, user)
	}

	return result, nil
}

var GetCurrentOnCallUsers = mcpgrafana.MustTool(
	"get_current_oncall_users",
	"Get the list of users currently on-call for a specific Grafana OnCall schedule ID. Returns the schedule ID, name, and a list of detailed user objects for those currently on call.",
	getCurrentOnCallUsers,
	mcp.WithTitleAnnotation("Get current on-call users"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type ListOnCallTeamsParams struct {
	Page int `json:"page,omitempty" jsonschema:"description=The page number to return"`
}

func listOnCallTeams(ctx context.Context, args ListOnCallTeamsParams) ([]*aapi.Team, error) {
	teamService, err := getTeamServiceFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall team service: %w", err)
	}

	listOptions := &aapi.ListTeamOptions{}
	if args.Page > 0 {
		listOptions.Page = args.Page
	}

	response, _, err := teamService.ListTeams(listOptions)
	if err != nil {
		return nil, fmt.Errorf("listing OnCall teams: %w", err)
	}

	return response.Teams, nil
}

var ListOnCallTeams = mcpgrafana.MustTool(
	"list_oncall_teams",
	"List teams configured in Grafana OnCall. Returns a list of team objects with their details. Supports pagination.",
	listOnCallTeams,
	mcp.WithTitleAnnotation("List OnCall teams"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type ListOnCallUsersParams struct {
	UserID   string `json:"userId,omitempty" jsonschema:"description=The ID of the user to get details for. If provided\\, returns only that user's details"`
	Username string `json:"username,omitempty" jsonschema:"description=The username to filter users by. If provided\\, returns only the user matching this username"`
	Page     int    `json:"page,omitempty" jsonschema:"description=The page number to return"`
}

func listOnCallUsers(ctx context.Context, args ListOnCallUsersParams) ([]*aapi.User, error) {
	userService, err := getUserServiceFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall user service: %w", err)
	}

	if args.UserID != "" {
		user, _, err := userService.GetUser(args.UserID, &aapi.GetUserOptions{})
		if err != nil {
			return nil, fmt.Errorf("getting OnCall user %s: %w", args.UserID, err)
		}
		return []*aapi.User{user}, nil
	}

	// Otherwise, list all users
	listOptions := &aapi.ListUserOptions{}
	if args.Page > 0 {
		listOptions.Page = args.Page
	}
	if args.Username != "" {
		listOptions.Username = args.Username
	}

	response, _, err := userService.ListUsers(listOptions)
	if err != nil {
		return nil, fmt.Errorf("listing OnCall users: %w", err)
	}

	return response.Users, nil
}

var ListOnCallUsers = mcpgrafana.MustTool(
	"list_oncall_users",
	"List users from Grafana OnCall. Can retrieve all users, a specific user by ID, or filter by username. Returns a list of user objects with their details. Supports pagination.",
	listOnCallUsers,
	mcp.WithTitleAnnotation("List OnCall users"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

func getAlertGroupServiceFromContext(ctx context.Context) (*aapi.AlertGroupService, error) {
	client, err := oncallClientFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall client: %w", err)
	}

	return aapi.NewAlertGroupService(client), nil
}

type ListAlertGroupsParams struct {
	Page          int      `json:"page,omitempty" jsonschema:"description=The page number to return"`
	AlertGroupID  string   `json:"id,omitempty" jsonschema:"description=Filter by specific alert group ID"`
	RouteID       string   `json:"routeId,omitempty" jsonschema:"description=Filter by route ID"`
	IntegrationID string   `json:"integrationId,omitempty" jsonschema:"description=Filter by integration ID"`
	State         string   `json:"state,omitempty" jsonschema:"description=Filter by alert group state (one of: new\\, acknowledged\\, resolved\\, silenced)"`
	TeamID        string   `json:"teamId,omitempty" jsonschema:"description=Filter by team ID"`
	StartedAt     string   `json:"startedAt,omitempty" jsonschema:"description=Filter by time range in format '{start}_{end}' ISO 8601 timestamp range (UTC assumed\\, no timezone indicator needed) (e.g.\\, '2025-01-19T00:00:00_2025-01-19T23:59:59')"`
	Labels        []string `json:"labels,omitempty" jsonschema:"description=Filter by labels in format key:value (e.g.\\, ['env:prod'\\, 'severity:high'])"`
	Name          string   `json:"name,omitempty" jsonschema:"description=Filter by alert group name"`
}

func listAlertGroups(ctx context.Context, args ListAlertGroupsParams) ([]*aapi.AlertGroup, error) {
	alertGroupService, err := getAlertGroupServiceFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall alert group service: %w", err)
	}

	listOptions := &aapi.ListAlertGroupOptions{}
	if args.Page > 0 {
		listOptions.Page = args.Page
	}
	if args.AlertGroupID != "" {
		listOptions.AlertGroupID = args.AlertGroupID
	}
	if args.RouteID != "" {
		listOptions.RouteID = args.RouteID
	}
	if args.IntegrationID != "" {
		listOptions.IntegrationID = args.IntegrationID
	}
	if args.State != "" {
		listOptions.State = args.State
	}
	if args.TeamID != "" {
		listOptions.TeamID = args.TeamID
	}
	if args.StartedAt != "" {
		listOptions.StartedAt = args.StartedAt
	}
	if len(args.Labels) > 0 {
		listOptions.Labels = args.Labels
	}
	if args.Name != "" {
		listOptions.Name = args.Name
	}

	response, _, err := alertGroupService.ListAlertGroups(listOptions)
	if err != nil {
		return nil, fmt.Errorf("listing OnCall alert groups: %w", err)
	}

	return response.AlertGroups, nil
}

var ListAlertGroups = mcpgrafana.MustTool(
	"list_alert_groups",
	"List alert groups from Grafana OnCall with filtering options. Supports filtering by alert group ID, route ID, integration ID, state (new, acknowledged, resolved, silenced), team ID, time range, labels, and name. For time ranges, use format '{start}_{end}' ISO 8601 timestamp range (e.g., '2025-01-19T00:00:00_2025-01-19T23:59:59' for a specific day). For labels, use format 'key:value' (e.g., ['env:prod', 'severity:high']). Returns a list of alert group objects with their details. Supports pagination.",
	listAlertGroups,
	mcp.WithTitleAnnotation("List IRM alert groups"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type GetAlertGroupParams struct {
	AlertGroupID string `json:"alertGroupId" jsonschema:"required,description=The ID of the alert group to retrieve"`
}

func getAlertGroup(ctx context.Context, args GetAlertGroupParams) (*aapi.AlertGroup, error) {
	alertGroupService, err := getAlertGroupServiceFromContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall alert group service: %w", err)
	}

	alertGroup, _, err := alertGroupService.GetAlertGroup(args.AlertGroupID)
	if err != nil {
		return nil, fmt.Errorf("getting OnCall alert group %s: %w", args.AlertGroupID, err)
	}

	return alertGroup, nil
}

var GetAlertGroup = mcpgrafana.MustTool(
	"get_alert_group",
	"Get a specific alert group from Grafana OnCall by its ID. Returns the full alert group details.",
	getAlertGroup,
	mcp.WithTitleAnnotation("Get IRM alert group"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

func AddOnCallTools(mcp *server.MCPServer) {
	ListOnCallSchedules.Register(mcp)
	GetOnCallShift.Register(mcp)
	GetCurrentOnCallUsers.Register(mcp)
	ListOnCallTeams.Register(mcp)
	ListOnCallUsers.Register(mcp)
	ListAlertGroups.Register(mcp)
	GetAlertGroup.Register(mcp)
}
