package tools

import (
	"context"
	"fmt"

	"github.com/grafana/incident-go"
	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

type ListIncidentsParams struct {
	Limit  int    `json:"limit" jsonschema:"default=10,description=The maximum number of incidents to return"`
	Drill  bool   `json:"drill" jsonschema:"description=Whether to include drill incidents"`
	Status string `json:"status" jsonschema:"description=The status of the incidents to include. Valid values: 'active'\\, 'resolved'"`
}

func listIncidents(ctx context.Context, args ListIncidentsParams) (*incident.QueryIncidentPreviewsResponse, error) {
	c := mcpgrafana.IncidentClientFromContext(ctx)
	is := incident.NewIncidentsService(c)

	// Set default limit to 10 if not specified
	limit := args.Limit
	if limit <= 0 {
		limit = 10
	}

	query := ""
	if !args.Drill {
		query = "isdrill:false"
	}
	if args.Status != "" {
		query += fmt.Sprintf(" status:%s", args.Status)
	}
	incidents, err := is.QueryIncidentPreviews(ctx, incident.QueryIncidentPreviewsRequest{
		Query: incident.IncidentPreviewsQuery{
			QueryString:    query,
			OrderDirection: "DESC",
			Limit:          limit,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("list incidents: %w", err)
	}
	return incidents, nil
}

var ListIncidents = mcpgrafana.MustTool(
	"list_incidents",
	"List Grafana incidents. Allows filtering by status ('active', 'resolved') and optionally including drill incidents. Returns a preview list with basic details.",
	listIncidents,
	mcp.WithTitleAnnotation("List incidents"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type CreateIncidentParams struct {
	Title         string                   `json:"title" jsonschema:"description=The title of the incident"`
	Severity      string                   `json:"severity" jsonschema:"description=The severity of the incident"`
	RoomPrefix    string                   `json:"roomPrefix" jsonschema:"description=The prefix of the room to create the incident in"`
	IsDrill       bool                     `json:"isDrill" jsonschema:"description=Whether the incident is a drill incident"`
	Status        string                   `json:"status" jsonschema:"description=The status of the incident"`
	AttachCaption string                   `json:"attachCaption" jsonschema:"description=The caption of the attachment"`
	AttachURL     string                   `json:"attachUrl" jsonschema:"description=The URL of the attachment"`
	Labels        []incident.IncidentLabel `json:"labels" jsonschema:"description=The labels to add to the incident"`
}

func createIncident(ctx context.Context, args CreateIncidentParams) (*incident.Incident, error) {
	c := mcpgrafana.IncidentClientFromContext(ctx)
	is := incident.NewIncidentsService(c)
	incident, err := is.CreateIncident(ctx, incident.CreateIncidentRequest{
		Title:         args.Title,
		Severity:      args.Severity,
		RoomPrefix:    args.RoomPrefix,
		IsDrill:       args.IsDrill,
		Status:        args.Status,
		AttachCaption: args.AttachCaption,
		AttachURL:     args.AttachURL,
		Labels:        args.Labels,
	})
	if err != nil {
		return nil, fmt.Errorf("create incident: %w", err)
	}
	return &incident.Incident, nil
}

var CreateIncident = mcpgrafana.MustTool(
	"create_incident",
	"Create a new Grafana incident. Requires title, severity, and room prefix. Allows setting status and labels. This tool should be used judiciously and sparingly, and only after confirmation from the user, as it may notify or alarm lots of people.",
	createIncident,
	mcp.WithTitleAnnotation("Create incident"),
)

type AddActivityToIncidentParams struct {
	IncidentID string `json:"incidentId" jsonschema:"description=The ID of the incident to add the activity to"`
	Body       string `json:"body" jsonschema:"description=The body of the activity. URLs will be parsed and attached as context"`
	EventTime  string `json:"eventTime" jsonschema:"description=The time that the activity occurred. If not provided\\, the current time will be used"`
}

func addActivityToIncident(ctx context.Context, args AddActivityToIncidentParams) (*incident.ActivityItem, error) {
	c := mcpgrafana.IncidentClientFromContext(ctx)
	as := incident.NewActivityService(c)
	activity, err := as.AddActivity(ctx, incident.AddActivityRequest{
		IncidentID:   args.IncidentID,
		ActivityKind: "userNote",
		Body:         args.Body,
		EventTime:    args.EventTime,
	})
	if err != nil {
		return nil, fmt.Errorf("add activity to incident: %w", err)
	}
	return &activity.ActivityItem, nil
}

var AddActivityToIncident = mcpgrafana.MustTool(
	"add_activity_to_incident",
	"Add a note (userNote activity) to an existing incident's timeline using its ID. The note body can include URLs which will be attached as context. Use this to add context to an incident.",
	addActivityToIncident,
	mcp.WithTitleAnnotation("Add activity to incident"),
)

func AddIncidentTools(mcp *server.MCPServer, enableWriteTools bool) {
	ListIncidents.Register(mcp)
	if enableWriteTools {
		CreateIncident.Register(mcp)
		AddActivityToIncident.Register(mcp)
	}
	GetIncident.Register(mcp)
}

type GetIncidentParams struct {
	ID string `json:"id" jsonschema:"description=The ID of the incident to retrieve"`
}

func getIncident(ctx context.Context, args GetIncidentParams) (*incident.Incident, error) {
	c := mcpgrafana.IncidentClientFromContext(ctx)
	is := incident.NewIncidentsService(c)

	incidentResp, err := is.GetIncident(ctx, incident.GetIncidentRequest{
		IncidentID: args.ID,
	})
	if err != nil {
		return nil, fmt.Errorf("get incident by ID: %w", err)
	}

	return &incidentResp.Incident, nil
}

var GetIncident = mcpgrafana.MustTool(
	"get_incident",
	"Get a single incident by ID. Returns the full incident details including title, status, severity, labels, timestamps, and other metadata.",
	getIncident,
	mcp.WithTitleAnnotation("Get incident details"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)
