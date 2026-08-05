package tools

import (
	"context"
	"fmt"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"strconv"

	mcpgrafana "github.com/grafana/mcp-grafana"

	"github.com/grafana/grafana-openapi-client-go/client/annotations"
	"github.com/grafana/grafana-openapi-client-go/models"
)

// GetAnnotationsInput filters annotation search.
type GetAnnotationsInput struct {
	From         *int64   `jsonschema:"description=Epoch ms start time"`
	To           *int64   `jsonschema:"description=Epoch ms end time"`
	Limit        *int64   `jsonschema:"description=Max results default 100"`
	AlertID      *int64   `jsonschema:"description=Deprecated. Use AlertUID"`
	AlertUID     *string  `jsonschema:"description=Filter by alert UID"`
	DashboardID  *int64   `jsonschema:"description=Deprecated. Use DashboardUID"`
	DashboardUID *string  `jsonschema:"description=Filter by dashboard UID"`
	PanelID      *int64   `jsonschema:"description=Filter by panel ID"`
	UserID       *int64   `jsonschema:"description=Filter by creator user ID"`
	Type         *string  `jsonschema:"description=annotation or alert"`
	Tags         []string `jsonschema:"description=Multiple tags allowed tags=tag1&tags=tag2"`
	MatchAny     *bool    `jsonschema:"description=true OR tag match false AND"`
}

// getAnnotations retrieves Grafana annotations using filters.
func getAnnotations(ctx context.Context, args GetAnnotationsInput) (*annotations.GetAnnotationsOK, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)

	req := annotations.GetAnnotationsParams{
		From:         args.From,
		To:           args.To,
		Limit:        args.Limit,
		AlertID:      args.AlertID,
		AlertUID:     args.AlertUID,
		DashboardID:  args.DashboardID,
		DashboardUID: args.DashboardUID,
		PanelID:      args.PanelID,
		UserID:       args.UserID,
		Type:         args.Type,
		Tags:         args.Tags,
		MatchAny:     args.MatchAny,
		Context:      ctx,
	}

	resp, err := c.Annotations.GetAnnotations(&req)
	if err != nil {
		return nil, fmt.Errorf("get annotations: %w", err)
	}

	return resp, nil
}

var GetAnnotationsTool = mcpgrafana.MustTool(
	"get_annotations",
	"Fetch Grafana annotations using filters such as dashboard UID, time range and tags.",
	getAnnotations,
	mcp.WithTitleAnnotation("Get Annotations"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

// CreateAnnotationInput creates a new annotation.
type CreateAnnotationInput struct {
	DashboardID  int64          `json:"dashboardId,omitempty"  jsonschema:"description=Deprecated. Use dashboardUID"`
	DashboardUID string         `json:"dashboardUID,omitempty" jsonschema:"description=Preferred dashboard UID"`
	PanelID      int64          `json:"panelId,omitempty"      jsonschema:"description=Panel ID"`
	Time         int64          `json:"time,omitempty"         jsonschema:"description=Start time epoch ms"`
	TimeEnd      int64          `json:"timeEnd,omitempty"      jsonschema:"description=End time epoch ms"`
	Tags         []string       `json:"tags,omitempty"         jsonschema:"description=Optional list of tags"`
	Text         string         `json:"text"                   jsonschema:"description=Annotation text required"`
	Data         map[string]any `json:"data,omitempty"         jsonschema:"description=Optional JSON payload"`
}

// createAnnotation sends a POST request to create a Grafana annotation.
func createAnnotation(ctx context.Context, args CreateAnnotationInput) (*annotations.PostAnnotationOK, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)

	req := models.PostAnnotationsCmd{
		DashboardID:  args.DashboardID,
		DashboardUID: args.DashboardUID,
		PanelID:      args.PanelID,
		Time:         args.Time,
		TimeEnd:      args.TimeEnd,
		Tags:         args.Tags,
		Text:         &args.Text,
		Data:         args.Data,
	}

	resp, err := c.Annotations.PostAnnotation(&req)
	if err != nil {
		return nil, fmt.Errorf("create annotation: %w", err)
	}

	return resp, nil
}

var CreateAnnotationTool = mcpgrafana.MustTool(
	"create_annotation",
	"Create a new annotation on a dashboard or panel.",
	createAnnotation,
	mcp.WithTitleAnnotation("Create Annotation"),
	mcp.WithIdempotentHintAnnotation(false),
)

// CreateGraphiteAnnotationInput represents the payload format for creating a Graphite-style annotation.
type CreateGraphiteAnnotationInput struct {
	What string   `json:"what"  jsonschema:"description=Annotation text"`
	When int64    `json:"when"  jsonschema:"description=Epoch ms timestamp"`
	Tags []string `json:"tags,omitempty" jsonschema:"description=Optional list of tags"`
	Data string   `json:"data,omitempty" jsonschema:"description=Optional payload"`
}

// createAnnotationGraphiteFormat creates an annotation using the Graphite annotation format.
func createAnnotationGraphiteFormat(ctx context.Context, args CreateGraphiteAnnotationInput) (*annotations.PostGraphiteAnnotationOK, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)

	req := &models.PostGraphiteAnnotationsCmd{
		What: args.What,
		When: args.When,
		Tags: args.Tags,
		Data: args.Data,
	}

	resp, err := c.Annotations.PostGraphiteAnnotation(req)
	if err != nil {
		return nil, fmt.Errorf("create graphite annotation: %w", err)
	}

	return resp, nil
}

var CreateGraphiteAnnotationTool = mcpgrafana.MustTool(
	"create_graphite_annotation",
	"Create an annotation using Graphite annotation format.",
	createAnnotationGraphiteFormat,
	mcp.WithTitleAnnotation("Create Graphite Annotation"),
	mcp.WithIdempotentHintAnnotation(false),
)

// UpdateAnnotationInput represents the payload used to update an existing annotation by ID.
type UpdateAnnotationInput struct {
	ID      int64          `json:"id"       jsonschema:"description=Annotation ID to update"`
	Time    int64          `json:"time,omitempty"    jsonschema:"description=Start time epoch ms"`
	TimeEnd int64          `json:"timeEnd,omitempty" jsonschema:"description=End time epoch ms"`
	Text    string         `json:"text,omitempty"    jsonschema:"description=Annotation text"`
	Tags    []string       `json:"tags,omitempty"    jsonschema:"description=Tags to replace existing tags"`
	Data    map[string]any `json:"data,omitempty" jsonschema:"description=Optional JSON payload"`
}

// updateAnnotation updates an annotation using its ID.
func updateAnnotation(ctx context.Context, args UpdateAnnotationInput) (*annotations.UpdateAnnotationOK, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)
	annotationID := strconv.FormatInt(args.ID, 10)
	req := &models.UpdateAnnotationsCmd{
		Time:    args.Time,
		TimeEnd: args.TimeEnd,
		Text:    args.Text,
		Tags:    args.Tags,
		Data:    args.Data,
	}

	resp, err := c.Annotations.UpdateAnnotation(annotationID, req)
	if err != nil {
		return nil, fmt.Errorf("update annotation: %w", err)
	}

	return resp, nil
}

var UpdateAnnotationTool = mcpgrafana.MustTool(
	"update_annotation",
	"Updates all properties of an annotation that matches the specified ID. Sends a full update (PUT). For partial updates, use patch_annotation instead.",
	updateAnnotation,
	mcp.WithTitleAnnotation("Update Annotation"),
	mcp.WithIdempotentHintAnnotation(false),
)

// PatchAnnotationInput updates only the provided fields.
type PatchAnnotationInput struct {
	ID      int64          `json:"id" jsonschema:"description=Annotation ID"`
	Text    *string        `json:"text,omitempty"     jsonschema:"description=Optional new text"`
	Tags    []string       `json:"tags,omitempty"     jsonschema:"description=Optional replace tags"`
	Time    *int64         `json:"time,omitempty"     jsonschema:"description=Optional new start epoch ms"`
	TimeEnd *int64         `json:"timeEnd,omitempty"  jsonschema:"description=Optional new end epoch ms"`
	Data    map[string]any `json:"data,omitempty"     jsonschema:"description=Optional metadata"`
}

// patchAnnotation patches only the provided annotation fields.
func patchAnnotation(ctx context.Context, args PatchAnnotationInput) (*annotations.PatchAnnotationOK, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)
	id := strconv.FormatInt(args.ID, 10)

	body := &models.PatchAnnotationsCmd{}

	if args.Text != nil {
		body.Text = *args.Text
	}
	if args.Time != nil {
		body.Time = *args.Time
	}
	if args.TimeEnd != nil {
		body.TimeEnd = *args.TimeEnd
	}
	if args.Tags != nil {
		body.Tags = args.Tags
	}
	if args.Data != nil {
		body.Data = args.Data
	}

	resp, err := c.Annotations.PatchAnnotation(id, body)
	if err != nil {
		return nil, fmt.Errorf("patch annotation: %w", err)
	}
	return resp, nil
}

var PatchAnnotationTool = mcpgrafana.MustTool(
	"patch_annotation",
	"Updates only the provided properties of an annotation. Fields omitted are not modified. Use update_annotation for full replacement.",
	patchAnnotation,
	mcp.WithTitleAnnotation("Patch Annotation"),
	mcp.WithIdempotentHintAnnotation(false),
)

// GetAnnotationTagsInput defines filters for retrieving annotation tags.
type GetAnnotationTagsInput struct {
	Tag   *string `json:"tag,omitempty"   jsonschema:"description=Optional filter by tag name"`
	Limit *string `json:"limit,omitempty" jsonschema:"description=Max results\\, default 100"`
}

func getAnnotationTags(ctx context.Context, args GetAnnotationTagsInput) (*annotations.GetAnnotationTagsOK, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)

	req := annotations.GetAnnotationTagsParams{
		Tag:     args.Tag,
		Limit:   args.Limit,
		Context: ctx,
	}

	resp, err := c.Annotations.GetAnnotationTags(&req)
	if err != nil {
		return nil, fmt.Errorf("get annotation tags: %w", err)
	}

	return resp, nil
}

var GetAnnotationTagsTool = mcpgrafana.MustTool(
	"get_annotation_tags",
	"Returns annotation tags with optional filtering by tag name. Only the provided filters are applied.",
	getAnnotationTags,
	mcp.WithTitleAnnotation("Get Annotation Tags"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

func AddAnnotationTools(mcp *server.MCPServer, enableWriteTools bool) {
	GetAnnotationsTool.Register(mcp)
	if enableWriteTools {
		CreateAnnotationTool.Register(mcp)
		CreateGraphiteAnnotationTool.Register(mcp)
		UpdateAnnotationTool.Register(mcp)
		PatchAnnotationTool.Register(mcp)
	}
	GetAnnotationTagsTool.Register(mcp)
}
