package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	"github.com/grafana/grafana-openapi-client-go/models"
	mcpgrafana "github.com/grafana/mcp-grafana"
)

type ListDatasourcesParams struct {
	Type string `json:"type,omitempty" jsonschema:"description=The type of datasources to search for. For example\\, 'prometheus'\\, 'loki'\\, 'tempo'\\, etc..."`
}

type dataSourceSummary struct {
	ID        int64  `json:"id"`
	UID       string `json:"uid"`
	Name      string `json:"name"`
	Type      string `json:"type"`
	IsDefault bool   `json:"isDefault"`
}

func listDatasources(ctx context.Context, args ListDatasourcesParams) ([]dataSourceSummary, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)
	resp, err := c.Datasources.GetDataSources()
	if err != nil {
		return nil, fmt.Errorf("list datasources: %w", err)
	}
	datasources := filterDatasources(resp.Payload, args.Type)
	return summarizeDatasources(datasources), nil
}

// filterDatasources returns only datasources of the specified type `t`. If `t`
// is an empty string no filtering is done.
func filterDatasources(datasources models.DataSourceList, t string) models.DataSourceList {
	if t == "" {
		return datasources
	}
	filtered := models.DataSourceList{}
	t = strings.ToLower(t)
	for _, ds := range datasources {
		if strings.Contains(strings.ToLower(ds.Type), t) {
			filtered = append(filtered, ds)
		}
	}
	return filtered
}

func summarizeDatasources(dataSources models.DataSourceList) []dataSourceSummary {
	result := make([]dataSourceSummary, 0, len(dataSources))
	for _, ds := range dataSources {
		result = append(result, dataSourceSummary{
			ID:        ds.ID,
			UID:       ds.UID,
			Name:      ds.Name,
			Type:      ds.Type,
			IsDefault: ds.IsDefault,
		})
	}
	return result
}

var ListDatasources = mcpgrafana.MustTool(
	"list_datasources",
	"List available Grafana datasources. Optionally filter by datasource type (e.g., 'prometheus', 'loki'). Returns a summary list including ID, UID, name, type, and default status.",
	listDatasources,
	mcp.WithTitleAnnotation("List datasources"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type GetDatasourceByUIDParams struct {
	UID string `json:"uid" jsonschema:"required,description=The uid of the datasource"`
}

func getDatasourceByUID(ctx context.Context, args GetDatasourceByUIDParams) (*models.DataSource, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)
	datasource, err := c.Datasources.GetDataSourceByUID(args.UID)
	if err != nil {
		// Check if it's a 404 Not Found Error
		if strings.Contains(err.Error(), "404") {
			return nil, fmt.Errorf("datasource with UID '%s' not found. Please check if the datasource exists and is accessible", args.UID)
		}
		return nil, fmt.Errorf("get datasource by uid %s: %w", args.UID, err)
	}
	return datasource.Payload, nil
}

var GetDatasourceByUID = mcpgrafana.MustTool(
	"get_datasource_by_uid",
	"Retrieves detailed information about a specific datasource using its UID. Returns the full datasource model, including name, type, URL, access settings, JSON data, and secure JSON field status.",
	getDatasourceByUID,
	mcp.WithTitleAnnotation("Get datasource by UID"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type GetDatasourceByNameParams struct {
	Name string `json:"name" jsonschema:"required,description=The name of the datasource"`
}

func getDatasourceByName(ctx context.Context, args GetDatasourceByNameParams) (*models.DataSource, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)
	datasource, err := c.Datasources.GetDataSourceByName(args.Name)
	if err != nil {
		return nil, fmt.Errorf("get datasource by name %s: %w", args.Name, err)
	}
	return datasource.Payload, nil
}

var GetDatasourceByName = mcpgrafana.MustTool(
	"get_datasource_by_name",
	"Retrieves detailed information about a specific datasource using its name. Returns the full datasource model, including UID, type, URL, access settings, JSON data, and secure JSON field status.",
	getDatasourceByName,
	mcp.WithTitleAnnotation("Get datasource by name"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

func AddDatasourceTools(mcp *server.MCPServer) {
	ListDatasources.Register(mcp)
	GetDatasourceByUID.Register(mcp)
	GetDatasourceByName.Register(mcp)
}
