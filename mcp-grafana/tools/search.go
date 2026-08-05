package tools

import (
	"context"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	"github.com/grafana/grafana-openapi-client-go/client/search"
	"github.com/grafana/grafana-openapi-client-go/models"
	mcpgrafana "github.com/grafana/mcp-grafana"
)

var dashboardTypeStr = "dash-db"
var folderTypeStr = "dash-folder"

type SearchDashboardsParams struct {
	Query string `json:"query" jsonschema:"description=The query to search for"`
}

func searchDashboards(ctx context.Context, args SearchDashboardsParams) (models.HitList, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)
	params := search.NewSearchParamsWithContext(ctx)
	if args.Query != "" {
		params.SetQuery(&args.Query)
		params.SetType(&dashboardTypeStr)
	}
	search, err := c.Search.Search(params)
	if err != nil {
		return nil, fmt.Errorf("search dashboards for %+v: %w", c, err)
	}
	return search.Payload, nil
}

var SearchDashboards = mcpgrafana.MustTool(
	"search_dashboards",
	"Search for Grafana dashboards by a query string. Returns a list of matching dashboards with details like title, UID, folder, tags, and URL.",
	searchDashboards,
	mcp.WithTitleAnnotation("Search dashboards"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

type SearchFoldersParams struct {
	Query string `json:"query" jsonschema:"description=The query to search for"`
}

func searchFolders(ctx context.Context, args SearchFoldersParams) (models.HitList, error) {
	c := mcpgrafana.GrafanaClientFromContext(ctx)
	params := search.NewSearchParamsWithContext(ctx)
	if args.Query != "" {
		params.SetQuery(&args.Query)
	}
	params.SetType(&folderTypeStr)
	search, err := c.Search.Search(params)
	if err != nil {
		return nil, fmt.Errorf("search folders for %+v: %w", c, err)
	}
	return search.Payload, nil
}

var SearchFolders = mcpgrafana.MustTool(
	"search_folders",
	"Search for Grafana folders by a query string. Returns matching folders with details like title, UID, and URL.",
	searchFolders,
	mcp.WithTitleAnnotation("Search folders"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

func AddSearchTools(mcp *server.MCPServer) {
	SearchDashboards.Register(mcp)
	SearchFolders.Register(mcp)
}
