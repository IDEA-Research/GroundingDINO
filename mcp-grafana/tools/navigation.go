package tools

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	mcpgrafana "github.com/grafana/mcp-grafana"
)

type GenerateDeeplinkParams struct {
	ResourceType  string            `json:"resourceType" jsonschema:"required,description=Type of resource: dashboard\\, panel\\, or explore"`
	DashboardUID  *string           `json:"dashboardUid,omitempty" jsonschema:"description=Dashboard UID (required for dashboard and panel types)"`
	DatasourceUID *string           `json:"datasourceUid,omitempty" jsonschema:"description=Datasource UID (required for explore type)"`
	PanelID       *int              `json:"panelId,omitempty" jsonschema:"description=Panel ID (required for panel type)"`
	QueryParams   map[string]string `json:"queryParams,omitempty" jsonschema:"description=Additional query parameters"`
	TimeRange     *TimeRange        `json:"timeRange,omitempty" jsonschema:"description=Time range for the link"`
}

type TimeRange struct {
	From string `json:"from" jsonschema:"description=Start time (e.g.\\, 'now-1h')"`
	To   string `json:"to" jsonschema:"description=End time (e.g.\\, 'now')"`
}

func generateDeeplink(ctx context.Context, args GenerateDeeplinkParams) (string, error) {
	config := mcpgrafana.GrafanaConfigFromContext(ctx)
	baseURL := strings.TrimRight(config.URL, "/")

	if baseURL == "" {
		return "", fmt.Errorf("grafana url not configured. Please set GRAFANA_URL environment variable or X-Grafana-URL header")
	}

	var deeplink string

	switch strings.ToLower(args.ResourceType) {
	case "dashboard":
		if args.DashboardUID == nil {
			return "", fmt.Errorf("dashboardUid is required for dashboard links")
		}
		deeplink = fmt.Sprintf("%s/d/%s", baseURL, *args.DashboardUID)
	case "panel":
		if args.DashboardUID == nil {
			return "", fmt.Errorf("dashboardUid is required for panel links")
		}
		if args.PanelID == nil {
			return "", fmt.Errorf("panelId is required for panel links")
		}
		deeplink = fmt.Sprintf("%s/d/%s?viewPanel=%d", baseURL, *args.DashboardUID, *args.PanelID)
	case "explore":
		if args.DatasourceUID == nil {
			return "", fmt.Errorf("datasourceUid is required for explore links")
		}
		params := url.Values{}
		exploreState := fmt.Sprintf(`{"datasource":"%s"}`, *args.DatasourceUID)
		params.Set("left", exploreState)
		deeplink = fmt.Sprintf("%s/explore?%s", baseURL, params.Encode())
	default:
		return "", fmt.Errorf("unsupported resource type: %s. Supported types are: dashboard, panel, explore", args.ResourceType)
	}

	if args.TimeRange != nil {
		separator := "?"
		if strings.Contains(deeplink, "?") {
			separator = "&"
		}
		timeParams := url.Values{}
		if args.TimeRange.From != "" {
			timeParams.Set("from", args.TimeRange.From)
		}
		if args.TimeRange.To != "" {
			timeParams.Set("to", args.TimeRange.To)
		}
		if len(timeParams) > 0 {
			deeplink = fmt.Sprintf("%s%s%s", deeplink, separator, timeParams.Encode())
		}
	}

	if len(args.QueryParams) > 0 {
		separator := "?"
		if strings.Contains(deeplink, "?") {
			separator = "&"
		}
		additionalParams := url.Values{}
		for key, value := range args.QueryParams {
			additionalParams.Set(key, value)
		}
		deeplink = fmt.Sprintf("%s%s%s", deeplink, separator, additionalParams.Encode())
	}

	return deeplink, nil
}

var GenerateDeeplink = mcpgrafana.MustTool(
	"generate_deeplink",
	"Generate deeplink URLs for Grafana resources. Supports dashboards (requires dashboardUid), panels (requires dashboardUid and panelId), and Explore queries (requires datasourceUid). Optionally accepts time range and additional query parameters.",
	generateDeeplink,
	mcp.WithTitleAnnotation("Generate navigation deeplink"),
	mcp.WithIdempotentHintAnnotation(true),
	mcp.WithReadOnlyHintAnnotation(true),
)

func AddNavigationTools(mcp *server.MCPServer) {
	GenerateDeeplink.Register(mcp)
}
