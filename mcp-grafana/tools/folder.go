package tools

import (
	"context"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	"github.com/grafana/grafana-openapi-client-go/models"
	mcpgrafana "github.com/grafana/mcp-grafana"
)

type CreateFolderParams struct {
	Title     string `json:"title" jsonschema:"required,description=The title of the folder."`
	UID       string `json:"uid,omitempty" jsonschema:"description=Optional folder UID. If omitted\\, Grafana will generate one."`
	ParentUID string `json:"parentUid,omitempty" jsonschema:"description=Optional parent folder UID. If set\\, the folder will be created under this parent."`
}

func createFolder(ctx context.Context, args CreateFolderParams) (*models.Folder, error) {
	if args.Title == "" {
		return nil, fmt.Errorf("title is required")
	}

	c := mcpgrafana.GrafanaClientFromContext(ctx)
	cmd := &models.CreateFolderCommand{Title: args.Title}
	if args.UID != "" {
		cmd.UID = args.UID
	}
	if args.ParentUID != "" {
		cmd.ParentUID = args.ParentUID
	}

	resp, err := c.Folders.CreateFolder(cmd)
	if err != nil {
		return nil, fmt.Errorf("create folder '%s': %w", args.Title, err)
	}
	return resp.Payload, nil
}

var CreateFolder = mcpgrafana.MustTool(
	"create_folder",
	"Create a Grafana folder. Provide a title and optional UID. Returns the created folder.",
	createFolder,
	mcp.WithTitleAnnotation("Create folder"),
	mcp.WithIdempotentHintAnnotation(false),
	mcp.WithReadOnlyHintAnnotation(false),
)

func AddFolderTools(mcp *server.MCPServer, enableWriteTools bool) {
	if enableWriteTools {
		CreateFolder.Register(mcp)
	}
}
