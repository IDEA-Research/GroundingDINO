# Windsurf

This guide helps you set up the `mcp-grafana` server for Windsurf.

## Prerequisites

- Windsurf IDE installed
- Grafana 9.0+ with a service account token
- `mcp-grafana` binary in your PATH

## Configuration

Configuration file location:

| OS          | Path                                              |
| :---------- | :------------------------------------------------ |
| macOS/Linux | `~/.codeium/windsurf/mcp_config.json`             |
| Windows     | `%USERPROFILE%\.codeium\windsurf\mcp_config.json` |

### Add using the UI

1. Open Windsurf Settings (Cmd+Shift+P -> "Open Windsurf Settings")
2. Scroll to Cascade section
3. Click "Add Server" or "View raw config"

### Manual configuration

Create or edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "grafana": {
      "command": "mcp-grafana",
      "args": [],
      "env": {
        "GRAFANA_URL": "http://localhost:3000",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
      }
    }
  }
}
```

### Docker configuration

```json
{
  "mcpServers": {
    "grafana": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-e",
        "GRAFANA_URL",
        "-e",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN",
        "mcp/grafana"
      ],
      "env": {
        "GRAFANA_URL": "http://host.docker.internal:3000",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
      }
    }
  }
}
```

## Debug mode

```json
{
  "mcpServers": {
    "grafana": {
      "command": "mcp-grafana",
      "args": ["-debug"],
      "env": {
        "GRAFANA_URL": "http://localhost:3000",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
      }
    }
  }
}
```

## Verify configuration

1. Click the refresh button after adding the server
2. Open Cascade view
3. Click the hammer icon (MCP servers)
4. Grafana should show green status
5. Ask: "List my Grafana dashboards"

## Tool limit

Windsurf limits total MCP tools to 100. If you hit the limit:

1. Go to Windsurf Settings -> Manage plugins
2. Disable unused servers
3. Toggle off individual tools you don't need

## Troubleshooting

**Server not connecting:**

- Press refresh button in Cascade settings
- Check JSON syntax
- Verify binary exists: `which mcp-grafana`

**SSE transport (remote server):**

If you need HTTP-based connection instead of stdio:

```bash
mcp-grafana --transport streamable-http --address localhost:8000
```

Then configure with `serverUrl`:

```json
{
  "mcpServers": {
    "grafana": {
      "serverUrl": "http://localhost:8000/mcp"
    }
  }
}
```

## Read-only mode

```json
{
  "mcpServers": {
    "grafana": {
      "command": "mcp-grafana",
      "args": ["--disable-write"],
      "env": {
        "GRAFANA_URL": "http://localhost:3000",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
      }
    }
  }
}
```
