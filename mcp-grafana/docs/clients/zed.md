# Zed

This guide helps you set up the `mcp-grafana` server for the Zed editor.

## Prerequisites

- Zed editor installed
- Grafana 9.0+ with a service account token
- `mcp-grafana` binary in your PATH

## Configuration

Zed uses `context_servers` in `settings.json`, not `mcpServers`.

### Add using the UI

1. Open Agent Panel (Cmd+Shift+A)
2. Click Settings (gear icon)
3. Click "Add Custom Server"
4. Fill in command and args

### Manual configuration

Open Zed settings (Cmd+,) and add:

```json
{
  "context_servers": {
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

**Note:** Zed uses `context_servers`, not `mcpServers`.

## Docker configuration

```json
{
  "context_servers": {
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
  "context_servers": {
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

1. Open Agent Panel settings
2. Check indicator next to "grafana"
   - Green = server active
   - Other colors = check tooltip for status
3. Open Agent Panel chat
4. Ask: "List my Grafana dashboards"

**Tip:** Mention "grafana" in your prompt to help the model pick the right tools.

## Tool permissions

By default, Zed asks permission for each tool call. To auto-allow:

```json
{
  "agent": {
    "always_allow_tool_actions": true
  }
}
```

Use with caution - this enables all MCP tools without confirmation.

## Troubleshooting

**Server not starting:**

- Check Zed logs: Cmd+Shift+P -> "zed: open logs"
- Verify binary path: `which mcp-grafana`
- Restart Zed after configuration changes

**Tools not appearing:**

- Zed supports both stdio and HTTP transports
- For remote servers, use native URL syntax or `mcp-remote` shim

**Remote server (native URL syntax):**

Zed supports direct URL connections for remote MCP servers:

```json
{
  "context_servers": {
    "grafana": {
      "url": "http://localhost:8000/sse",
      "headers": {}
    }
  }
}
```

First start the server:

```bash
mcp-grafana --transport sse --address localhost:8000
```

**Remote server (mcp-remote fallback):**

Alternative using `mcp-remote` shim:

```json
{
  "context_servers": {
    "grafana": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8000/sse"]
    }
  }
}
```

## Read-only mode

```json
{
  "context_servers": {
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
