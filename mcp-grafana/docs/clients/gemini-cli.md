# Gemini CLI

This guide helps you set up the `mcp-grafana` server for the Google Gemini CLI.

## Prerequisites

- Gemini CLI installed (`npm install -g @google/gemini-cli`)
- Grafana 9.0+ with a service account token
- `mcp-grafana` binary in your PATH

## Configuration

Gemini CLI stores MCP configuration in `~/.gemini/settings.json`.

### Manual configuration

Create or edit `~/.gemini/settings.json`:

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

### CLI commands

```bash
# List configured servers
gemini mcp list

# Remove a server
gemini mcp remove grafana
```

## Docker configuration

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

1. Start Gemini CLI:
   ```bash
   gemini
   ```
2. Run `/mcp` to see available tools
3. Ask: "List my Grafana dashboards"

## SSE transport (remote server)

For HTTP-based connection:

1. Start `mcp-grafana` as an HTTP server:

   ```bash
   export GRAFANA_URL="http://localhost:3000"
   export GRAFANA_SERVICE_ACCOUNT_TOKEN="<your-token>"
   mcp-grafana --transport sse --address localhost:8000
   ```

2. Configure in `settings.json`:
   ```json
   {
     "mcpServers": {
       "grafana": {
         "httpUrl": "http://localhost:8000/sse"
       }
     }
   }
   ```

## Troubleshooting

**Tools not appearing:**

- Run `/mcp` in Gemini CLI to check registered tools
- Verify `settings.json` syntax
- Check binary path: `which mcp-grafana`

**Connection errors:**

- Verify `GRAFANA_URL` is reachable
- Check token permissions in Grafana

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
