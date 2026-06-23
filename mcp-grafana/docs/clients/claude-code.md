# Claude Code

This guide helps you set up the `mcp-grafana` server for the Claude Code CLI.

## Prerequisites

- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
- Grafana 9.0+ with a service account token
- `mcp-grafana` binary in your PATH

## One-command setup

```bash
claude mcp add-json "grafana" '{"command":"mcp-grafana","args":[],"env":{"GRAFANA_URL":"http://localhost:3000","GRAFANA_SERVICE_ACCOUNT_TOKEN":"<your-token>"}}'
```

## Manual configuration

Claude Code stores MCP configuration alongside other settings. Use the CLI to manage servers.

```bash
# List configured servers
claude mcp list

# Add a server
claude mcp add grafana -- mcp-grafana

# Remove a server
claude mcp remove grafana
```

## Scope options

Claude Code supports three scopes for MCP servers:

| Scope             | Description                              |
| :---------------- | :--------------------------------------- |
| `local` (default) | Available only to you in current project |
| `project`         | Shared with team via `.mcp.json` file    |
| `user`            | Available to you across all projects     |

```bash
# Add for all your projects
claude mcp add grafana --scope user -- mcp-grafana

# Add for current project only (default)
claude mcp add grafana --scope local -- mcp-grafana
```

## Full configuration with environment variables

```bash
claude mcp add-json "grafana" '{
  "command": "mcp-grafana",
  "args": [],
  "env": {
    "GRAFANA_URL": "http://localhost:3000",
    "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
  }
}'
```

## Docker setup

```bash
claude mcp add-json "grafana" '{
  "command": "docker",
  "args": ["run", "--rm", "-i", "-e", "GRAFANA_URL", "-e", "GRAFANA_SERVICE_ACCOUNT_TOKEN", "mcp/grafana"],
  "env": {
    "GRAFANA_URL": "http://host.docker.internal:3000",
    "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
  }
}'
```

## Debug mode

```bash
claude mcp add-json "grafana" '{
  "command": "mcp-grafana",
  "args": ["-debug"],
  "env": {
    "GRAFANA_URL": "http://localhost:3000",
    "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
  }
}'
```

Then run Claude Code with debug output:

```bash
claude --debug
```

## Verify configuration

1.  Start a new Claude Code session:

    ```bash
    claude
    ```

2.  Ask: "List my Grafana dashboards"
3.  Claude should use the Grafana MCP tools automatically

## View current configuration

```bash
claude mcp list --json
```

## Troubleshooting

**Server not found:**

- Verify binary path: `which mcp-grafana`
- Use full path in configuration if needed

**Permission errors:**

- Check Grafana service account token
- Verify token has required RBAC permissions

## Read-only mode

```bash
claude mcp add-json "grafana" '{
  "command": "mcp-grafana",
  "args": ["--disable-write"],
  "env": {
    "GRAFANA_URL": "http://localhost:3000",
    "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
  }
}'
```
