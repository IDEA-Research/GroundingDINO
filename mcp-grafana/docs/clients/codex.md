# Codex CLI

This guide helps you set up the `mcp-grafana` server for the OpenAI Codex CLI.

## Prerequisites

- Codex CLI installed (`npm install -g @openai/codex`)
- Grafana 9.0+ with a service account token
- `mcp-grafana` binary in your PATH

## Important: TOML format

Codex uses **TOML** configuration, not JSON. Configuration file: `~/.codex/config.toml`

## Configuration

### CLI setup (recommended)

```bash
codex mcp add grafana -- mcp-grafana
```

Add environment variables:

```bash
codex mcp add grafana \
  --env GRAFANA_URL=http://localhost:3000 \
  --env GRAFANA_SERVICE_ACCOUNT_TOKEN=<your-token> \
  -- mcp-grafana
```

### Manual configuration

Create or edit `~/.codex/config.toml`:

```toml
[mcp_servers.grafana]
command = "mcp-grafana"
args = []
env = { GRAFANA_URL = "http://localhost:3000", GRAFANA_SERVICE_ACCOUNT_TOKEN = "<your-token>" }
```

**Note:** Use `mcp_servers` (underscore, not hyphen).

## Debug mode

```toml
[mcp_servers.grafana]
command = "mcp-grafana"
args = ["-debug"]
env = { GRAFANA_URL = "http://localhost:3000", GRAFANA_SERVICE_ACCOUNT_TOKEN = "<your-token>" }
```

## Docker setup

```toml
[mcp_servers.grafana]
command = "docker"
args = ["run", "--rm", "-i", "-e", "GRAFANA_URL", "-e", "GRAFANA_SERVICE_ACCOUNT_TOKEN", "mcp/grafana"]
env = { GRAFANA_URL = "http://host.docker.internal:3000", GRAFANA_SERVICE_ACCOUNT_TOKEN = "<your-token>" }
```

## Verify configuration

```bash
# List configured servers
codex mcp list

# Show specific server config
codex mcp get grafana

# Start Codex and test
codex
```

Then ask: "List my Grafana dashboards"

## Timeout settings

If Grafana operations take time, increase timeout:

```toml
[mcp_servers.grafana]
command = "mcp-grafana"
args = []
env = { GRAFANA_URL = "http://localhost:3000", GRAFANA_SERVICE_ACCOUNT_TOKEN = "<your-token>" }
startup_timeout_ms = 20000
tool_timeout_ms = 120000
```

## Troubleshooting

**Server not found in Codex:**

- Verify TOML syntax (no trailing commas, use `=` not `:`)
- Check key is `mcp_servers` not `mcp-servers`
- Restart Codex after configuration changes

**Config shared across CLI and IDE:**
Codex CLI and VS Code extension share `~/.codex/config.toml`. A syntax error breaks both.

**Common TOML mistakes:**

```toml
# Wrong - JSON-style
env = {"GRAFANA_URL": "http://localhost:3000"}

# Correct - TOML-style
env = { GRAFANA_URL = "http://localhost:3000" }
```

## Read-only mode

```toml
[mcp_servers.grafana]
command = "mcp-grafana"
args = ["--disable-write"]
env = { GRAFANA_URL = "http://localhost:3000", GRAFANA_SERVICE_ACCOUNT_TOKEN = "<your-token>" }
```
