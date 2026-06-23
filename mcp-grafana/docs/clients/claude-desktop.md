# Claude Desktop

This guide helps you set up the `mcp-grafana` server for Claude Desktop.

## Prerequisites

- Claude Desktop installed
- Grafana 9.0+ with a service account token
- `mcp-grafana` binary in your PATH

## Installation

### Option 1: Go install

```bash
GOBIN="$HOME/go/bin" go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest
```

### Option 2: Download binary

Get the latest release from [GitHub Releases](https://github.com/grafana/mcp-grafana/releases) and add to your PATH.

### Option 3: Docker

No installation needed - use Docker in the configuration below.

## Configuration

Edit your Claude Desktop configuration file:

| OS      | Path                                                              |
| :------ | :---------------------------------------------------------------- |
| macOS   | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json`                     |
| Linux   | `~/.config/Claude/claude_desktop_config.json`                     |

### Binary configuration

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

If you get `ENOENT`, use the full path:

```json
{
  "mcpServers": {
    "grafana": {
      "command": "/Users/yourname/go/bin/mcp-grafana",
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

Add `-debug` to args for verbose logging:

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

1. Fully quit Claude Desktop (Cmd+Q on macOS)
2. Restart Claude Desktop
3. Open a new conversation
4. Ask: "List my Grafana dashboards"

If it works, you'll see dashboard names. If not, check logs at `~/Library/Logs/Claude/mcp*.log` (macOS).

## Read-only mode

Prevent accidental modifications:

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

## TLS client certificates

For Grafana instances requiring mTLS:

```json
{
  "mcpServers": {
    "grafana": {
      "command": "mcp-grafana",
      "args": [
        "--tls-cert-file",
        "/path/to/client.crt",
        "--tls-key-file",
        "/path/to/client.key",
        "--tls-ca-file",
        "/path/to/ca.crt"
      ],
      "env": {
        "GRAFANA_URL": "https://secure-grafana.example.com",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your-token>"
      }
    }
  }
}
```
