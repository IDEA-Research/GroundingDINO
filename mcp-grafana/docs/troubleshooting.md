# Troubleshooting

Common issues and solutions for mcp-grafana.

## Connection Errors

### `spawn mcp-grafana ENOENT`

Client can't find the binary.

**Fix:** Add to PATH or use full path:
```json
{ "command": "/full/path/to/mcp-grafana" }
```

Install via Go:
```bash
GOBIN="$HOME/go/bin" go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest
export PATH="$HOME/go/bin:$PATH"
```

### `connection refused`

Grafana not running or wrong URL.

**Check:**
1. `curl http://localhost:3000/api/health`
2. Verify `GRAFANA_URL` is correct
3. Docker: use `host.docker.internal` instead of `localhost`

### `401 Unauthorized`

Invalid token.

**Fix:**
1. Grafana → Administration → Service accounts
2. Create account with Editor role
3. Generate token
4. Set `GRAFANA_SERVICE_ACCOUNT_TOKEN`

### `403 Forbidden`

Token lacks permissions.

**Fix:** Assign correct RBAC permissions (see table below) or assign `Editor` role for quick setup.

## RBAC Permissions

| Tool Category | Required Permission | Required Scope |
|---------------|---------------------|----------------|
| Dashboard read | `dashboards:read` | `dashboards:*` |
| Dashboard write | `dashboards:write` | `dashboards:*` |
| Datasource list | `datasources:read` | `datasources:*` |
| Prometheus query | `datasources:query` | `datasources:*` |
| Alerting | `alert.rules:read` | `folders:*` |
| Incidents | Viewer role | N/A |

## Transport Issues

### SSE/HTTP not working

Start as HTTP server:
```bash
mcp-grafana -t sse --address localhost:8000
```

Or streamable-http for multi-client:
```bash
mcp-grafana -t streamable-http --address localhost:8000
```

Point client to `http://localhost:8000/sse` or `http://localhost:8000/mcp`.

### `transport not recognized`

Old version. Update:
```bash
go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest
```

## TLS Errors

### `x509: certificate signed by unknown authority`

Use TLS flags:
```json
{
  "args": ["--tls-ca-file", "/path/to/ca.crt"]
}
```

Mutual TLS:
```json
{
  "args": [
    "--tls-cert-file", "/path/to/client.crt",
    "--tls-key-file", "/path/to/client.key",
    "--tls-ca-file", "/path/to/ca.crt"
  ]
}
```

## Docker Issues

### Container can't reach host Grafana

Use `host.docker.internal`:
```json
{
  "env": {
    "GRAFANA_URL": "http://host.docker.internal:3000"
  }
}
```

On Linux, add `--network=host` to docker args instead.

### Permission denied

```bash
chmod +x $(which mcp-grafana)
```

## Version Compatibility

### `get datasource by uid: getDataSourceByUidBadRequest`

Grafana version too old.

**Fix:** Upgrade to Grafana 9.0 or later.

### Check versions

```bash
# mcp-grafana version
mcp-grafana --version

# Grafana version
curl $GRAFANA_URL/api/health | jq .version
```

## Debug Mode

Enable verbose logging:
```json
{
  "args": ["-debug"]
}
```

Or run manually:
```bash
GRAFANA_URL="http://localhost:3000" \
GRAFANA_SERVICE_ACCOUNT_TOKEN="<token>" \
mcp-grafana -debug
```

## Still stuck?

1. Run with `-debug` flag
2. Check Grafana server logs
3. Verify network connectivity: `curl -v $GRAFANA_URL/api/health`
4. Open issue: https://github.com/grafana/mcp-grafana/issues
