#!/bin/bash

# Configuration
# Default to localhost if not set
export GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"

# You MUST set this token. 
# Create a Service Account in Grafana -> Administration -> Service Accounts, 
# give it 'Editor' role, add a token, and paste it here.
if [ -z "$GRAFANA_SERVICE_ACCOUNT_TOKEN" ]; then
    echo "Please set GRAFANA_SERVICE_ACCOUNT_TOKEN in this script or environment."
    echo "export GRAFANA_SERVICE_ACCOUNT_TOKEN='your_token_here'"
    exit 1
fi

# Run the server
# Use -t stdio for MCP clients
./mcp-grafana/dist/mcp-grafana -t stdio "$@"
