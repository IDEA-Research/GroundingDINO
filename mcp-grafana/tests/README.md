# Tests

This directory contains an e2e test suite for the Grafana MCP server.

The test suite evaluates the LLM's ability to use Grafana MCP tools effectively:

- **Loki tests**: Evaluates how well the LLM can use Grafana tools to:
  - Navigate and use available tools
  - Make appropriate tool calls
  - Process and present the results in a meaningful way
  - Evaluating the LLM responses using `langevals` package, using custom LLM-as-a-Judge approach.

The tests are run against two LLM models:
- GPT-4
- Claude 3.5 Sonnet

Tests are using [`uv`] to manage dependencies. Install uv following the instructions for your platform.

## Prerequisites
- Docker installed and running on your system
- Docker containers for the test environment must be started before running tests

## Setup
1. Create a virtual environment and install the dependencies:
   ```bash
   uv sync --all-groups
   ```

2. Create a `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. Start the required Docker containers

4. Start the MCP server in SSE mode; from the root of the project:
   ```bash
   go run ./cmd/mcp-grafana -t sse
   ```

5. Run the tests:
   ```bash
   uv run pytest
   ```

[`uv`]: https://docs.astral.sh/uv/
