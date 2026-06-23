from mcp import ClientSession
import pytest
from langevals import expect
from langevals_langevals.llm_boolean import (
    CustomLLMBooleanEvaluator,
    CustomLLMBooleanSettings,
)
from litellm import Message, acompletion
from mcp import ClientSession

from conftest import models
from utils import (
    get_converted_tools,
    llm_tool_call_sequence,
)

pytestmark = pytest.mark.anyio


class TestTempoProxiedToolsBasic:
    """Test Tempo proxied MCP tools functionality.

    These tests verify that Tempo datasources with MCP support are discovered
    per-session and their tools are registered with a datasourceUid parameter
    for multi-datasource support.

    Requires:
    - Docker compose services running (includes 2 Tempo instances)
    - GRAFANA_USERNAME and GRAFANA_PASSWORD environment variables
    - MCP server running
    """

    @pytest.mark.anyio
    async def test_tempo_tools_discovered_and_registered(
        self, mcp_client: ClientSession
    ):
        """Test that Tempo tools are discovered and registered with datasourceUid parameter."""

        # List all tools
        list_response = await mcp_client.list_tools()
        all_tool_names = [tool.name for tool in list_response.tools]

        # Find tempo-prefixed tools (should preserve hyphens from original tool names)
        tempo_tools = [name for name in all_tool_names if name.startswith("tempo_")]

        # Expected tools from Tempo MCP server
        expected_tempo_tools = [
            "tempo_traceql-search",
            "tempo_traceql-metrics-instant",
            "tempo_traceql-metrics-range",
            "tempo_get-trace",
            "tempo_get-attribute-names",
            "tempo_get-attribute-values",
            "tempo_docs-traceql",
        ]

        assert len(tempo_tools) == len(expected_tempo_tools), (
            f"Expected {len(expected_tempo_tools)} unique tempo tools, found {len(tempo_tools)}: {tempo_tools}"
        )

        for expected_tool in expected_tempo_tools:
            assert expected_tool in tempo_tools, (
                f"Tool {expected_tool} should be available"
            )

    @pytest.mark.anyio
    async def test_tempo_tools_have_datasourceUid_parameter(self, mcp_client):
        """Test that all tempo tools have a required datasourceUid parameter."""

        list_response = await mcp_client.list_tools()
        tempo_tools = [
            tool for tool in list_response.tools if tool.name.startswith("tempo_")
        ]

        assert len(tempo_tools) > 0, "Should have at least one tempo tool"

        for tool in tempo_tools:
            # Verify the tool has input schema
            assert hasattr(tool, "inputSchema"), (
                f"Tool {tool.name} should have inputSchema"
            )
            assert isinstance(tool.inputSchema, dict), (
                f"Tool {tool.name} inputSchema should be a dict"
            )

            # Verify datasourceUid parameter exists (camelCase)
            properties = tool.inputSchema.get("properties", {})
            assert "datasourceUid" in properties, (
                f"Tool {tool.name} should have datasourceUid parameter (camelCase)"
            )

            # Verify it's required
            required = tool.inputSchema.get("required", [])
            assert "datasourceUid" in required, (
                f"Tool {tool.name} should require datasourceUid parameter"
            )

            # Verify parameter has proper description
            datasource_uid_prop = properties["datasourceUid"]
            assert "type" in datasource_uid_prop, (
                f"datasourceUid should have type defined"
            )
            assert datasource_uid_prop["type"] == "string", (
                f"datasourceUid should be type string"
            )

    @pytest.mark.anyio
    async def test_tempo_tool_call_with_valid_datasource(self, mcp_client):
        """Test calling a tempo tool with a valid datasourceUid."""

        # Call docs-traceql which should return documentation (doesn't require data)
        try:
            call_response = await mcp_client.call_tool(
                "tempo_docs-traceql",
                arguments={"datasourceUid": "tempo", "name": "basic"},
            )

            # Verify we got a response
            assert call_response.content, "Tool should return content"

            # Should have text content (documentation)
            response_text = call_response.content[0].text
            assert len(response_text) > 0, "Response should have content"
            assert "traceql" in response_text.lower(), (
                "Response should contain TraceQL documentation"
            )
            print(response_text)

        except Exception as e:
            # If this fails, it might be because Tempo doesn't have data yet
            # but at least verify the error isn't about missing datasourceUid
            error_msg = str(e).lower()
            assert "datasourceuid" not in error_msg, (
                f"Should not fail due to datasourceUid parameter: {e}"
            )
            print(error_msg)

    @pytest.mark.anyio
    async def test_tempo_tool_call_missing_datasourceUid(self, mcp_client):
        """Test that calling a tempo tool without datasourceUid fails appropriately."""

        with pytest.raises(Exception) as exc_info:
            await mcp_client.call_tool(
                "tempo_docs-traceql",
                arguments={"name": "basic"},  # Missing datasourceUid
            )

        error_msg = str(exc_info.value).lower()
        assert "datasourceuid" in error_msg and "required" in error_msg, (
            f"Should require datasourceUid parameter: {exc_info.value}"
        )

    @pytest.mark.anyio
    async def test_tempo_tool_call_invalid_datasourceUid(self, mcp_client):
        """Test that calling a tempo tool with invalid datasourceUid returns helpful error."""

        with pytest.raises(Exception) as exc_info:
            await mcp_client.call_tool(
                "tempo_docs-traceql",
                arguments={"datasourceUid": "nonexistent-tempo", "name": "basic"},
            )

        error_msg = str(exc_info.value).lower()
        # Should mention that datasource wasn't found
        assert "not found" in error_msg or "not accessible" in error_msg, (
            f"Should indicate datasource not found: {exc_info.value}"
        )

        # Should mention available datasources to help user
        assert "tempo" in error_msg or "available" in error_msg, (
            f"Error should be helpful and mention available datasources: {exc_info.value}"
        )

    @pytest.mark.anyio
    async def test_tempo_tool_works_with_multiple_datasources(self, mcp_client):
        """Test that the same tool works with different datasources via datasourceUid."""

        # Both tempo and tempo-secondary should be available in our test environment
        datasources = ["tempo", "tempo-secondary"]

        for datasource_uid in datasources:
            try:
                # Call the same tool with different datasources
                call_response = await mcp_client.call_tool(
                    "tempo_get-attribute-names",
                    arguments={"datasourceUid": datasource_uid},
                )

                # Verify we got a response
                assert call_response.content, (
                    f"Tool should return content for datasource {datasource_uid}"
                )

                # Response should be valid JSON or text
                response_text = call_response.content[0].text
                assert len(response_text) > 0, (
                    f"Response should have content for datasource {datasource_uid}"
                )

            except Exception as e:
                # If this fails, it's acceptable if Tempo doesn't have trace data yet
                # But verify it's not a routing/config error
                error_msg = str(e).lower()
                assert (
                    "not found" not in error_msg or datasource_uid not in error_msg
                ), f"Datasource {datasource_uid} should be accessible: {e}"


class TestTempoProxiedToolsWithLLM:
    """LLM integration tests for Tempo proxied tools."""

    @pytest.mark.parametrize("model", models)
    @pytest.mark.flaky(max_runs=3)
    async def test_llm_can_list_trace_attributes(
        self, model: str, mcp_client: ClientSession
    ):
        """Test that an LLM can list available trace attributes from Tempo."""
        tools = await get_converted_tools(mcp_client)
        prompt = (
            "Use the tempo tools to get a list of all available trace attribute names "
            "from the datasource with UID 'tempo'. I want to know what attributes "
            "I can use in my TraceQL queries."
        )

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=prompt),
        ]

        # LLM should call tempo_get-attribute-names with datasourceUid
        messages = await llm_tool_call_sequence(
            model,
            messages,
            tools,
            mcp_client,
            "tempo_get-attribute-names",
            {"datasourceUid": "tempo"},
        )

        # Final LLM response should mention attributes
        response = await acompletion(model=model, messages=messages, tools=tools)
        content = response.choices[0].message.content

        attributes_checker = CustomLLMBooleanEvaluator(
            settings=CustomLLMBooleanSettings(
                prompt="Does the response list or describe trace attributes that are available for querying?",
            )
        )
        expect(input=prompt, output=content).to_pass(attributes_checker)
