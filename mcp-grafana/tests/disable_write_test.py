import pytest
import os
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

pytestmark = pytest.mark.anyio


@pytest.fixture
def grafana_env():
    env = {"GRAFANA_URL": os.environ.get("GRAFANA_URL", "http://localhost:3000")}
    # Check for the new service account token environment variable first
    if key := os.environ.get("GRAFANA_SERVICE_ACCOUNT_TOKEN"):
        env["GRAFANA_SERVICE_ACCOUNT_TOKEN"] = key
    elif key := os.environ.get("GRAFANA_API_KEY"):
        env["GRAFANA_API_KEY"] = key
    return env


async def test_disable_write_flag_disables_write_tools(grafana_env):
    """Test that --disable-write flag disables write tools."""
    params = StdioServerParameters(
        command=os.environ.get("MCP_GRAFANA_PATH", "../dist/mcp-grafana"),
        args=["--disable-write"],
        env=grafana_env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List all available tools
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            
            # Verify write tools are NOT present
            write_tools = [
                "update_dashboard",
                "create_folder",
                "create_incident",
                "add_activity_to_incident",
                "create_alert_rule",
                "update_alert_rule",
                "delete_alert_rule",
                "create_annotation",
                "create_graphite_annotation",
                "update_annotation",
                "patch_annotation",
                "find_error_pattern_logs",
                "find_slow_requests",
            ]
            
            for tool in write_tools:
                assert tool not in tool_names, f"Write tool '{tool}' should not be available with --disable-write flag"
            
            # Verify read tools ARE still present
            read_tools = [
                "get_dashboard_by_uid",
                "list_alert_rules",
                "get_alert_rule_by_uid",
                "list_contact_points",
                "list_incidents",
                "get_incident",
                "get_sift_investigation",
                "get_annotations",
                "get_annotation_tags",
            ]
            
            for tool in read_tools:
                assert tool in tool_names, f"Read tool '{tool}' should still be available with --disable-write flag"


async def test_without_disable_write_flag_enables_write_tools(grafana_env):
    """Test that without --disable-write flag, write tools are enabled."""
    params = StdioServerParameters(
        command=os.environ.get("MCP_GRAFANA_PATH", "../dist/mcp-grafana"),
        args=[],  # No --disable-write flag
        env=grafana_env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List all available tools
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            
            # Verify write tools ARE present
            write_tools = [
                "update_dashboard",
                "create_folder",
                "create_incident",
                "add_activity_to_incident",
                "create_alert_rule",
                "update_alert_rule",
                "delete_alert_rule",
                "create_annotation",
                "create_graphite_annotation",
                "update_annotation",
                "patch_annotation",
                "find_error_pattern_logs",
                "find_slow_requests",
            ]
            
            for tool in write_tools:
                assert tool in tool_names, f"Write tool '{tool}' should be available without --disable-write flag"
            
            # Verify read tools are also present
            read_tools = [
                "get_dashboard_by_uid",
                "list_alert_rules",
                "get_alert_rule_by_uid",
                "list_contact_points",
                "list_incidents",
                "get_incident",
                "get_sift_investigation",
                "get_annotations",
                "get_annotation_tags",
            ]
            
            for tool in read_tools:
                assert tool in tool_names, f"Read tool '{tool}' should be available without --disable-write flag"

