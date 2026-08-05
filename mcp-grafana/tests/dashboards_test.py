import json
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

@pytest.mark.parametrize("model", models)
@pytest.mark.flaky(max_runs=3)
async def test_dashboard_panel_queries_tool(model: str, mcp_client: ClientSession):
    tools = await get_converted_tools(mcp_client)
    prompt = "Can you list the panel queries for the dashboard with UID fe9gm6guyzi0wd?"

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=prompt),
    ]

    # 1. Call the dashboard panel queries tool
    messages = await llm_tool_call_sequence(
        model, messages, tools, mcp_client, "get_dashboard_panel_queries",
        {"uid": "fe9gm6guyzi0wd"}
    )

    # 2. Final LLM response
    response = await acompletion(model=model, messages=messages, tools=tools)
    content = response.choices[0].message.content
    panel_queries_checker = CustomLLMBooleanEvaluator(
        settings=CustomLLMBooleanSettings(
            prompt="Does the response contain specific information about the panel queries and titles for a grafana dashboard?",
        )
    )
    print("content", content)
    expect(input=prompt, output=content).to_pass(panel_queries_checker)


@pytest.mark.parametrize("model", models)
@pytest.mark.flaky(max_runs=3)
async def test_dashboard_update_with_patch_operations(model: str, mcp_client: ClientSession):
    """Test that LLMs naturally use patch operations for dashboard updates"""
    tools = await get_converted_tools(mcp_client)

    # First, create a non-provisioned test dashboard by copying the demo dashboard
    # 1. Get the demo dashboard JSON
    demo_result = await mcp_client.call_tool("get_dashboard_by_uid", {"uid": "fe9gm6guyzi0wd"})
    demo_data = json.loads(demo_result.content[0].text)
    dashboard_json = demo_data["dashboard"]

    # 2. Remove uid and id to create a new dashboard
    if "uid" in dashboard_json:
        del dashboard_json["uid"]
    if "id" in dashboard_json:
        del dashboard_json["id"]

    # 3. Set a new title
    title = f"Test Dashboard"
    dashboard_json["title"] = title
    dashboard_json["tags"] = ["python-integration-test"]

    # 4. Create the dashboard in Grafana
    create_result = await mcp_client.call_tool("update_dashboard", {
        "dashboard": dashboard_json,
        "folderUid": "",
        "overwrite": False
    })
    create_data = json.loads(create_result.content[0].text)
    created_dashboard_uid = create_data["uid"]

    # 5. Update the dashboard title
    updated_title = f"Updated {title}"
    title_prompt = f"Update the title of the Test Dashboard to {updated_title}. Search for the dashboard by title first."

    messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content=title_prompt),
    ]

    # 6. Search for the test dashboard
    messages = await llm_tool_call_sequence(
        model, messages, tools, mcp_client, "search_dashboards",
        {"query": title}
    )

    # 7. Update the dashboard using patch operations
    messages = await llm_tool_call_sequence(
        model, messages, tools, mcp_client, "update_dashboard",
        {
            "uid": created_dashboard_uid,
            "operations": [
                {
                    "op": "replace",
                    "path": "$.title",
                    "value": updated_title
                }
            ]
        }
    )

    # 8. Final LLM response - just verify it completes successfully
    response = await acompletion(model=model, messages=messages, tools=tools)
    content = response.choices[0].message.content

    # Test passes if we get here - the tool call sequence worked correctly
    assert len(content) > 0, "LLM should provide a response after updating the dashboard"

