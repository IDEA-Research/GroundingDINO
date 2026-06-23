from typing import Dict
import pytest
from langevals import expect
from langevals_langevals.llm_boolean import (
    CustomLLMBooleanEvaluator,
    CustomLLMBooleanSettings,
)
from litellm import Message, acompletion
from mcp import ClientSession
import aiohttp
import uuid
import os
from conftest import DEFAULT_GRAFANA_URL

from conftest import models
from utils import (
    get_converted_tools,
    llm_tool_call_sequence,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
async def grafana_team():
    """Create a temporary test team and clean it up after the test is done."""
    # Generate a unique team name to avoid conflicts
    team_name = f"test-team-{uuid.uuid4().hex[:8]}"

    # Get Grafana URL and service account token from environment
    grafana_url = os.environ.get("GRAFANA_URL", DEFAULT_GRAFANA_URL)

    auth_header = None
    # Check for the new service account token environment variable first
    if api_key := os.environ.get("GRAFANA_SERVICE_ACCOUNT_TOKEN"):
        auth_header = {"Authorization": f"Bearer {api_key}"}
    elif api_key := os.environ.get("GRAFANA_API_KEY"):
        auth_header = {"Authorization": f"Bearer {api_key}"}
        import warnings

        warnings.warn(
            "GRAFANA_API_KEY is deprecated, please use GRAFANA_SERVICE_ACCOUNT_TOKEN instead. See https://grafana.com/docs/grafana/latest/administration/service-accounts/#add-a-token-to-a-service-account-in-grafana for details on creating service account tokens.",
            DeprecationWarning,
        )

    if not auth_header:
        pytest.skip("No authentication credentials available to create team")

    # Create the team using Grafana API
    team_id = None
    async with aiohttp.ClientSession() as session:
        create_url = f"{grafana_url}/api/teams"
        async with session.post(
            create_url,
            headers=auth_header,
            json={"name": team_name, "email": f"{team_name}@example.com"},
        ) as response:
            if response.status != 200:
                resp_text = await response.text()
                pytest.skip(f"Failed to create team: {resp_text}")
            resp_data = await response.json()
            team_id = resp_data.get("teamId")

    # Yield the team info for the test to use
    yield {"id": team_id, "name": team_name}

    # Clean up after the test
    if team_id:
        async with aiohttp.ClientSession() as session:
            delete_url = f"{grafana_url}/api/teams/{team_id}"
            async with session.delete(delete_url, headers=auth_header) as response:
                if response.status != 200:
                    resp_text = await response.text()
                    print(f"Warning: Failed to delete team: {resp_text}")


@pytest.mark.parametrize("model", models)
@pytest.mark.flaky(max_runs=3)
async def test_list_teams_tool(
    model: str, mcp_client: ClientSession, grafana_team: Dict[str, str]
):
    tools = await get_converted_tools(mcp_client)
    team_name = grafana_team["name"]
    prompt = "Can you list the teams in Grafana?"

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=prompt),
    ]

    # 1. Call the list teams tool
    messages = await llm_tool_call_sequence(
        model,
        messages,
        tools,
        mcp_client,
        "list_teams",
    )

    # 2. Final LLM response
    response = await acompletion(model=model, messages=messages, tools=tools)
    content = response.choices[0].message.content
    panel_queries_checker = CustomLLMBooleanEvaluator(
        settings=CustomLLMBooleanSettings(
            prompt=(
                "Does the response contain specific information about "
                "the teams in Grafana?"
                f"There should be a team named {team_name}. "
            ),
        )
    )
    expect(input=prompt, output=content).to_pass(panel_queries_checker)


@pytest.mark.parametrize("model", models)
@pytest.mark.flaky(max_runs=3)
async def test_list_users_by_org_tool(model: str, mcp_client: ClientSession):
    tools = await get_converted_tools(mcp_client)
    prompt = "Can you list the users in Grafana?"

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=prompt),
    ]

    # 1. Call the list_users_by_org tool
    messages = await llm_tool_call_sequence(
        model, messages, tools, mcp_client, "list_users_by_org"
    )

    # 2. Final LLM response
    response = await acompletion(model=model, messages=messages, tools=tools)
    content = response.choices[0].message.content
    user_checker = CustomLLMBooleanEvaluator(
        settings=CustomLLMBooleanSettings(
            prompt="Does the response contain specific information about users in Grafana, such as usernames, emails, or roles?",
        )
    )
    expect(input=prompt, output=content).to_pass(user_checker)
