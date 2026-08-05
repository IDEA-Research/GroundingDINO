import json
from litellm.types.utils import ModelResponse
from litellm import acompletion, Choices, Message
from mcp.types import TextContent, Tool


async def assert_and_handle_tool_call(
    response: ModelResponse,
    mcp_client,
    expected_tool: str,
    expected_args: dict = None,
) -> list:
    messages = []
    tool_calls = []
    for c in response.choices:
        assert isinstance(c, Choices)
        tool_calls.extend(c.message.tool_calls or [])
        messages.append(c.message)

    # Better error message if wrong number of tool calls
    if len(tool_calls) != 1:
        actual_calls = [tc.function.name for tc in tool_calls] if tool_calls else []
        assert len(tool_calls) == 1, (
            f"\n❌ Expected exactly 1 tool call, got {len(tool_calls)}\n"
            f"Expected tool: {expected_tool}\n"
            f"Actual tools called: {actual_calls}\n"
            f"LLM response: {response.choices[0].message.content if response.choices else 'N/A'}"
        )

    for tool_call in tool_calls:
        actual_tool = tool_call.function.name
        if actual_tool != expected_tool:
            # Parse arguments to understand what LLM was trying to do
            try:
                actual_args = (
                    json.loads(tool_call.function.arguments)
                    if tool_call.function.arguments
                    else {}
                )
            except:
                actual_args = tool_call.function.arguments

            assert False, (
                f"\n❌ LLM called wrong tool!\n"
                f"Expected: {expected_tool}\n"
                f"Got:      {actual_tool}\n"
                f"With args: {json.dumps(actual_args, indent=2)}\n"
                f"\n💡 Debugging tips:\n"
                f"   - Check if the prompt clearly indicates which tool to use\n"
                f"   - Verify the expected tool exists in the available tools\n"
                f"   - Consider if the tool description is clear enough\n"
            )
        arguments = (
            {}
            if len(tool_call.function.arguments) == 0
            else json.loads(tool_call.function.arguments)
        )
        if expected_args:
            for key, value in expected_args.items():
                if key not in arguments:
                    assert False, (
                        f"\n❌ Missing expected parameter '{key}'\n"
                        f"Expected args: {json.dumps(expected_args, indent=2)}\n"
                        f"Actual args:   {json.dumps(arguments, indent=2)}\n"
                    )
                if arguments[key] != value:
                    assert False, (
                        f"\n❌ Wrong value for parameter '{key}'\n"
                        f"Expected: {value}\n"
                        f"Got:      {arguments[key]}\n"
                        f"Full args: {json.dumps(arguments, indent=2)}\n"
                    )
        result = await mcp_client.call_tool(tool_call.function.name, arguments)
        assert len(result.content) == 1, (
            f"Expected one result for tool {tool_call.function.name}, got {len(result.content)}"
        )
        assert isinstance(result.content[0], TextContent), (
            f"Expected TextContent for tool {tool_call.function.name}, got {type(result.content[0])}"
        )
        messages.append(
            Message(
                role="tool", tool_call_id=tool_call.id, content=result.content[0].text
            )
        )
    return messages


def convert_tool(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                **tool.inputSchema,
                "properties": tool.inputSchema.get("properties", {}),
            },
        },
    }


async def llm_tool_call_sequence(
    model, messages, tools, mcp_client, tool_name, tool_args=None
):
    print(f"\n🤖 Calling LLM ({model}) and expecting tool: {tool_name}")
    print(f"📝 Last message: {messages[-1].get('content', messages[-1])[:200]}...")

    response = await acompletion(
        model=model,
        messages=messages,
        tools=tools,
    )
    assert isinstance(response, ModelResponse)

    # Print what tool was actually called for debugging
    if response.choices and response.choices[0].message.tool_calls:
        actual_tool = response.choices[0].message.tool_calls[0].function.name
        print(f"✅ LLM called: {actual_tool}")
        if actual_tool != tool_name:
            print(f"⚠️  WARNING: Expected {tool_name} but got {actual_tool}")

    messages.extend(
        await assert_and_handle_tool_call(
            response, mcp_client, tool_name, tool_args or {}
        )
    )
    return messages


async def get_converted_tools(mcp_client):
    tools = await mcp_client.list_tools()
    return [convert_tool(t) for t in tools.tools]


async def flexible_tool_call(
    model, messages, tools, mcp_client, expected_tool_name, required_params=None
):
    """
    Make a flexible tool call that only checks essential parameters.
    Returns updated messages list.

    Args:
        model: The LLM model to use
        messages: Current conversation messages
        tools: Available tools
        mcp_client: MCP client session
        expected_tool_name: Name of the tool we expect to be called
        required_params: Dict of essential parameters to check (optional)

    Returns:
        Updated messages list including tool call and result
    """
    response = await acompletion(model=model, messages=messages, tools=tools)

    # Check that a tool call was made
    assert response.choices[0].message.tool_calls is not None, (
        f"Expected tool call for {expected_tool_name}"
    )
    assert len(response.choices[0].message.tool_calls) >= 1, (
        f"Expected at least one tool call for {expected_tool_name}"
    )

    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == expected_tool_name, (
        f"Expected {expected_tool_name} tool, got {tool_call.function.name}"
    )

    arguments = json.loads(tool_call.function.arguments)

    # Check required parameters if specified
    if required_params:
        for key, expected_value in required_params.items():
            assert key in arguments, f"Expected parameter '{key}' in tool arguments"
            if expected_value is not None:
                assert arguments[key] == expected_value, (
                    f"Expected {key}='{expected_value}', got {key}='{arguments.get(key)}'"
                )

    # Call the tool to verify it works
    result = await mcp_client.call_tool(tool_call.function.name, arguments)
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)

    # Add both the tool call and result to message history
    messages.append(response.choices[0].message)
    messages.append(
        Message(role="tool", tool_call_id=tool_call.id, content=result.content[0].text)
    )

    return messages
