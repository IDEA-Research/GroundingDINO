import httpx
import pytest

pytestmark = pytest.mark.anyio


async def test_healthz(mcp_transport: str, mcp_url: str):
    if mcp_transport == "stdio":
        return
    health_url = f"{mcp_url}/healthz"
    async with httpx.AsyncClient() as client:
        response = await client.get(health_url)
        assert response.status_code == 200
        assert response.text == "ok"
