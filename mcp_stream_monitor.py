#!/usr/bin/env python3
"""NICU 即時串流監測 MCP。

獨立 HTTP 服務，只轉發正在跑的 web_app.py API，
不自行建立 VideoScreenDigitExtractor、不接 Cursor。
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from typing import Any, Optional
from urllib.parse import quote

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 mcp 套件。請用 ./run_mcp_stream_monitor.sh 啟動"
        "（會建立獨立 .venv-mcp-stream），不要裝進 webapp_mongo_gpu。"
    ) from exc

logging.getLogger("httpx").setLevel(logging.WARNING)

WEB_APP_BASE_URL = os.environ.get("WEB_APP_BASE_URL", "http://127.0.0.1:3001").rstrip("/")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
DEFAULT_HOST = os.environ.get("MCP_STREAM_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MCP_STREAM_PORT", "8765"))
DEFAULT_TIMEOUT = httpx.Timeout(30.0)
START_TIMEOUT = httpx.Timeout(60.0)

mcp = FastMCP(
    "nicu-stream-monitor",
    instructions=(
        "NICU 即時 RTSP 監測 MCP。啟動/停止會作用在 Jetson 上正在跑的 web_app。"
        "啟動前先 list_stream_monitors、list_camera_templates、list_patient_ids。"
        "可用 template_id 帶入攝影機名稱與 RTSP。"
        "不要把 RTSP 密碼或 API key 唸給使用者聽。"
        "web_app 沒在跑時不要試圖重啟監測服務。"
    ),
)


def _redact_rtsp(url: str | None) -> str | None:
    if not url:
        return url
    return re.sub(r"(rtsp[s]?://)([^/@]+)@", r"\1***@", url)


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in {"api_key", "password"}:
                continue
            if key == "rtsp_url" and isinstance(value, str):
                out[key] = _redact_rtsp(value)
            else:
                out[key] = _sanitize(value)
        return out
    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    return obj


def _request(
    method: str,
    path: str,
    *,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    sanitize: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    url = f"{WEB_APP_BASE_URL}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, **kwargs)
    except httpx.ConnectError:
        return {
            "ok": False,
            "error": (
                f"連不上 web_app ({WEB_APP_BASE_URL})。"
                "請確認 medical-webapp / python web_app.py 已在 :3001 執行。"
                "MCP 不會自行啟動監測服務。"
            ),
        }
    except httpx.TimeoutException:
        return {"ok": False, "error": f"呼叫 web_app 逾時: {method} {path}"}

    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "status": resp.status_code, "error": resp.text[:500]}

    payload = data if isinstance(data, dict) else {"data": data}
    if resp.status_code >= 400:
        return {
            "ok": False,
            "status": resp.status_code,
            "error": payload.get("error", payload),
        }
    result = {"ok": True, "status": resp.status_code, **payload}
    return _sanitize(result) if sanitize else result


def _encode(value: str) -> str:
    return quote(str(value), safe="")


@mcp.tool()
def list_patient_ids() -> dict[str, Any]:
    """列出已存在的病患 ID。啟動監測前用這個確認 ID，或讓使用者輸入新的。"""
    return _request("GET", "/api/patient-ids")


@mcp.tool()
def list_camera_templates() -> dict[str, Any]:
    """列出已儲存的攝影機範本（例如 camera-1、camera-2）。RTSP 帳密會被遮罩。"""
    return _request("GET", "/api/stream/templates")


@mcp.tool()
def save_camera_template(
    template_id: str,
    camera_name: str,
    rtsp_url: str,
    provider: str = "openrouter",
    model: str = "google/gemini-3.8-flash",
    capture_interval_minutes: int = 1,
) -> dict[str, Any]:
    """儲存攝影機範本。對應畫面「儲存範本」。不要在對話裡回傳完整 RTSP。"""
    return _request(
        "POST",
        "/api/stream/templates",
        json={
            "template_id": template_id,
            "camera_name": camera_name,
            "rtsp_url": rtsp_url,
            "provider": provider,
            "model": model,
            "capture_interval_minutes": capture_interval_minutes,
        },
    )


@mcp.tool()
def delete_camera_template(template_id: str) -> dict[str, Any]:
    """刪除攝影機範本。"""
    return _request("DELETE", f"/api/stream/templates/{_encode(template_id)}")


@mcp.tool()
def list_stream_monitors() -> dict[str, Any]:
    """列出目前監測卡片（含已停止）。對應畫面右側「目前監測中」。"""
    return _request("GET", "/api/stream/list")


@mcp.tool()
def start_stream_monitor(
    patient_id: str,
    camera_name: str = "",
    rtsp_url: str = "",
    provider: str = "openrouter",
    model: str = "google/gemini-3.8-flash",
    capture_interval_minutes: int = 1,
    session_id: Optional[str] = None,
    template_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """啟動即時監測。

    新任務需要 patient_id + camera_name + rtsp_url，或改傳 template_id 帶入攝影機設定。
    重啟已停止卡片時傳 session_id。
    OpenRouter 金鑰優先用參數，否則用環境變數 OPENROUTER_API_KEY。
    """
    resolved_camera = camera_name.strip()
    resolved_rtsp = rtsp_url.strip()
    resolved_provider = provider.strip() or "openrouter"
    resolved_model = model.strip() or "google/gemini-3.8-flash"

    if template_id:
        templates_resp = _request("GET", "/api/stream/templates", sanitize=False)
        if not templates_resp.get("ok"):
            return templates_resp
        match = None
        for item in templates_resp.get("templates") or []:
            if str(item.get("template_id") or "") == template_id:
                match = item
                break
        if match is None:
            return {"ok": False, "error": f"找不到攝影機範本: {template_id}"}
        resolved_camera = resolved_camera or str(match.get("camera_name") or "").strip()
        resolved_rtsp = resolved_rtsp or str(match.get("rtsp_url") or "").strip()
        template_provider = str(match.get("provider") or "").strip()
        template_model = str(match.get("model") or match.get("llm_model") or "").strip()
        if template_provider and (not provider.strip() or provider.strip() == "openrouter"):
            resolved_provider = template_provider
        if template_model and (
            not model.strip() or model.strip() == "google/gemini-3.8-flash"
        ):
            resolved_model = template_model

    key = (api_key or OPENROUTER_API_KEY).strip()
    if resolved_provider == "openrouter" and not key:
        return {
            "ok": False,
            "error": "缺少 OpenRouter API Key。請設 OPENROUTER_API_KEY，或在參數傳 api_key。",
        }
    if not str(patient_id).strip() or not resolved_camera or not resolved_rtsp:
        return {
            "ok": False,
            "error": "缺少病患 ID、攝影機名稱或 RTSP URL。可改傳 template_id 帶入攝影機設定。",
        }

    payload: dict[str, Any] = {
        "patient_id": str(patient_id).strip(),
        "camera_name": resolved_camera,
        "rtsp_url": resolved_rtsp,
        "provider": resolved_provider,
        "model": resolved_model,
        "capture_interval_minutes": capture_interval_minutes,
        "capture_interval_seconds": capture_interval_minutes * 60,
        "api_key": key,
    }
    if session_id:
        payload["session_id"] = session_id
    return _request("POST", "/api/stream/start", timeout=START_TIMEOUT, json=payload)


@mcp.tool()
def stop_stream_monitor(session_id: str) -> dict[str, Any]:
    """停止正在跑的監測。對應卡片「停止」。"""
    return _request("POST", f"/api/stream/stop/{_encode(session_id)}")


@mcp.tool()
def delete_stream_monitor(session_id: str) -> dict[str, Any]:
    """移除監測卡片。必須先停止。對應「移除卡片」。"""
    return _request("POST", f"/api/stream/delete/{_encode(session_id)}")


@mcp.tool()
def get_stream_status(session_id: str) -> dict[str, Any]:
    """查單一 session 是否在記憶體中運行、最新分析、last_error。"""
    return _request("GET", f"/api/stream/status/{_encode(session_id)}")


@mcp.tool()
def get_latest_vitals(
    session_id: str,
    page: int = 1,
    page_size: int = 12,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """讀該 session 最新 vitals（MongoDB）。對應「歷史資料」。"""
    params: dict[str, Any] = {"page": page, "page_size": min(max(page_size, 1), 50)}
    if model:
        params["model"] = model
    return _request(
        "GET",
        f"/api/mongodb/medical_values/{_encode(session_id)}",
        params=params,
    )


@mcp.tool()
def get_capture_interval() -> dict[str, Any]:
    """讀全域分析間隔（分鐘）。"""
    return _request("GET", "/api/stream/global_interval")


@mcp.tool()
def set_capture_interval(capture_interval_minutes: int) -> dict[str, Any]:
    """設定全域分析間隔，僅接受 1~60 整數分鐘。"""
    return _request(
        "POST",
        "/api/stream/global_interval",
        json={"capture_interval_minutes": capture_interval_minutes},
    )


def _run(host: str, port: int) -> None:
    # mcp 1.x: host/port 在 FastMCP.settings，不在 run()。
    mcp.settings.host = host
    mcp.settings.port = port
    print(
        f"NICU stream MCP listening on http://{host}:{port}/mcp "
        f"(web_app={WEB_APP_BASE_URL})",
        flush=True,
    )
    mcp.run(transport="streamable-http")


def main() -> None:
    parser = argparse.ArgumentParser(description="NICU 即時監測 MCP（獨立 HTTP，不接 Cursor）")
    parser.add_argument("--host", default=DEFAULT_HOST, help="只綁 127.0.0.1，不要對區網開放")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    _run(args.host, args.port)


if __name__ == "__main__":
    main()
