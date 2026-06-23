import { NextResponse } from "next/server";

const BACKEND_BASE = (
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const TARGET = `${BACKEND_BASE}/api/chat/message`;
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;

async function forwardOnce(payload: string, contentType: string): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(TARGET, {
      method: "POST",
      headers: {
        "content-type": contentType,
      },
      body: payload,
      signal: controller.signal,
      cache: "no-store",
    });
  } finally {
    clearTimeout(timer);
  }
}

export async function POST(req: Request) {
  try {
    const payload = await req.text();
    const contentType = req.headers.get("content-type") || "application/json";

    let res: Response;
    try {
      res = await forwardOnce(payload, contentType);
    } catch (e) {
      // Retry once on transient network/proxy issues (e.g. ECONNRESET)
      console.error("[next-proxy] first attempt failed", e);
      res = await forwardOnce(payload, contentType);
    }

    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: {
        "content-type": res.headers.get("content-type") || "application/json",
      },
    });
  } catch (e) {
    const detail = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    console.error("[next-proxy] /api/chat/message failed", detail);
    return NextResponse.json(
      { detail: `frontend proxy error: ${detail}` },
      { status: 502 },
    );
  }
}
