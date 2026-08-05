import { NextResponse } from "next/server";

const BACKEND_BASE = (
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const REQUEST_TIMEOUT_MS = 30 * 1000;

export async function GET(
  _req: Request,
  ctx: { params: { jobId: string } },
) {
  const jobId = ctx.params.jobId;
  const target = `${BACKEND_BASE}/api/chat/message/${encodeURIComponent(jobId)}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(target, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    });
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: {
        "content-type": res.headers.get("content-type") || "application/json",
      },
    });
  } catch (e) {
    const detail = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    return NextResponse.json(
      { detail: `frontend proxy polling error: ${detail}` },
      { status: 502 },
    );
  } finally {
    clearTimeout(timer);
  }
}

