"use client";

import { useEffect, useRef, useState } from "react";

export type BackendStatus = "checking" | "up" | "down";

export interface BackendStatusState {
  status: BackendStatus;
  lastCheck: number | null;
  httpCode: number | null;
  error: string | null;
}

const POLL_MS = 15000;

export function useBackendStatus(): BackendStatusState {
  const [state, setState] = useState<BackendStatusState>({
    status: "checking",
    lastCheck: null,
    httpCode: null,
    error: null,
  });
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;

    async function ping() {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        if (cancelled.current) return;
        setState({
          status: res.ok ? "up" : "down",
          lastCheck: Date.now(),
          httpCode: res.status,
          error: res.ok ? null : res.statusText || null,
        });
      } catch (e) {
        if (cancelled.current) return;
        setState({
          status: "down",
          lastCheck: Date.now(),
          httpCode: null,
          error: e instanceof Error ? e.message : String(e),
        });
      }
    }

    ping();
    const t = setInterval(ping, POLL_MS);
    return () => {
      cancelled.current = true;
      clearInterval(t);
    };
  }, []);

  return state;
}
