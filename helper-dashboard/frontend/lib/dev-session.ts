"use client";

// Developer-session token plumbing (client side only).
//
// The token is entered by hand in the developer workspace, verified
// server-side (GET /api/devfix/config answers 200 only with a matching
// X-Developer-Token), and kept in sessionStorage so it dies with the tab.
// It is NEVER baked into the bundle and never persisted to localStorage.
// Possessing the token in the browser authorizes nothing by itself: every
// developer route re-checks it on the server (backend/app/api/_auth.py).

export const DEV_TOKEN_STORAGE_KEY = "hd.devToken";
export const DEV_TOKEN_HEADER = "X-Developer-Token";

function storage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function getDevToken(): string | null {
  const s = storage();
  if (!s) return null;
  try {
    const v = s.getItem(DEV_TOKEN_STORAGE_KEY);
    return v && v.trim() ? v : null;
  } catch {
    return null;
  }
}

export function setDevToken(token: string): void {
  const s = storage();
  if (!s) return;
  try {
    s.setItem(DEV_TOKEN_STORAGE_KEY, token.trim());
  } catch {
    // sessionStorage unavailable — the token still lives in React state
    // for this page load; the user re-enters it after a reload.
  }
}

export function clearDevToken(): void {
  const s = storage();
  if (!s) return;
  try {
    s.removeItem(DEV_TOKEN_STORAGE_KEY);
  } catch {
    // ignore
  }
}

// Header bag for fetch(): {} when no token is known, so a call made
// without a token simply gets the server's 403 (fail closed).
export function devHeaders(token?: string | null): Record<string, string> {
  const t = token ?? getDevToken();
  return t ? { [DEV_TOKEN_HEADER]: t } : {};
}
