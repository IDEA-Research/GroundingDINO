"use client";

// Theme preference (light | dark | system) persisted in localStorage under
// "hd.theme". The resolved theme is written to <html data-theme="..."> —
// the inline boot script in app/layout.tsx does the same before first paint.
// Every change dispatches a "hd:theme" CustomEvent on window so non-React
// consumers (lib/chart-theme.ts) can re-read CSS variables.

import { useCallback, useEffect, useState } from "react";

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "hd.theme";
export const THEME_EVENT = "hd:theme";

export interface ThemeEventDetail {
  preference: ThemePreference;
  resolved: ResolvedTheme;
}

export const THEME_LABELS: Record<ThemePreference, string> = {
  light: "淺色",
  dark: "深色",
  system: "系統",
};

const ORDER: ThemePreference[] = ["light", "dark", "system"];

function isPreference(v: unknown): v is ThemePreference {
  return v === "light" || v === "dark" || v === "system";
}

export function readStoredTheme(): ThemePreference {
  if (typeof window === "undefined") return "system";
  try {
    const v = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isPreference(v) ? v : "system";
  } catch {
    return "system";
  }
}

function systemPrefersDark(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    return false;
  }
}

export function resolveTheme(pref: ThemePreference): ResolvedTheme {
  if (pref === "light" || pref === "dark") return pref;
  return systemPrefersDark() ? "dark" : "light";
}

// Current theme as painted (reads the attribute the boot script set).
export function currentResolvedTheme(): ResolvedTheme {
  if (typeof document === "undefined") return "light";
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "dark" || attr === "light") return attr;
  return resolveTheme(readStoredTheme());
}

// Persist + apply + notify. Safe to call from anywhere on the client.
export function applyTheme(pref: ThemePreference): ResolvedTheme {
  const resolved = resolveTheme(pref);
  if (typeof document !== "undefined") {
    const html = document.documentElement;
    html.setAttribute("data-theme", resolved);
    html.style.colorScheme = resolved;
  }
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, pref);
    } catch {
      // storage unavailable (private mode) — the attribute still applied
    }
    const detail: ThemeEventDetail = { preference: pref, resolved };
    window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail }));
  }
  return resolved;
}

export interface ThemeState {
  theme: ThemePreference;
  resolved: ResolvedTheme;
  setTheme: (pref: ThemePreference) => void;
  // 淺色 -> 深色 -> 系統 -> 淺色
  cycle: () => void;
  label: string;
  nextLabel: string;
}

// SSR-safe: renders "system"/"light" on the server and syncs on mount.
export function useTheme(): ThemeState {
  const [theme, setThemeState] = useState<ThemePreference>("system");
  const [resolved, setResolved] = useState<ResolvedTheme>("light");

  useEffect(() => {
    setThemeState(readStoredTheme());
    setResolved(currentResolvedTheme());

    const onTheme = (e: Event) => {
      const d = (e as CustomEvent<ThemeEventDetail>).detail;
      if (d && isPreference(d.preference)) {
        setThemeState(d.preference);
        setResolved(d.resolved);
      }
    };
    const onStorage = (e: StorageEvent) => {
      if (e.key === THEME_STORAGE_KEY) {
        const pref = readStoredTheme();
        setThemeState(pref);
        setResolved(applyTheme(pref));
      }
    };
    let mql: MediaQueryList | null = null;
    const onSystem = () => {
      if (readStoredTheme() === "system") {
        setResolved(applyTheme("system"));
      }
    };
    window.addEventListener(THEME_EVENT, onTheme);
    window.addEventListener("storage", onStorage);
    try {
      mql = window.matchMedia("(prefers-color-scheme: dark)");
      mql.addEventListener("change", onSystem);
    } catch {
      mql = null;
    }
    return () => {
      window.removeEventListener(THEME_EVENT, onTheme);
      window.removeEventListener("storage", onStorage);
      if (mql) mql.removeEventListener("change", onSystem);
    };
  }, []);

  const setTheme = useCallback((pref: ThemePreference) => {
    setThemeState(pref);
    setResolved(applyTheme(pref));
  }, []);

  const cycle = useCallback(() => {
    const next = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];
    setTheme(next);
  }, [theme, setTheme]);

  const nextPref = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];

  return {
    theme,
    resolved,
    setTheme,
    cycle,
    label: THEME_LABELS[theme],
    nextLabel: THEME_LABELS[nextPref],
  };
}
