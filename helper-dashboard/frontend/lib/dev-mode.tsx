"use client";

// Developer-mode context. The VALUE is provided by the shell (app/page.tsx)
// once the developer token has been verified server-side; this module only
// defines the contract. Widgets and notices read `unlocked` to decide
// whether a 「技術細節」 disclosure (PromQL, raw errors) may render at all.
// Capabilities never authorize anything — the backend gates every dev route.

import React, { createContext, useContext } from "react";

export interface DevModeState {
  unlocked: boolean;
}

export const DevModeContext = createContext<DevModeState>({ unlocked: false });

export function useDevMode(): DevModeState {
  return useContext(DevModeContext);
}

export function DevModeProvider({
  unlocked,
  children,
}: {
  unlocked: boolean;
  children: React.ReactNode;
}) {
  return (
    <DevModeContext.Provider value={{ unlocked }}>
      {children}
    </DevModeContext.Provider>
  );
}
