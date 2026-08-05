"use client";

import React, { useEffect, useState } from "react";

import { DashboardPreview } from "@/components/dashboard/DashboardPreview";
import { api } from "@/lib/api";
import type { DashboardSpec } from "@/lib/spec-schema";

// Public rendering route used by the browser evaluator (Playwright).
// It fetches the spec by id and renders it with the standard preview.

export default function DashboardByIdPage({
  params,
}: {
  params: { id: string };
}) {
  const [spec, setSpec] = useState<DashboardSpec | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getDashboard(params.id)
      .then((r) => {
        if (!cancelled) setSpec(r.spec);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [params.id]);

  if (error) {
    return (
      <div
        data-dashboard-state="error"
        className="flex h-screen items-center justify-center text-rose-300"
      >
        {error}
      </div>
    );
  }

  return (
    <div data-dashboard-state={spec ? "ready" : "loading"} className="h-screen">
      <DashboardPreview spec={spec} />
    </div>
  );
}
