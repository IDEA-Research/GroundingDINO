// Client-side patch application, used by the inspector for optimistic
// previews. The backend is the source of truth — this helper only
// exists so the UI can render a patched spec before the server
// responds. It mirrors backend/app/services/patch_service.py.

import type { DashboardSpec, PatchOperation, PatchSpec } from "./spec-schema";

export function applyPatch(
  spec: DashboardSpec,
  patch: PatchSpec,
): DashboardSpec {
  if (patch.target_dashboard_id !== spec.dashboard_id) {
    throw new Error("patch target does not match dashboard");
  }
  const next: DashboardSpec = JSON.parse(JSON.stringify(spec));
  for (const op of patch.operations) {
    applyOp(next, op);
  }
  return next;
}

function applyOp(spec: DashboardSpec, op: PatchOperation): void {
  switch (op.op) {
    case "add_widget":
      spec.widgets.push(op.widget);
      return;
    case "remove_widget":
      spec.widgets = spec.widgets.filter((w) => w.id !== op.widget_id);
      return;
    case "update_widget": {
      const w = spec.widgets.find((x) => x.id === op.widget_id);
      if (!w) return;
      Object.assign(w, op.fields);
      return;
    }
    case "update_dashboard":
      Object.assign(spec, op.fields);
      return;
    case "reorder_widgets": {
      const byId = new Map(spec.widgets.map((w) => [w.id, w]));
      spec.widgets = op.order
        .map((id) => byId.get(id))
        .filter((w): w is DashboardSpec["widgets"][number] => Boolean(w));
      return;
    }
  }
}
