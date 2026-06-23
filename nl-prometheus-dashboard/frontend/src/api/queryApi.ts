import { apiRequest } from "./base";
import type { WidgetQueryResponse, WidgetSpec } from "../types/dashboard";

export async function queryWidget(widget: WidgetSpec, variables: Record<string, string>) {
  return apiRequest<WidgetQueryResponse>("/api/query/widget", {
    method: "POST",
    body: JSON.stringify({ widget, variables })
  });
}

