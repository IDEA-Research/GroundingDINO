"use client";

import React from "react";

import type { WidgetSpec } from "@/lib/spec-schema";
import { useQueryData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

export function GaugeWidget({ widget }: { widget: WidgetSpec }) {
  const data = useQueryData(widget.query);
  const raw = data.latest ?? 0;
  const min = Number((widget.options as any)?.min ?? 0);
  const max = Number((widget.options as any)?.max ?? 100);
  const pct = Math.min(
    1,
    Math.max(0, max === min ? 0 : (raw - min) / (max - min)),
  );
  const angle = -90 + 180 * pct; // half-circle gauge

  return (
    <WidgetFrame widget={widget} badge={data.source === "mock" ? "mock" : null}>
      <div className="flex h-full items-center justify-center">
        <svg viewBox="-110 -100 220 120" className="h-full w-full max-h-40">
          <path
            d="M -90 0 A 90 90 0 0 1 90 0"
            stroke="#1e293b"
            strokeWidth={14}
            fill="none"
          />
          <path
            d={arcPath(90, pct)}
            stroke={widget.encoding?.color || "#38bdf8"}
            strokeWidth={14}
            fill="none"
            strokeLinecap="round"
          />
          <g transform={`rotate(${angle})`}>
            <line x1={0} y1={0} x2={0} y2={-72} stroke="#e2e8f0" strokeWidth={2} />
            <circle r={5} fill="#e2e8f0" />
          </g>
          <text
            x={0}
            y={22}
            textAnchor="middle"
            fill="#e2e8f0"
            fontSize={18}
            fontWeight={700}
          >
            {raw.toFixed(1)}
            {widget.encoding?.unit ? ` ${widget.encoding.unit}` : ""}
          </text>
        </svg>
      </div>
    </WidgetFrame>
  );
}

function arcPath(r: number, pct: number): string {
  const end = -Math.PI + Math.PI * pct;
  const x = r * Math.cos(end);
  const y = r * Math.sin(end);
  const large = pct > 0.5 ? 1 : 0;
  return `M -${r} 0 A ${r} ${r} 0 ${large} 1 ${x.toFixed(2)} ${y.toFixed(2)}`;
}
