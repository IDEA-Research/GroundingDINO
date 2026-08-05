"use client";

import React from "react";

import { api } from "@/lib/api";
import type {
  AlertEventDTO,
  DecisionFlowSpec,
  FlowNode,
  WidgetSpec,
} from "@/lib/spec-schema";
import { WidgetFrame } from "./WidgetFrame";

// Persistent, non-suppressible disclaimer. This is a hard clinical-safety
// requirement: every alerting surface carries "decision-support, not a
// diagnosis". It is rendered unconditionally, outside any branch.
const DISCLAIMER =
  "Decision support, not a diagnosis. Verify signal and physiology before acting.";

// How the data-integrity gate resolves for the CURRENTLY guided rule, derived
// ONLY from real alert events. This mirrors the evaluator: the gate runs first,
// and signal_lost / insufficient_baseline are first-class NON-clinical states —
// they are surfaced loudly and must never be interpreted as a clinical branch.
type GateState =
  | { kind: "ok" }
  | { kind: "signal_lost"; reason: string | null; message: string }
  | { kind: "insufficient_baseline"; message: string }
  | { kind: "no_data" }
  | { kind: "store_unavailable" }
  | { kind: "loading" };

function nodeKindBadge(kind: FlowNode["kind"]): { text: string; cls: string } {
  switch (kind) {
    case "data_integrity_gate":
      return { text: "gate", cls: "bg-sky-800 text-sky-100" };
    case "decision":
      return { text: "decision", cls: "bg-indigo-800 text-indigo-100" };
    case "action":
      return { text: "action", cls: "bg-amber-800 text-amber-100" };
    case "terminal":
      return { text: "done", cls: "bg-emerald-800 text-emerald-100" };
    case "signal_lost":
      return { text: "signal lost", cls: "bg-red-800 text-red-100" };
    default:
      return { text: kind, cls: "bg-slate-700 text-slate-200" };
  }
}

// Reduce the real alert-event stream to a single gate verdict. Fail-closed:
// anything that is not an explicit clinical "ok" (pending/firing/resolved)
// keeps the flow OUT of the verdict path.
function deriveGate(
  events: AlertEventDTO[] | null,
  storeUnavailable: boolean,
): GateState {
  if (storeUnavailable) return { kind: "store_unavailable" };
  if (events === null) return { kind: "loading" };
  if (events.length === 0) return { kind: "no_data" };

  // Most recent event wins (events arrive oldest-first from the history log).
  const latest = events[events.length - 1];
  switch (latest.state) {
    case "signal_lost":
      return {
        kind: "signal_lost",
        reason: latest.signal_lost_reason ?? null,
        message: latest.message,
      };
    case "insufficient_baseline":
      return { kind: "insufficient_baseline", message: latest.message };
    default:
      // pending / firing / resolved — the data-integrity gate passed, so the
      // clinician may walk the verdict branches.
      return { kind: "ok" };
  }
}

export function DecisionFlowWidget({ widget }: { widget: WidgetSpec }) {
  const flow: DecisionFlowSpec | null = widget.decision_flow ?? null;

  const [events, setEvents] = React.useState<AlertEventDTO[] | null>(null);
  const [storeUnavailable, setStoreUnavailable] = React.useState(false);
  const [stepIdx, setStepIdx] = React.useState(0);

  // Load REAL alert events from the anomaly store (INC2/INC3). Never the
  // AlertListWidget MOCK_ALERTS. Store-read failure degrades LOUDLY.
  React.useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await api.anomalyAlerts();
        if (cancelled) return;
        setEvents(res.events ?? []);
        setStoreUnavailable(Boolean(res.store_unavailable));
      } catch {
        if (cancelled) return;
        setEvents([]);
        setStoreUnavailable(true);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [widget.id]);

  const gate = deriveGate(events, storeUnavailable);

  if (!flow || flow.nodes.length === 0 || flow.steps.length === 0) {
    return (
      <WidgetFrame widget={widget} badge="decision flow">
        <div className="flex h-full flex-col justify-between">
          <div className="text-xs text-slate-400">
            No decision flow configured.
          </div>
          <Disclaimer />
        </div>
      </WidgetFrame>
    );
  }

  const nodesById = new Map(flow.nodes.map((n) => [n.id, n]));
  const steps = flow.steps;
  const clampedIdx = Math.min(Math.max(stepIdx, 0), steps.length - 1);
  const step = steps[clampedIdx];
  const stepNode = nodesById.get(step.node);

  // The first node of every clinical flow is the data-integrity gate. Until the
  // gate resolves to "ok", the clinician is HELD on step 0 (the gate); verdict
  // branches are unreachable. This mirrors the evaluator's inviolable ordering.
  const gateBlocks = gate.kind !== "ok";
  const onGateStep = clampedIdx === 0;

  return (
    <WidgetFrame widget={widget} badge="decision flow">
      <div className="flex h-full flex-col text-xs">
        {/* Gate status banner — always visible, loud when degraded. */}
        <GateBanner gate={gate} />

        {/* Current step. When the gate blocks and we are past step 0, we hold
            the clinician back rather than showing a verdict. */}
        <div className="mt-2 flex-1 overflow-auto">
          {gateBlocks && !onGateStep ? (
            <div className="rounded border border-red-800 bg-red-950/40 p-2 text-red-100">
              Verdict path is locked: the data-integrity gate has not passed.
              Resolve signal integrity before continuing.
            </div>
          ) : (
            <div className="rounded border border-slate-800 bg-slate-900/40 p-2">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-semibold text-slate-100">
                  Step {clampedIdx + 1} / {steps.length}: {step.title}
                </span>
                {stepNode ? (
                  <span
                    className={
                      "rounded px-1.5 py-0.5 text-[10px] uppercase " +
                      nodeKindBadge(stepNode.kind).cls
                    }
                  >
                    {nodeKindBadge(stepNode.kind).text}
                  </span>
                ) : null}
              </div>
              {stepNode ? (
                <div className="mb-1 text-slate-300">
                  {stepNode.label}
                  {stepNode.signal ? (
                    <span className="ml-2 rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-300">
                      {stepNode.signal}
                    </span>
                  ) : null}
                </div>
              ) : null}
              <div className="text-slate-400">{step.guidance}</div>

              {/* Outgoing branches for this node, from the real graph edges. */}
              <BranchList flow={flow} nodeId={step.node} />
            </div>
          )}
        </div>

        {/* Step navigation — client-side state only. */}
        <div className="mt-2 flex items-center justify-between">
          <button
            type="button"
            className="rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40"
            disabled={clampedIdx === 0}
            onClick={() => setStepIdx((i) => Math.max(0, i - 1))}
          >
            Back
          </button>
          <span className="text-slate-500">
            {flow.inputs.length} inputs · {flow.nodes.length} nodes
          </span>
          <button
            type="button"
            className="rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40"
            // Cannot advance past the gate step until the gate is ok.
            disabled={
              clampedIdx >= steps.length - 1 || (onGateStep && gateBlocks)
            }
            onClick={() =>
              setStepIdx((i) => Math.min(steps.length - 1, i + 1))
            }
          >
            Next
          </button>
        </div>

        <Disclaimer />
      </div>
    </WidgetFrame>
  );
}

function BranchList({
  flow,
  nodeId,
}: {
  flow: DecisionFlowSpec;
  nodeId: string;
}) {
  const nodesById = new Map(flow.nodes.map((n) => [n.id, n]));
  const outgoing = flow.edges.filter((e) => e.from === nodeId);
  if (outgoing.length === 0) return null;
  return (
    <ul className="mt-2 space-y-1">
      {outgoing.map((e, i) => {
        const target = nodesById.get(e.to);
        return (
          <li
            key={i}
            className="flex items-center justify-between rounded border border-slate-800 px-2 py-1"
          >
            <span className="text-slate-300">{e.condition}</span>
            <span className="text-slate-500">
              → {target ? target.label : e.to}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function GateBanner({ gate }: { gate: GateState }) {
  if (gate.kind === "loading") {
    return (
      <div className="rounded border border-slate-700 bg-slate-900/40 px-2 py-1 text-slate-400">
        Checking data-integrity gate…
      </div>
    );
  }
  if (gate.kind === "ok") {
    return (
      <div className="rounded border border-emerald-800 bg-emerald-950/40 px-2 py-1 text-emerald-200">
        Data-integrity gate: PASSED (real signal, fresh).
      </div>
    );
  }
  if (gate.kind === "store_unavailable") {
    return (
      <div className="rounded border border-red-700 bg-red-950/50 px-2 py-1 font-semibold text-red-100">
        SIGNAL UNKNOWN — alert store unreachable. Do not interpret; treat as
        signal lost.
      </div>
    );
  }
  if (gate.kind === "no_data") {
    return (
      <div className="rounded border border-amber-700 bg-amber-950/40 px-2 py-1 text-amber-100">
        No alert events yet — awaiting first evaluation. Do not interpret as
        "normal".
      </div>
    );
  }
  if (gate.kind === "insufficient_baseline") {
    return (
      <div className="rounded border border-amber-700 bg-amber-950/40 px-2 py-1 text-amber-100">
        INSUFFICIENT BASELINE — under 24h of history. No verdict.{" "}
        {gate.message}
      </div>
    );
  }
  // signal_lost
  return (
    <div className="rounded border border-red-700 bg-red-950/50 px-2 py-1 font-semibold text-red-100">
      SIGNAL LOST{gate.reason ? ` (${gate.reason})` : ""} — do NOT interpret.{" "}
      {gate.message}
    </div>
  );
}

function Disclaimer() {
  return (
    <div
      data-non-diagnostic="true"
      className="mt-2 border-t border-slate-800 pt-1 text-[10px] italic text-slate-500"
    >
      {DISCLAIMER}
    </div>
  );
}
