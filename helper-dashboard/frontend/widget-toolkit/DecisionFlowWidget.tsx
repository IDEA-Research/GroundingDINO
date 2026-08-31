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
      return { text: "gate", cls: "bg-chart-4/15 text-chart-4" };
    case "decision":
      return { text: "decision", cls: "bg-chart-3/15 text-chart-3" };
    case "action":
      return { text: "action", cls: "bg-chart-5/15 text-chart-5" };
    case "terminal":
      return { text: "done", cls: "bg-ok/15 text-ok" };
    case "signal_lost":
      return { text: "signal lost", cls: "bg-alarm-high/15 text-alarm-high" };
    default:
      return { text: kind, cls: "bg-surface-3 text-ink-dim" };
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
  // dashboard query path. Store-read failure degrades LOUDLY.
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
          <div className="text-xs text-ink-faint">
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
            <div className="rounded border border-alarm-high/60 bg-alarm-high/10 p-2 text-ink">
              Verdict path is locked: the data-integrity gate has not passed.
              Resolve signal integrity before continuing.
            </div>
          ) : (
            <div className="rounded border border-edge bg-surface-2/60 p-2">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-semibold text-ink">
                  Step {clampedIdx + 1} / {steps.length}: {step.title}
                </span>
                {stepNode ? (
                  <span
                    className={
                      "rounded px-1.5 py-0.5 text-2xs uppercase " +
                      nodeKindBadge(stepNode.kind).cls
                    }
                  >
                    {nodeKindBadge(stepNode.kind).text}
                  </span>
                ) : null}
              </div>
              {stepNode ? (
                <div className="mb-1 text-ink-dim">
                  {stepNode.label}
                  {stepNode.signal ? (
                    <span className="ml-2 rounded bg-surface-3 px-1 py-0.5 font-mono text-2xs text-ink-dim">
                      {stepNode.signal}
                    </span>
                  ) : null}
                </div>
              ) : null}
              <div className="text-ink-faint">{step.guidance}</div>

              {/* Outgoing branches for this node, from the real graph edges. */}
              <BranchList flow={flow} nodeId={step.node} />
            </div>
          )}
        </div>

        {/* Step navigation — client-side state only. */}
        <div className="mt-2 flex items-center justify-between">
          <button
            type="button"
            className="rounded border border-edge-strong px-2 py-1 text-ink-dim hover:bg-surface-2 hover:text-ink disabled:opacity-40"
            disabled={clampedIdx === 0}
            onClick={() => setStepIdx((i) => Math.max(0, i - 1))}
          >
            Back
          </button>
          <span className="text-ink-faint">
            {flow.inputs.length} inputs · {flow.nodes.length} nodes
          </span>
          <button
            type="button"
            className="rounded border border-edge-strong px-2 py-1 text-ink-dim hover:bg-surface-2 hover:text-ink disabled:opacity-40"
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
            className="flex items-center justify-between rounded border border-edge px-2 py-1"
          >
            <span className="text-ink-dim">{e.condition}</span>
            <span className="text-ink-faint">
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
      <div className="rounded border border-edge bg-surface-2/60 px-2 py-1 text-ink-faint">
        Checking data-integrity gate…
      </div>
    );
  }
  if (gate.kind === "ok") {
    return (
      <div className="rounded border border-ok/40 bg-ok/10 px-2 py-1 text-ok">
        Data-integrity gate: PASSED (real signal, fresh).
      </div>
    );
  }
  if (gate.kind === "store_unavailable") {
    return (
      <div className="rounded border border-alarm-high bg-alarm-high/10 px-2 py-1 font-semibold text-alarm-high">
        SIGNAL UNKNOWN — alert store unreachable. Do not interpret; treat as
        signal lost.
      </div>
    );
  }
  if (gate.kind === "no_data") {
    return (
      <div className="rounded border border-alarm-med/60 bg-alarm-med/10 px-2 py-1 text-alarm-med">
        No alert events yet — awaiting first evaluation. Do not interpret as
        "normal".
      </div>
    );
  }
  if (gate.kind === "insufficient_baseline") {
    return (
      <div className="rounded border border-alarm-med/60 bg-alarm-med/10 px-2 py-1 text-alarm-med">
        INSUFFICIENT BASELINE — under 24h of history. No verdict.{" "}
        {gate.message}
      </div>
    );
  }
  // signal_lost
  return (
    <div className="rounded border border-alarm-high bg-alarm-high/10 px-2 py-1 font-semibold text-alarm-high">
      SIGNAL LOST{gate.reason ? ` (${gate.reason})` : ""} — do NOT interpret.{" "}
      {gate.message}
    </div>
  );
}

function Disclaimer() {
  return (
    <div
      data-non-diagnostic="true"
      className="mt-2 border-t border-edge pt-1 text-2xs italic text-ink-faint"
    >
      {DISCLAIMER}
    </div>
  );
}
