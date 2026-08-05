"""Single source of truth for widget-toolkit + envelope schemas,
rendered as short markdown blocks that get appended to LLM system
prompts.

The agents would otherwise have to "imagine" the schema, which is the
root cause of the dashboard-spec-agent inventing widget types AND the
big-guy-developer-agent's rescue path returning a DeveloperTicket with
the wrong field names (causing the user-visible "Something went wrong"
failure observed during the UI walk-through).

By generating these docs from the actual Pydantic models at import
time, the agents and the Python validator are guaranteed to agree.

This module is intentionally read-only and side-effect-free at import.
"""

from __future__ import annotations

from .patch_spec import (
    PatchOp,
    UPDATE_DASHBOARD_FIELDS,
    UPDATE_WIDGET_FIELDS,
)
from .widget_spec import (
    ALLOWED_OPTION_KEYS,
    FORBIDDEN_WIDGET_FIELDS,
    WidgetType,
)


# Hand-written per-key descriptions for the `options` whitelist. These
# do not exist anywhere else in the codebase; the validator only knows
# the keys are allowed, not what each one is for. Keep this in lockstep
# with ALLOWED_OPTION_KEYS — if you add a key there, add a one-line
# description here.
_OPTION_KEY_DOC: dict[str, str] = {
    "decimals": "int — fraction digits for numeric display",
    "show_grid": "bool — line/area: draw the cartesian grid",
    "show_legend": "bool — line/area: show series legend",
    "stacked": "bool — line: stack series instead of overlay",
    "fill": "bool — line: fill area below the curve",
    "min": "number — gauge/line: y-axis lower bound",
    "max": "number — gauge/line: y-axis upper bound",
    "columns": "list[str] — table: column ids in display order",
    "severity_filter": "str — alert_list: keep only this severity or higher",
    "row_limit": "int — table/alert_list: cap rendered rows",
    "sort_by": "str — table: column id to sort by",
    "sort_dir": "'asc' | 'desc' — table: sort direction",
    "show_labels": "bool — pie_chart: show slice labels",
    "donut": "bool — pie_chart: render as donut (hollow centre)",
    "horizontal": "bool — bar_chart: draw horizontal bars instead of vertical",
    "show_values": "bool — bar_chart: display numeric value on each bar",
    "x_label": "str — heatmap: label for the x-axis (e.g. 'hour')",
    "y_label": "str — heatmap: label for the y-axis (e.g. 'endpoint')",
    "color_scale": "str — heatmap: color scale name ('warm', 'cool', 'viridis')",
}


# Per-widget-type "typical" options. These are advisory hints for the
# LLM, not enforced by the validator (the validator only checks the
# union ALLOWED_OPTION_KEYS). The LLM uses this to pick sensible
# defaults for each widget type.
_TYPICAL_OPTIONS: dict[str, list[str]] = {
    "line_chart": ["show_grid", "show_legend", "stacked", "fill", "min", "max", "decimals"],
    "stat_card": ["decimals"],
    "gauge": ["min", "max", "decimals"],
    "table": ["columns", "row_limit", "sort_by", "sort_dir"],
    "alert_list": ["severity_filter", "row_limit"],
    "pie_chart": ["show_labels", "show_legend", "donut", "decimals"],
    "bar_chart": ["show_grid", "show_legend", "show_values", "horizontal", "stacked", "decimals"],
    "heatmap": ["x_label", "y_label", "color_scale", "decimals", "show_legend"],
    "decision_flow": [],
}


_TYPE_PURPOSE: dict[str, str] = {
    "line_chart": "Time-series over a range query. Best for trends, percentiles, throughput.",
    "stat_card": "Single big number from an instant query. Best for current KPIs.",
    "gauge": "Bounded instant value (needs min/max in options). Best for utilization 0..1.",
    "table": "Per-label breakdown from an instant or range query. Pair with `columns`.",
    "alert_list": "List of firing alerts from ALERTS{alertstate=\"firing\"} or similar.",
    "pie_chart": "Proportional breakdown from an instant query. Best for share-of-total by label.",
    "bar_chart": "Categorical comparison from an instant query. Best for per-node / per-service breakdowns.",
    "heatmap": "Two-dimensional grid of values coloured by intensity. Best for latency-by-endpoint-and-hour, error-rate matrices.",
    "decision_flow": (
        "Clinical decision-support flowchart. Carries a nested "
        "`decision_flow` config (nodes/edges/steps + a multi-input labeled "
        "query set over rSO2+SpO2+HR+MAP+FiO2). The FIRST node MUST be the "
        "data-integrity gate; it reads REAL alert events, shows a persistent "
        "non-diagnostic disclaimer, and never uses mock inputs."
    ),
}


def _types_section() -> str:
    lines = [
        f"## Widget types (authoritative — only these {len(list(WidgetType))} exist)"
    ]
    for wt in sorted(WidgetType, key=lambda x: x.value):
        purpose = _TYPE_PURPOSE.get(wt.value, "")
        typical = _TYPICAL_OPTIONS.get(wt.value, [])
        opts = ", ".join(f"`{k}`" for k in typical) if typical else "—"
        lines.append(f"- `{wt.value}` — {purpose} Typical options: {opts}.")
    lines.append(
        "\nAnything outside this list (e.g. `sankey`, `treemap`) "
        "does NOT exist. Do not emit it. If the user genuinely needs a "
        "different visualization, emit a DeveloperTicket."
    )
    return "\n".join(lines)


def _widget_fields_section() -> str:
    return (
        "## WidgetSpec required shape\n"
        "Every widget MUST have exactly these top-level fields, no others:\n"
        "- `id` — 1..64 chars, `[a-zA-Z0-9_-]+`\n"
        "- `type` — one of the widget types listed in the section above\n"
        "- `title` — 1..128 chars, no script/markup\n"
        "- `description` — optional, max 512 chars\n"
        "- `query` — `{source, promql, query_type, range?, step?}`. "
        "`source` is `prometheus` or `mock`. `query_type` is `instant` or "
        "`range`. `range` and `step` are required only when `query_type=range`.\n"
        "- `position` — `{x, y, w, h}`, all non-negative ints; "
        "x in [0,23], y in [0,999], w in [1,24], h in [1,60]\n"
        "- `encoding` — only `unit`, `legend`, `color` (no other keys)\n"
        "- `thresholds` — list of `{value: number, color: '#hex' or name, label?}`, max 16\n"
        "- `options` — see allowed-keys list below"
    )


def _options_section() -> str:
    lines = ["## Allowed `options` keys (anything else is rejected)"]
    for key in sorted(ALLOWED_OPTION_KEYS):
        doc = _OPTION_KEY_DOC.get(key, "(no description)")
        lines.append(f"- `{key}` — {doc}")
    lines.append(
        "\nIf an option you want is not in this list, leave `options: {}` and "
        "do not invent a key. Inventing keys forces the validator to reject "
        "your spec."
    )
    return "\n".join(lines)


def _decision_flow_section() -> str:
    from .widget_spec import DECISION_FLOW_SIGNALS, DATA_INTEGRITY_GATE_KIND

    signals = ", ".join(f"`{s}`" for s in DECISION_FLOW_SIGNALS)
    return (
        "## `decision_flow` config (required only for a decision_flow widget)\n"
        "A `decision_flow` widget adds one top-level `decision_flow` object "
        "(forbidden on every other widget type). It contains:\n"
        "- `nodes` — 2..64 `{id, kind, label, signal?}`. `kind` is one of "
        f"`{DATA_INTEGRITY_GATE_KIND}`, `decision`, `action`, `terminal`, "
        "`signal_lost`. The FIRST node MUST be the "
        f"`{DATA_INTEGRITY_GATE_KIND}` (data-integrity gate runs first, "
        "always). `signal` (on a decision node) must be one of the neonatal "
        f"signals: {signals}.\n"
        "- `edges` — 0..128 `{from, to, condition}`; `from`/`to` must resolve "
        "to node ids, `condition` is the human-readable branch guard.\n"
        "- `steps` — 1..64 `{id, title, node, guidance}`; drive the client-side "
        "guided wizard, each pointing at a node.\n"
        "- `inputs` — 1..16 `{label, query}` labeled query set. `label` is a "
        f"neonatal signal ({signals}); every `query.source` MUST be "
        "`prometheus` (a `mock` input is rejected — a clinical flow may never "
        "branch on fake/stale data).\n"
        "The widget reads REAL alert events from the anomaly store (never the "
        "alert_list mock) and always shows the non-diagnostic disclaimer."
    )


def _forbidden_section() -> str:
    return (
        "## Forbidden fields (rejected anywhere in the JSON tree)\n"
        + ", ".join(f"`{k}`" for k in sorted(FORBIDDEN_WIDGET_FIELDS))
        + ".\nDo not emit these under any key, in any nested object. Doing "
        "so causes immediate rejection by the Python validator."
    )


def _patch_section() -> str:
    ops = ", ".join(f"`{op.value}`" for op in sorted(PatchOp, key=lambda x: x.value))
    return (
        "## PatchSpec operations\n"
        f"Allowed ops: {ops}.\n"
        f"- `update_widget.fields` may only set: "
        f"{', '.join(sorted(f'`{f}`' for f in UPDATE_WIDGET_FIELDS))}.\n"
        f"- `update_dashboard.fields` may only set: "
        f"{', '.join(sorted(f'`{f}`' for f in UPDATE_DASHBOARD_FIELDS))}.\n"
        "- `reorder_widgets.order` is the new full ordering of existing widget ids; "
        "all ids in `order` must already exist in the dashboard.\n"
        "- Widget `id` is never patchable; rename = remove + add."
    )


def _authoritativeness_footer() -> str:
    return (
        "## How this schema is enforced\n"
        "The Python validator (`backend/app/services/spec_validator.py`) parses "
        "every DashboardSpec and PatchSpec against the Pydantic models above. "
        "If it rejects your output, the failure is authoritative — you must "
        "fix the shape, not argue. When your prior attempt was rejected, the "
        "next call's `_prior_errors` and `_prior_feedback_message` describe "
        "exactly which fields were wrong; treat them as the spec, not as "
        "hints."
    )


def render_widget_schema_markdown() -> str:
    """Return the full widget-toolkit schema doc as a single markdown
    string. Deterministic ordering so prompt caches hit across calls.
    """
    return "\n\n".join([
        _types_section(),
        _widget_fields_section(),
        _options_section(),
        _decision_flow_section(),
        _forbidden_section(),
        _patch_section(),
        _authoritativeness_footer(),
    ])


# ===========================================================================
# Envelope schemas (the JSON shapes Helper / Big-guy agents emit *around*
# DashboardSpec / PatchSpec). The widget schema above covers what the
# *content* must look like; these cover what the *envelope* must look like.
#
# Missing this docs caused the live UI failure: the big-guy rescue agent
# returned {"kind":"ticket","ticket":{title:..., component:..., description:...}}
# because the prompt said "<DeveloperTicket inner fields>" without ever
# listing what those fields were.
# ===========================================================================


def _developer_ticket_section() -> str:
    return (
        "## DeveloperTicket envelope (exact fields)\n"
        "Required:\n"
        "- `ticket_id` — 1..64 chars, `[a-zA-Z0-9_-]+`\n"
        "- `source_agent` — 1..64 chars, the agent name that filed it\n"
        "- `summary` — 1..256 chars, one-line description\n"
        "- `user_visible_effect` — 1..512 chars, what the user saw go wrong\n"
        "- `requested_action` — 1..1024 chars, what the dev should do\n"
        "Optional:\n"
        "- `severity` — `\"low\"` | `\"medium\"` | `\"high\"` (default `\"medium\"`)\n"
        "- `technical_evidence` — string OR dict (default `{}`); pasteable repro details\n"
        "- `safety_notes` — string, max 1024 chars (default `\"\"`)\n"
        "- `status` — `\"open\"` | `\"in_progress\"` | `\"resolved\"` | `\"rejected\"` (default `\"open\"`)\n"
        "- `created_at`, `resolved_at` — ISO timestamps, the orchestrator sets these\n"
        "\n"
        "**Forbidden alternative names** (do NOT use these; they will be rejected): "
        "`title`, `component`, `description`, `affected_files`, `suggested_fix`, "
        "`reproduce_steps`. The fields above are the ONLY accepted spelling."
    )


def _review_decision_section() -> str:
    return (
        "## ReviewDecision envelope (helper-review-agent output)\n"
        "Exactly one of these three shapes:\n"
        "```\n"
        "{\"type\":\"ReviewDecision\",\"decision\":\"approve\",\"rationale\":\"<1..1024 chars>\"}\n"
        "```\n"
        "```\n"
        "{\"type\":\"ReviewDecision\",\"decision\":\"patch\",\"rationale\":\"<short>\",\n"
        " \"patch\":{<full PatchSpec inner fields — see PatchSpec section above>}}\n"
        "```\n"
        "```\n"
        "{\"type\":\"ReviewDecision\",\"decision\":\"escalate\",\"rationale\":\"<why>\"}\n"
        "```\n"
        "No other top-level keys are accepted. `rationale` is REQUIRED in all cases."
    )


def _rescue_decision_section() -> str:
    return (
        "## RescueDecision envelope (big-guy rescue_review output)\n"
        "Exactly one of these four shapes. The inner `patch` / `ticket` MUST\n"
        "use the exact field names from the PatchSpec / DeveloperTicket schemas\n"
        "above — do not paraphrase them.\n"
        "```\n"
        "{\"type\":\"RescueDecision\",\"kind\":\"patch\",\"rationale\":\"<short>\",\n"
        " \"patch\":{<PatchSpec inner fields>}}\n"
        "```\n"
        "```\n"
        "{\"type\":\"RescueDecision\",\"kind\":\"ask_user\",\"rationale\":\"<short>\",\n"
        " \"questions\":[\"Q1\",\"Q2\"]}        // 1..8 questions\n"
        "```\n"
        "```\n"
        "{\"type\":\"RescueDecision\",\"kind\":\"ticket\",\"rationale\":\"<short>\",\n"
        " \"ticket\":{<DeveloperTicket inner fields — see section above>}}\n"
        "```\n"
        "```\n"
        "{\"type\":\"RescueDecision\",\"kind\":\"extend\",\"rationale\":\"<short>\",\n"
        " \"extend\":{\"widget_type\":\"pie_chart\",\n"
        "             \"rationale\":\"<why this widget is needed>\",\n"
        "             \"component_hint\":\"<optional implementer hint>\"}}\n"
        "```\n"
        "Use `kind=\"extend\"` ONLY when the dashboard failed because the user\n"
        "asked for a widget type that is not in the toolkit (e.g. pie_chart,\n"
        "heatmap, sankey). The runtime may then route the request to a\n"
        "separate `rescue_extend` operation that actually adds the widget.\n"
        "Rules for `extend.widget_type`: lowercase snake_case, 3-32 chars,\n"
        "must start with a letter. Never use `script`, `iframe`, `eval`,\n"
        "`exec`, `system`, `shell` or similar — those are denylisted.\n"
        "Common mistake: emitting `{\"ticket\":{\"title\":..., \"component\":..., "
        "\"description\":...}}`. Those keys do NOT exist on DeveloperTicket; the\n"
        "validator will reject the whole rescue output and the user will see a\n"
        "generic failure message. Use `summary`, `source_agent`, "
        "`user_visible_effect`, `requested_action`."
    )


def _bug_report_section() -> str:
    return (
        "## BugReport envelope (browser-eval-agent output, optional)\n"
        "Required:\n"
        "- `bug_id` — slug, `[a-zA-Z0-9_-]+`, max 64\n"
        "- `source` — short string, who filed it\n"
        "- `severity` — `\"low\"` | `\"medium\"` | `\"high\"`\n"
        "- `summary` — 1..256 chars\n"
        "Optional:\n"
        "- `evidence` — `{console_errors:[], missing_widgets:[], layout_errors:[], prometheus_errors:[], screenshot_path?}`\n"
        "- `suspected_cause` — string, max 512\n"
        "- `suggested_fix_type` — `\"patch\"` | `\"code\"` | `\"promql\"` | `\"data_source\"` | `\"unknown\"`"
    )


def _clarification_section() -> str:
    return (
        "## ClarificationRequest envelope\n"
        "Required:\n"
        "- `message_to_user` — 1..512 chars, user-facing intro\n"
        "- `questions` — list of 1..8 strings\n"
        "Optional:\n"
        "- `origin` — short string, default `\"rescue_review\"`"
    )


def _developer_report_section() -> str:
    return (
        "## DeveloperReport envelope (developer_fix output)\n"
        "Required:\n"
        "- `report_id` — slug, `[a-zA-Z0-9_-]+`, max 64\n"
        "- `summary` — 1..2048 chars, what you did and why\n"
        "Optional:\n"
        "- `ticket_id` — referenced ticket\n"
        "- `instruction` — the free-form instruction you acted on\n"
        "- `actions_taken` — list of strings, max 64\n"
        "- `tests_run` — list of strings, max 64\n"
        "- `status` — `\"resolved\"` | `\"in_progress\"` | `\"rejected\"` (default `\"resolved\"`)"
    )


def _prometheus_query_section() -> str:
    # PrometheusQueryReport has no Pydantic model — it's a free-form
    # report shape the runtime + heuristic produce. Document the
    # commonly-expected keys so the LLM doesn't drift.
    return (
        "## PrometheusQueryReport envelope (free-form, but expected shape)\n"
        "There is no strict Pydantic schema for this output yet, but downstream\n"
        "consumers expect these keys:\n"
        "- `summary` — string, what you found\n"
        "- `queries_checked` — list of `{promql, ok, reason?}` entries\n"
        "- `suggestions` — list of corrected/suggested PromQL strings\n"
        "Keep it small and concrete."
    )


# Map operation -> which envelope sections to inject. Keep this in
# sync with bin/opencode's _SCHEMA_AWARE_OPERATIONS dispatch. Only
# include envelopes the LLM might actually *emit* for that operation.
_OPERATION_ENVELOPES: dict[str, tuple[str, ...]] = {
    "user_message":        ("DeveloperTicket", "ClarificationRequest"),
    "generate_dashboard":  ("DeveloperTicket",),
    "patch_dashboard":     ("DeveloperTicket",),
    "review_rendered":     ("ReviewDecision",),  # patches go via PatchSpec already covered
    "rescue_review":       ("RescueDecision", "DeveloperTicket"),  # ticket is nested inside
    "evaluate_dashboard":  ("BugReport", "DeveloperTicket"),
    "prometheus_query":    ("PrometheusQueryReport", "DeveloperTicket"),
    "rescue_extend":       ("DeveloperReport", "DeveloperTicket"),
    "developer_fix":       ("DeveloperReport", "DeveloperTicket"),
}

_ENVELOPE_RENDERERS = {
    "DeveloperTicket": _developer_ticket_section,
    "ReviewDecision":  _review_decision_section,
    "RescueDecision":  _rescue_decision_section,
    "BugReport":       _bug_report_section,
    "ClarificationRequest": _clarification_section,
    "DeveloperReport": _developer_report_section,
    "PrometheusQueryReport": _prometheus_query_section,
}


def render_envelope_schemas_markdown(operation: str) -> str:
    """Return the envelope schema doc for the given operation, or an
    empty string if no envelopes are configured.

    The operation -> envelopes map is the single point that decides
    which schemas to inject; agent prompt files should not duplicate
    these definitions.
    """
    kinds = _OPERATION_ENVELOPES.get(operation, ())
    if not kinds:
        return ""
    sections = []
    for k in kinds:
        renderer = _ENVELOPE_RENDERERS.get(k)
        if renderer is None:  # pragma: no cover - defensive
            continue
        sections.append(renderer())
    if not sections:
        return ""
    return "\n\n".join(sections)


# Materialize once at import for speed and to surface any AttributeError
# (e.g. if someone removes a constant) at process start rather than at
# the first LLM call.
WIDGET_SCHEMA_MARKDOWN: str = render_widget_schema_markdown()
