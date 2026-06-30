"""Golden harness — replay fixed time-series through the evaluator core.

The harness owns the INJECTED CLOCK. It walks a list of synthetic
`Sample`s, derives the per-tick `DataStatus` (provenance + freshness) and
baseline coverage, feeds them to a single `AnomalyEvaluatorCore` instance,
and collects the produced `AlertEvent` stream.

Everything is deterministic: same fixture in, same event stream out. The
golden tests assert on that stream (which states appear, at what tick / ts,
at what severity) — a missed must-fire is the top-severity failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.prometheus.neonatal_sim import NeonatalSim, Sample, Scenario
from app.services.anomaly_core import AnomalyEvaluatorCore, DataStatus
from app.specs.alert_rule_spec import AlertRuleSpec
from app.specs.anomaly_evaluation_report import AlertEvent, AlertState


@dataclass
class ReplayResult:
    events: list[AlertEvent] = field(default_factory=list)
    states: list[AlertState] = field(default_factory=list)  # per-tick final state
    ts: list[float] = field(default_factory=list)  # per-tick injected clock

    # ------------------------------------------------------------------
    def first_event(self, state: AlertState) -> AlertEvent | None:
        for ev in self.events:
            if ev.state == state:
                return ev
        return None

    def has_state(self, state: AlertState) -> bool:
        return any(ev.state == state for ev in self.events)

    def fired(self) -> bool:
        return self.has_state(AlertState.firing)


def replay(
    rule: AlertRuleSpec,
    samples: list[Sample],
    *,
    staleness_budget_s: float = 60.0,
    history_coverage_s: float = 24 * 3600.0,
    baseline: float | None = None,
    promoted: bool = False,
) -> ReplayResult:
    """Replay `samples` through one evaluator core with an injected clock.

    `history_coverage_s` is how much contiguous history backs the baseline;
    pass a value < 24h to exercise INSUFFICIENT_BASELINE. `baseline` may be
    pinned; if None the sim's healthy baseline for the rule's metric is used.
    """
    core = AnomalyEvaluatorCore(
        rule, staleness_budget_s=staleness_budget_s, promoted=promoted
    )
    result = ReplayResult()
    sim_baseline = (
        baseline
        if baseline is not None
        else NeonatalSim().baseline(rule.metric)
    )

    for s in samples:
        # The injected clock advances with the INTENDED tick time. For the
        # `stale` scenario the sample's freshness ts (`s.ts`) is frozen but
        # `intended_ts` keeps moving, so the freshness gate trips.
        now = s.intended_ts if s.intended_ts is not None else s.ts
        status = DataStatus(
            reachable=s.reachable,
            returns_data=s.has_data,
            source=s.source,
            sample_ts=s.ts,
            history_coverage_s=history_coverage_s,
        )
        value = s.value if (s.reachable and s.has_data) else None
        report = core.evaluate(
            now=now,
            value=value,
            baseline=sim_baseline,
            status=status,
        )
        result.events.extend(report.events)
        result.states.append(report.state)
        result.ts.append(now)

    return result


def baseline_rule() -> AlertRuleSpec:
    """The canonical neonatal rSO2 rule used by the goldens."""
    return AlertRuleSpec.model_validate(
        {
            "id": "neo-rso2-left-desat",
            "description": "Neonatal left cerebral rSO2 desaturation",
            "metric": "rso2_left",
            "labels": {"patient": "neo-001"},
            "baseline": {"fn": "avg_over_time", "window": "24h"},
            "comparator": "<",
            "ratio": 0.80,
            "for": "5m",
            "severity": "critical",
            "source": "prometheus",
            "mode": "shadow",
        }
    )
