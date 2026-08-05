from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Intent = Literal["create_dashboard", "patch_dashboard"]


@dataclass(frozen=True)
class IntentSummary:
    action: Intent
    domain: str
    summary: str


class IntentAgent:
    """Extracts high-level user intent before task-model generation."""

    def classify(self, prompt: str, *, has_current_dashboard: bool = False) -> Intent:
        lowered = prompt.lower()
        patch_keywords = ("add", "remove", "delete", "update", "change", "新增", "刪除", "删除", "修改", "改變")
        if has_current_dashboard or any(keyword in lowered for keyword in patch_keywords):
            return "patch_dashboard"
        return "create_dashboard"

    def extract(self, prompt: str, *, has_current_dashboard: bool = False) -> IntentSummary:
        lowered = prompt.lower()
        domain = "medical_monitoring" if any(
            keyword in lowered for keyword in ("patient", "medical", "icu", "hospital", "病人", "患者", "醫療", "医疗")
        ) else "monitoring"
        summary = " ".join(prompt.split())[:200] or "monitoring dashboard request"
        return IntentSummary(
            action=self.classify(prompt, has_current_dashboard=has_current_dashboard),
            domain=domain,
            summary=summary,
        )
