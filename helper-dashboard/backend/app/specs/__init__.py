"""Pydantic schemas for Helper Dashboard specs and reports.

These are the contract between Helper agents, backend validation, and
the frontend renderer. Helper agents emit JSON that must parse into
these models. The frontend only ever renders data shaped like these
models. Nothing else is trusted.
"""

from .widget_spec import (
    WidgetSpec,
    WidgetType,
    QuerySpec,
    QueryType,
    QuerySource,
    WidgetPosition,
    WidgetEncoding,
    WidgetThreshold,
)
from .dashboard_spec import DashboardSpec, DashboardLayout, DashboardVariable
from .patch_spec import PatchSpec, PatchOperation
from .bug_report import BugReport, BugSeverity, SuggestedFixType, BugEvidence
from .developer_ticket import DeveloperTicket
from .developer_report import DeveloperReport
from .evaluation_report import BrowserEvaluationReport
from .review_decision import ExtendRequest, ReviewDecision, RescueDecision
from .clarification_request import ClarificationRequest
from .saved_dashboard import SavedDashboard, SavedDashboardSummary

__all__ = [
    "WidgetSpec",
    "WidgetType",
    "QuerySpec",
    "QueryType",
    "QuerySource",
    "WidgetPosition",
    "WidgetEncoding",
    "WidgetThreshold",
    "DashboardSpec",
    "DashboardLayout",
    "DashboardVariable",
    "PatchSpec",
    "PatchOperation",
    "BugReport",
    "BugSeverity",
    "SuggestedFixType",
    "BugEvidence",
    "DeveloperTicket",
    "DeveloperReport",
    "BrowserEvaluationReport",
    "ReviewDecision",
    "RescueDecision",
    "ExtendRequest",
    "ClarificationRequest",
    "SavedDashboard",
    "SavedDashboardSummary",
]
