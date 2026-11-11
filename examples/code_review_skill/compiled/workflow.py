"""Auto-generated AnyS1 Dual-System Workflow for skill: pr-security-and-test-review.

Generated from: /Users/ruili/Code/AnyS1/examples/code_review_skill/SKILL.md
Default cold-start threshold (tau_cold): 0.918
"""

from __future__ import annotations

from typing import Any, Callable
from anys1.runtime import DualSystemRuntime
from anys1.s1 import CalibratedGate

SKILL_NAME = 'pr-security-and-test-review'
TAU_DEFAULT = 0.918

DECISION_POINTS: list[dict[str, Any]] = [
    {
        "id": "dp_01",
        "type": "D",
        "section": "Step 1: Pre-Flight Diff Check",
        "anchor": "Check whether the pull request diff is empty or only touches markdown documentation files (`.md`).",
        "question": "In section 'Step 1: Pre-Flight Diff Check', how should the agent proceed at: \"Check whether the pull request diff is empty or only touches markdown documentation files (`.md`).\"?",
        "options": [
            "**Option A**: Skip test execution and approve documentation-only diff immediately.",
            "**Option B**: Proceed to security and test triage."
        ],
        "rule_hint": "bool(state_context.get('condition_met', False))"
    },
    {
        "id": "dp_02",
        "type": "E",
        "section": "Step 2: Security Risk Classification",
        "anchor": "Triage the modified files and diff hunks to classify the primary security risk surface:",
        "question": "In section 'Step 2: Security Risk Classification', how should the agent proceed at: \"Triage the modified files and diff hunks to classify the primary security risk surface:\"?",
        "options": [
            "**Option A**: Authentication, session tokens, or permission boundary changes (High Risk).",
            "**Option B**: Database schema, SQL query construction, or serialization changes (Medium Risk).",
            "**Option C**: Pure UI styling, frontend layout, or non-sensitive business logic (Low Risk)."
        ],
        "rule_hint": ""
    },
    {
        "id": "dp_03",
        "type": "E",
        "section": "Step 3: Verification Suite Selection",
        "anchor": "Select one of the verification suites based on the diff scope and risk surface:",
        "question": "In section 'Step 3: Verification Suite Selection', how should the agent proceed at: \"Select one of the verification suites based on the diff scope and risk surface:\"?",
        "options": [
            "**Option A**: Run full security + integration + unit test suite (`pytest tests/security tests/integration`).",
            "**Option B**: Run targeted unit tests for modified modules only (`pytest -k modified`).",
            "**Option C**: Run static linter and type-check only (`ruff check && mypy`)."
        ],
        "rule_hint": ""
    },
    {
        "id": "dp_04",
        "type": "G",
        "section": "Step 4: Final Review Synthesis",
        "anchor": "Write a comprehensive pull request review summary explaining the security findings, test outcomes, and concrete line-level recommendations for the author.",
        "question": "In section 'Step 4: Final Review Synthesis', how should the agent proceed at: \"Write a comprehensive pull request review summary explaining the security findings, test outcomes, and concrete line-lev\"?",
        "options": [],
        "rule_hint": ""
    }
]


class CompiledSkillWorkflow:
    """Executable dual-system workflow wrapper around ``pr-security-and-test-review``."""

    def __init__(
        self,
        runtime: DualSystemRuntime | None = None,
        tau: float = TAU_DEFAULT,
        mode: str = "anys1",
        shadow_log_path: str | None = None,
    ) -> None:
        self.runtime = runtime or DualSystemRuntime(
            gate=CalibratedGate(tau=tau, mode="cold"),
            mode=mode,
            shadow_log_path=shadow_log_path,
        )
        self.dps = {dp["id"]: dp for dp in DECISION_POINTS}

    def step(
        self,
        dp_id: str,
        state: str,
        *,
        predicate_fn: Callable[[], int] | None = None,
    ) -> str:
        """Execute a single compiled decision point (`D`, `E`, or `G`) by ID."""
        if dp_id not in self.dps:
            raise KeyError(f"Unknown decision point ID: {dp_id!r}")
        spec = self.dps[dp_id]
        dtype = spec["type"]

        if dtype == "D":
            fn = predicate_fn or (lambda: 0)
            return self.runtime.decide_deterministic(
                dp_id=dp_id,
                question=spec["question"],
                options=spec["options"],
                predicate_fn=fn,
            )
        if dtype == "E":
            return self.runtime.decide_enum(
                dp_id=dp_id,
                state=state,
                question=spec["question"],
                options=spec["options"],
            )
        return self.runtime.generate(
            dp_id=dp_id,
            prompt=f"### Current State\n{state}\n\n### Task\n{spec['question']}",
        )

    def run_all_enumerative(self, state: str) -> dict[str, str]:
        """Run all Type-E decision points against the provided execution state."""
        results: dict[str, str] = {}
        for dp in DECISION_POINTS:
            if dp["type"] == "E":
                results[dp["id"]] = self.step(dp["id"], state)
        return results
