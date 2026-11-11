"""Dual-System Runtime for executing compiled D/E/G skills."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from anys1.llm import System2Client
from anys1.s1 import TAU_COLD_DEFAULT, CalibratedGate, S1Prediction, System1Client


@dataclass
class DecisionRecord:
    dp_id: str
    dp_type: str  # "D", "E", or "G"
    question: str
    options: list[str]
    chosen_idx: int
    chosen_option: str
    routed_to: str  # "rule", "s1", "s2_defer", "s2_only", "shadow"
    p_max: float | None = None
    tau: float | None = None
    s1_choice: int | None = None
    s2_choice: int | None = None
    agree: bool | None = None
    latency_ms: float = 0.0


class DualSystemRuntime:
    """Executes compiled procedural skills with calibrated System-1 / System-2 routing.

    - ``D`` (Deterministic) nodes execute zero-token Python predicates.
    - ``E`` (Enumerative) nodes query ``System1Client`` (`AnyS1-4B` pointer head):
      * If ``p_max >= gate.tau``, System-1's choice is accepted immediately.
      * If ``p_max < gate.tau``, the runtime defers to ``System2Client``.
      * If ``shadow_log_path`` is set (shadow calibration mode), both System-1
        and System-2 are queried and logged so ``CalibratedGate.fit_from_jsonl``
        can fit ``tau_warm`` for ``>= 95%`` agreement.
    - ``G`` (Generative) nodes invoke ``System2Client`` directly.
    """

    def __init__(
        self,
        s1: System1Client | None = None,
        s2: System2Client | None = None,
        gate: CalibratedGate | None = None,
        *,
        mode: str = "anys1",  # "anys1" | "all_s1" | "all_s2" | "shadow"
        shadow_log_path: str | Path | None = None,
    ) -> None:
        self.s1 = s1 or System1Client()
        self.s2 = s2 or System2Client()
        self.gate = gate or CalibratedGate(tau=TAU_COLD_DEFAULT, mode="cold")
        self.mode = mode
        self.shadow_log_path = Path(shadow_log_path) if shadow_log_path else None
        self.history: list[DecisionRecord] = []

    def decide_deterministic(
        self,
        dp_id: str,
        question: str,
        options: list[str],
        predicate_fn: Callable[[], int],
    ) -> str:
        """Execute a Type-D deterministic rule with zero LLM tokens."""
        t0 = time.perf_counter()
        idx = int(predicate_fn())
        dt_ms = (time.perf_counter() - t0) * 1000.0
        chosen = options[idx]
        self.history.append(
            DecisionRecord(
                dp_id=dp_id,
                dp_type="D",
                question=question,
                options=list(options),
                chosen_idx=idx,
                chosen_option=chosen,
                routed_to="rule",
                latency_ms=round(dt_ms, 3),
            )
        )
        return chosen

    def decide_enum(
        self,
        dp_id: str,
        state: str,
        question: str,
        options: list[str],
    ) -> str:
        """Execute a Type-E enumerative decision point via calibrated dual-system routing."""
        t0 = time.perf_counter()

        if self.mode == "all_s2":
            idx, chosen = self.s2.decide_option(state, question, options, role=f"s2:{dp_id}")
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.history.append(
                DecisionRecord(
                    dp_id=dp_id,
                    dp_type="E",
                    question=question,
                    options=list(options),
                    chosen_idx=idx,
                    chosen_option=chosen,
                    routed_to="s2_only",
                    s2_choice=idx,
                    latency_ms=round(dt_ms, 2),
                )
            )
            return chosen

        pred: S1Prediction = self.s1.predict(state=state, question=question, options=options)

        if self.mode == "shadow":
            s2_idx, s2_chosen = self.s2.decide_option(state, question, options, role=f"s2_shadow:{dp_id}")
            dt_ms = (time.perf_counter() - t0) * 1000.0
            agree = pred.choice_idx == s2_idx
            rec = DecisionRecord(
                dp_id=dp_id,
                dp_type="E",
                question=question,
                options=list(options),
                chosen_idx=s2_idx,
                chosen_option=s2_chosen,
                routed_to="shadow",
                p_max=pred.p_max,
                tau=self.gate.tau,
                s1_choice=pred.choice_idx,
                s2_choice=s2_idx,
                agree=agree,
                latency_ms=round(dt_ms, 2),
            )
            self.history.append(rec)
            self._append_shadow_log(rec)
            return s2_chosen

        if self.mode == "all_s1" or self.gate.should_accept(pred):
            dt_ms = (time.perf_counter() - t0) * 1000.0
            rec = DecisionRecord(
                dp_id=dp_id,
                dp_type="E",
                question=question,
                options=list(options),
                chosen_idx=pred.choice_idx,
                chosen_option=pred.choice_text,
                routed_to="s1",
                p_max=pred.p_max,
                tau=self.gate.tau,
                s1_choice=pred.choice_idx,
                latency_ms=round(dt_ms, 2),
            )
            self.history.append(rec)
            return pred.choice_text

        # p_max < tau -> defer to System-2
        s2_idx, s2_chosen = self.s2.decide_option(state, question, options, role=f"s2_defer:{dp_id}")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        rec = DecisionRecord(
            dp_id=dp_id,
            dp_type="E",
            question=question,
            options=list(options),
            chosen_idx=s2_idx,
            chosen_option=s2_chosen,
            routed_to="s2_defer",
            p_max=pred.p_max,
            tau=self.gate.tau,
            s1_choice=pred.choice_idx,
            s2_choice=s2_idx,
            agree=(pred.choice_idx == s2_idx),
            latency_ms=round(dt_ms, 2),
        )
        self.history.append(rec)
        return s2_chosen

    def generate(
        self,
        dp_id: str,
        prompt: str,
        *,
        system: str | None = None,
    ) -> str:
        """Execute a Type-G open-ended generative step on System-2."""
        t0 = time.perf_counter()
        out = self.s2.complete(prompt, system=system, role=f"s2_gen:{dp_id}")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.history.append(
            DecisionRecord(
                dp_id=dp_id,
                dp_type="G",
                question=prompt[:120],
                options=[],
                chosen_idx=0,
                chosen_option=out[:120],
                routed_to="s2_only",
                latency_ms=round(dt_ms, 2),
            )
        )
        return out

    def summary(self) -> dict[str, Any]:
        e_nodes = [r for r in self.history if r.dp_type == "E"]
        s1_accepted = [r for r in e_nodes if r.routed_to == "s1"]
        return {
            "total_decisions": len(self.history),
            "by_type": {
                t: sum(1 for r in self.history if r.dp_type == t)
                for t in ("D", "E", "G")
            },
            "e_total": len(e_nodes),
            "e_s1_accepted": len(s1_accepted),
            "e_s1_coverage": round(len(s1_accepted) / len(e_nodes), 4) if e_nodes else 0.0,
            "tau": self.gate.tau,
            "gate_mode": self.gate.mode,
            "s2_total_tokens": self.s2.total_tokens,
        }

    def _append_shadow_log(self, rec: DecisionRecord) -> None:
        if not self.shadow_log_path:
            return
        self.shadow_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.shadow_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
