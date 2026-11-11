"""System-1 pointer client and calibrated selective deferral gate (tau_cold / tau_warm)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Held-out SkillDP calibration threshold achieving >= 95% selective agreement
# across unseen skills without any target-workflow shadow runs (§3.3).
TAU_COLD_DEFAULT: float = 0.918


@dataclass
class S1Prediction:
    choice_idx: int
    choice_letter: str
    choice_text: str
    p_max: float
    probs: list[float]
    latency_ms: float


class System1Client:
    """HTTP client for the AnyS1-4B pointer server (`anys1.model.serve_s1`)."""

    def __init__(
        self,
        url: str | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.url = (
            url
            or os.environ.get("ANYS1_S1_URL")
            or "http://127.0.0.1:8895/decide"
        )
        self.timeout_s = timeout_s
        self._http = httpx.Client(timeout=timeout_s)

    def predict(
        self,
        state: str,
        question: str,
        options: list[str],
    ) -> S1Prediction:
        """Score a single enumerative decision point in one forward pass."""
        if not 2 <= len(options) <= 26:
            raise ValueError(f"Expected 2..26 options, got {len(options)}")
        t0 = time.perf_counter()
        resp = self._http.post(
            self.url,
            json={
                "state": state,
                "questions": [
                    {
                        "instruction": question,
                        "options": list(options),
                    }
                ],
            },
        )
        resp.raise_for_status()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        payload = resp.json()
        d0 = payload["decisions"][0]
        probs = [float(p) for p in d0["probs"]]
        idx = int(max(range(len(probs)), key=probs.__getitem__))
        letter = chr(ord("A") + idx)
        return S1Prediction(
            choice_idx=idx,
            choice_letter=letter,
            choice_text=options[idx],
            p_max=float(probs[idx]),
            probs=probs,
            latency_ms=round(dt_ms, 2),
        )


@dataclass
class CalibrationReport:
    tau: float
    mode: str
    target_agreement: float
    empirical_agreement: float
    coverage: float
    n_total: int
    n_accepted: int


class CalibratedGate:
    """Confidence-gated selective prediction rule:

    - Accepts System-1's option iff ``p_max >= tau``.
    - Defers to System-2 iff ``p_max < tau``.

    Supports two operating modes matching §3.3 of the paper:
    1. ``cold`` (``AnyS1_cold``): uses ``tau_cold = 0.918`` transferred directly
       from held-out SkillDP states with zero target-workflow runs.
    2. ``warm`` (``AnyS1_warm``): fits ``tau`` on development shadow records to
       maximize System-1 coverage subject to ``Agreement(tau) >= 0.95``.
    """

    def __init__(
        self,
        tau: float = TAU_COLD_DEFAULT,
        mode: str = "cold",
    ) -> None:
        self.tau = float(tau)
        self.mode = mode

    def should_accept(self, prediction: S1Prediction) -> bool:
        return prediction.p_max >= self.tau

    @classmethod
    def fit_from_shadow(
        cls,
        shadow_records: list[dict[str, Any]],
        *,
        target_agreement: float = 0.95,
        min_coverage: float = 0.10,
        fallback_tau: float = TAU_COLD_DEFAULT,
    ) -> tuple[CalibratedGate, CalibrationReport]:
        """Fit ``tau_warm`` from shadow execution logs where both S1 and S2 were recorded.

        Selects the smallest threshold ``tau`` on a 0.001 grid such that empirical
        agreement ``Pr[y_hat_S1 == y_S2 | p_max >= tau] >= target_agreement`` with
        coverage ``Pr[p_max >= tau] >= min_coverage``.
        """
        valid = [
            r
            for r in shadow_records
            if "p_max" in r and ("agree" in r or ("s1_choice" in r and "s2_choice" in r))
        ]
        n_total = len(valid)
        if n_total == 0:
            report = CalibrationReport(
                tau=fallback_tau,
                mode="cold_fallback",
                target_agreement=target_agreement,
                empirical_agreement=0.0,
                coverage=0.0,
                n_total=0,
                n_accepted=0,
            )
            return cls(tau=fallback_tau, mode="cold_fallback"), report

        candidates = sorted({round(float(r["p_max"]), 4) for r in valid})
        best_tau: float | None = None
        best_agr = 0.0
        best_cov = 0.0
        best_acc = 0

        for tau in candidates:
            accepted = [r for r in valid if float(r["p_max"]) >= tau]
            n_acc = len(accepted)
            cov = n_acc / n_total
            if cov < min_coverage:
                continue
            n_agree = sum(
                1
                for r in accepted
                if r.get("agree", r.get("s1_choice") == r.get("s2_choice"))
            )
            agr = n_agree / n_acc if n_acc else 0.0
            if agr >= target_agreement:
                best_tau = tau
                best_agr = agr
                best_cov = cov
                best_acc = n_acc
                break

        if best_tau is None:
            # If no threshold satisfies both constraints, fall back to tau_cold
            accepted = [r for r in valid if float(r["p_max"]) >= fallback_tau]
            n_acc = len(accepted)
            n_agree = sum(
                1
                for r in accepted
                if r.get("agree", r.get("s1_choice") == r.get("s2_choice"))
            )
            report = CalibrationReport(
                tau=fallback_tau,
                mode="cold_fallback",
                target_agreement=target_agreement,
                empirical_agreement=round(n_agree / n_acc, 4) if n_acc else 0.0,
                coverage=round(n_acc / n_total, 4),
                n_total=n_total,
                n_accepted=n_acc,
            )
            return cls(tau=fallback_tau, mode="cold_fallback"), report

        report = CalibrationReport(
            tau=best_tau,
            mode="warm",
            target_agreement=target_agreement,
            empirical_agreement=round(best_agr, 4),
            coverage=round(best_cov, 4),
            n_total=n_total,
            n_accepted=best_acc,
        )
        return cls(tau=best_tau, mode="warm"), report

    @classmethod
    def fit_from_jsonl(
        cls,
        shadow_jsonl_path: str | Path,
        *,
        target_agreement: float = 0.95,
        min_coverage: float = 0.10,
    ) -> tuple[CalibratedGate, CalibrationReport]:
        rows: list[dict[str, Any]] = []
        with open(shadow_jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return cls.fit_from_shadow(
            rows,
            target_agreement=target_agreement,
            min_coverage=min_coverage,
        )
