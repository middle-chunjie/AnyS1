"""End-to-end runnable demo of compiling a SKILL.md into Dual-System form and calibrating tau."""

from __future__ import annotations

import json
from pathlib import Path

from anys1.compiler import SkillCompiler
from anys1.runtime import DualSystemRuntime
from anys1.s1 import CalibratedGate, S1Prediction


class MockSystem1Client:
    """Offline mock of AnyS1-4B pointer server for quick local verification without GPU."""

    def predict(self, state: str, question: str, options: list[str]) -> S1Prediction:
        if "auth" in state.lower() or "token" in state.lower():
            probs = [0.945, 0.035, 0.020][: len(options)]
        else:
            probs = [0.120, 0.840, 0.040][: len(options)]
        idx = int(max(range(len(probs)), key=probs.__getitem__))
        return S1Prediction(
            choice_idx=idx,
            choice_letter=chr(ord("A") + idx),
            choice_text=options[idx],
            p_max=probs[idx],
            probs=probs,
            latency_ms=14.2,
        )


class MockSystem2Client:
    """Offline mock of System-2 fallback client for quick local verification."""

    total_tokens: int = 0

    def decide_option(self, state: str, question: str, options: list[str], *, role: str = "s2") -> tuple[int, str]:
        self.total_tokens += 420
        idx = 0 if "auth" in state.lower() else min(1, len(options) - 1)
        return idx, options[idx]

    def complete(self, prompt: str, *, system: str | None = None, role: str = "s2_gen") -> str:
        self.total_tokens += 650
        return "## PR Review Summary\n- Risk Surface: Verified\n- Tests: Passed\n- Recommendation: Ready to merge."


def main() -> None:
    here = Path(__file__).parent
    skill_md = here / "SKILL.md"
    compiled_dir = here / "compiled"

    # 1. Compile SKILL.md -> decisions.json + SKILL.anys1.md + workflow.py
    compiler = SkillCompiler()
    compiled = compiler.compile_file(skill_md, compiled_dir, offline_heuristic=True)
    print(f"1. Compiled '{compiled.skill_name}' -> {compiled_dir}")
    for dp in compiled.decision_points:
        print(f"   - [{dp.id} | Type-{dp.type}] {dp.question[:75]} ({len(dp.options)} options)")

    # 2. Execute compiled workflow with DualSystemRuntime (AnyS1_cold: tau = 0.918)
    runtime = DualSystemRuntime(
        s1=MockSystem1Client(),  # type: ignore[arg-type]
        s2=MockSystem2Client(),  # type: ignore[arg-type]
        gate=CalibratedGate(tau=0.918, mode="cold"),
    )
    sample_state = "Modified files: auth/jwt_session.py (+42 -11). Added bearer token expiry check."

    for dp in compiled.decision_points:
        if dp.type == "D":
            choice = runtime.decide_deterministic(dp.id, dp.question, dp.options, lambda: 1)
        elif dp.type == "E":
            choice = runtime.decide_enum(dp.id, sample_state, dp.question, dp.options)
        else:
            choice = runtime.generate(dp.id, f"Synthesize review for state: {sample_state}")
        rec = runtime.history[-1]
        print(f"2. Executed {dp.id} (Type-{dp.type}) -> routed_to={rec.routed_to}, p_max={rec.p_max}, choice={choice[:55]!r}")

    print("3. Runtime Summary:", json.dumps(runtime.summary(), indent=2))


if __name__ == "__main__":
    main()
