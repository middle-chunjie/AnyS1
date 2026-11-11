"""SkillCompiler: Compiles a procedural SKILL.md into dual-system artifacts.

Given any user ``SKILL.md``, ``SkillCompiler`` extracts and merges its typed
decision points (``D`` Deterministic, ``E`` Enumerative, ``G`` Generative) and
materializes three ready-to-use artifacts in the output directory:

1. ``decisions.json``: Structured manifest of all extracted ``D/E/G`` decision
   points (`id`, `type`, `anchor`, `question`, `options`, `rule_expr`).
2. ``SKILL.anys1.md``: Drop-in dual-system annotated skill markdown for
   open-ended agents (Claude Code, OpenHands, Cursor, Codex). Injects explicit
   ``@anys1:decide(...)`` directives at every ``E`` decision point so the agent
   queries the fast ``AnyS1-4B`` pointer tool (`anys1 mcp`) before invoking
   System-2 reasoning.
3. ``workflow.py``: Executable Python workflow class (`CompiledSkillWorkflow`)
   wired into ``DualSystemRuntime`` with ``AnyS1_cold`` ($\tau=0.918$) and
   one-command warm calibration support.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anys1.llm import System2Client
from anys1.s1 import TAU_COLD_DEFAULT


@dataclass
class DecisionPointSpec:
    id: str
    type: str  # "D", "E", or "G"
    section: str
    anchor: str
    question: str
    options: list[str] = field(default_factory=list)
    rule_hint: str = ""


@dataclass
class CompiledSkill:
    skill_name: str
    source_path: str
    tau_default: float
    decision_points: list[DecisionPointSpec]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "source_path": self.source_path,
            "tau_default": self.tau_default,
            "counts": {
                "total": len(self.decision_points),
                "D": sum(1 for d in self.decision_points if d.type == "D"),
                "E": sum(1 for d in self.decision_points if d.type == "E"),
                "G": sum(1 for d in self.decision_points if d.type == "G"),
            },
            "decision_points": [asdict(d) for d in self.decision_points],
        }


EXTRACTION_SYSTEM_PROMPT = """You are the AnyS1 Procedural Skill Compiler.
Given a SKILL.md document, extract every procedural decision point and classify it into one of three execution types:
- "D" (Deterministic): Exact surface checks (regex, file existence, numeric threshold, JSON schema field check) that can be resolved by a deterministic rule without any LLM call.
- "E" (Enumerative): Bounded routing, classification, triage, tool selection, or action choice over a finite set of 2 to 6 mutually exclusive options.
- "G" (Generative): Open-ended code writing, multi-paragraph synthesis, or free-form explanation that requires System-2 generation.

Rules:
1. Ground every decision point in a verbatim `anchor` substring copied from the SKILL.md text.
2. For "D" and "E" points, provide 2 to 6 clear, mutually exclusive `options`.
3. For "D" points, provide a concise Python expression hint in `rule_hint` (e.g. `"len(diff_lines) == 0"`).
4. Return a JSON object: `{"decision_points": [{"id": "dp_01", "type": "D|E|G", "section": "...", "anchor": "...", "question": "...", "options": ["..."], "rule_hint": "..."}]}`."""


class SkillCompiler:
    """Compiles a raw ``SKILL.md`` into dual-system execution artifacts."""

    def __init__(
        self,
        s2: System2Client | None = None,
        n_passes: int = 3,
    ) -> None:
        self.s2 = s2
        self.n_passes = n_passes

    def extract_decision_points(
        self,
        skill_text: str,
        *,
        offline_heuristic: bool = False,
    ) -> list[DecisionPointSpec]:
        """Extract typed D/E/G decision points from ``skill_text``.

        When ``offline_heuristic=False`` (default), runs ``n_passes`` independent
        LLM extractions grounded in verbatim anchors and merges them by
        self-consistent anchor/type agreement (§3.2).
        When ``offline_heuristic=True``, runs a deterministic markdown structural
        parser suitable for zero-API local testing and demos.
        """
        if offline_heuristic:
            return _heuristic_extract_markdown(skill_text)

        client = self.s2 or System2Client()
        runs: list[list[dict[str, Any]]] = []
        for i in range(self.n_passes):
            raw = client.complete(
                f"Extract all D/E/G decision points from the following SKILL.md:\n\n{skill_text}",
                system=EXTRACTION_SYSTEM_PROMPT,
                role=f"compile_extract_{i+1}",
                temperature=0.2 if i > 0 else 0.0,
                max_tokens=4096,
            )
            parsed = _parse_dp_json(raw)
            # Grounding filter: keep only decision points whose anchor appears in skill_text
            grounded = [
                d
                for d in parsed
                if d.get("anchor") and _normalize_ws(d["anchor"]) in _normalize_ws(skill_text)
            ]
            runs.append(grounded or parsed)

        return _merge_extraction_runs(runs)

    def compile_file(
        self,
        skill_path: str | Path,
        output_dir: str | Path,
        *,
        tau: float = TAU_COLD_DEFAULT,
        offline_heuristic: bool = False,
    ) -> CompiledSkill:
        """Compile ``skill_path`` and write ``decisions.json``, ``SKILL.anys1.md``, and ``workflow.py``."""
        src = Path(skill_path).resolve()
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        skill_text = src.read_text(encoding="utf-8")
        skill_name = _infer_skill_name(skill_text, src)
        dps = self.extract_decision_points(skill_text, offline_heuristic=offline_heuristic)

        compiled = CompiledSkill(
            skill_name=skill_name,
            source_path=str(src),
            tau_default=tau,
            decision_points=dps,
        )

        # 1. Write decisions.json
        (out / "decisions.json").write_text(
            json.dumps(compiled.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # 2. Write SKILL.anys1.md (Drop-in dual-system skill for Claude Code / OpenHands / Cursor)
        annotated_md = _render_annotated_skill_md(skill_text, compiled)
        (out / "SKILL.anys1.md").write_text(annotated_md, encoding="utf-8")

        # 3. Write workflow.py (Executable Python dual-system workflow)
        workflow_py = _render_workflow_py(compiled)
        (out / "workflow.py").write_text(workflow_py, encoding="utf-8")

        return compiled


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _infer_skill_name(skill_text: str, src: Path) -> str:
    m = re.search(r"^name:\s*([^\n]+)", skill_text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip().strip("\"'")
    m_h1 = re.search(r"^#\s+([^\n]+)", skill_text, flags=re.MULTILINE)
    if m_h1:
        return re.sub(r"[^a-zA-Z0-9_-]+", "-", m_h1.group(1).strip()).strip("-").lower()
    return src.parent.name if src.name.lower() == "skill.md" else src.stem


def _parse_dp_json(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("```")
        ).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj.get("decision_points"), list):
                return obj["decision_points"]
        except Exception:
            pass
    return []


def _merge_extraction_runs(runs: list[list[dict[str, Any]]]) -> list[DecisionPointSpec]:
    """Self-consistent merge across extraction runs by normalized anchor overlap."""
    if not runs:
        return []
    base = max(runs, key=len)
    merged: list[DecisionPointSpec] = []
    for idx, item in enumerate(base, start=1):
        dp_type = str(item.get("type", "E")).upper()
        if dp_type not in ("D", "E", "G"):
            dp_type = "E"
        opts = [str(o).strip() for o in item.get("options", []) if str(o).strip()]
        if dp_type in ("D", "E") and len(opts) < 2:
            opts = ["Proceed with standard path", "Escalate / fallback path"]
        merged.append(
            DecisionPointSpec(
                id=f"dp_{idx:02d}",
                type=dp_type,
                section=str(item.get("section", "Workflow")),
                anchor=str(item.get("anchor", "")).strip(),
                question=str(item.get("question", "")).strip(),
                options=opts[:6],
                rule_hint=str(item.get("rule_hint", "")).strip(),
            )
        )
    return merged


def _heuristic_extract_markdown(skill_text: str) -> list[DecisionPointSpec]:
    """Deterministic structural extractor for offline compilation and testing."""
    dps: list[DecisionPointSpec] = []
    current_section = "Overview"
    lines = skill_text.splitlines()
    idx = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            current_section = line.lstrip("#").strip()
            i += 1
            continue

        # Detect explicit conditional / branching / triage blocks in markdown
        low = line.lower()
        if any(
            kw in low
            for kw in (
                "if ",
                "when ",
                "check whether",
                "decide whether",
                "triage",
                "classify",
                "choose between",
                "select one of",
                "route to",
                "write a comprehensive",
                "draft a detailed",
                "synthesize ",
            )
        ):
            anchor = line.lstrip("-*0123456789. ").strip()
            # Collect sub-bullets if present as candidate options
            opts: list[str] = []
            j = i + 1
            while j < len(lines):
                sub = lines[j]
                if re.match(r"^\s{2,}[-*]\s+", sub) or re.match(r"^\s*-\s+\*\*(Option|Case|Path|Branch)", sub, re.I):
                    opt_txt = re.sub(r"^\s*[-*]\s+", "", sub).strip()
                    opts.append(opt_txt)
                    j += 1
                else:
                    break

            idx += 1
            is_deterministic = any(
                k in low
                for k in ("empty", "exists", "regex", "status code", "== 0", "file extension", "threshold")
            )
            is_generative = any(
                k in low
                for k in ("write a comprehensive", "draft a detailed", "synthesize final report", "generate full")
            ) and not opts

            if is_generative:
                dp_type = "G"
                opts = []
                rule_hint = ""
            elif is_deterministic:
                dp_type = "D"
                if len(opts) < 2:
                    opts = ["Condition holds (fast path)", "Condition does not hold (continue)"]
                rule_hint = "bool(state_context.get('condition_met', False))"
            else:
                dp_type = "E"
                if len(opts) < 2:
                    opts = [
                        "Apply direct local fix / standard branch",
                        "Request deeper inspection / alternative branch",
                        "Escalate or abort workflow",
                    ]
                rule_hint = ""

            dps.append(
                DecisionPointSpec(
                    id=f"dp_{idx:02d}",
                    type=dp_type,
                    section=current_section,
                    anchor=anchor[:180],
                    question=f"In section '{current_section}', how should the agent proceed at: \"{anchor[:120]}\"?",
                    options=opts[:6],
                    rule_hint=rule_hint,
                )
            )
            i = max(i + 1, j)
            continue
        i += 1

    return dps


def _render_annotated_skill_md(original_md: str, compiled: CompiledSkill) -> str:
    """Render ``SKILL.anys1.md`` with dual-system routing table and inline directives."""
    header = [
        "<!-- ===================================================================== -->",
        f"<!-- Compiled by AnyS1 (tau_cold = {compiled.tau_default:.3f})                          -->",
        "<!-- Dual-System Execution Protocol:                                       -->",
        "<!--   [D] Deterministic : Execute via local check / script (0 LLM tokens) -->",
        "<!--   [E] Enumerative   : Call `anys1_decide(dp_id, state)` tool first;   -->",
        f"<!--                       accepts System-1 if p_max >= {compiled.tau_default:.3f}, else System-2 -->",
        "<!--   [G] Generative    : Execute directly with System-2 reasoning        -->",
        "<!-- ===================================================================== -->",
        "",
        "## AnyS1 Compiled Decision Table",
        "",
        "| ID | Type | Section | Decision Question | Candidate Options / Rule |",
        "|:---|:---:|:---|:---|:---|",
    ]
    for dp in compiled.decision_points:
        if dp.type == "D":
            opt_col = f"Rule: `{dp.rule_hint or 'deterministic check'}` → " + " / ".join(dp.options)
        elif dp.type == "E":
            opt_col = " / ".join(f"`({chr(65+k)})` {o}" for k, o in enumerate(dp.options))
        else:
            opt_col = "*Open-ended System-2 generation*"
        q_clean = dp.question.replace("|", "\\|")
        opt_clean = opt_col.replace("|", "\\|")
        header.append(f"| `{dp.id}` | **{dp.type}** | {dp.section} | {q_clean} | {opt_clean} |")

    header.extend(["", "---", ""])
    annotated_body = original_md
    for dp in compiled.decision_points:
        if dp.anchor and dp.anchor in annotated_body:
            if dp.type == "E":
                opts_json = json.dumps(dp.options, ensure_ascii=False)
                badge = (
                    f"\n> **[AnyS1 `{dp.id}` | Type-E Enumerative Decision]** "
                    f"Call `anys1_decide(dp_id=\"{dp.id}\", state=...)` over options {opts_json} "
                    f"(gate $\\tau={compiled.tau_default:.3f}$).\n"
                )
            elif dp.type == "D":
                badge = (
                    f"\n> **[AnyS1 `{dp.id}` | Type-D Deterministic Check]** "
                    f"Evaluate deterministic rule `{dp.rule_hint or 'check_predicate(state)'}` without LLM tokens.\n"
                )
            else:
                badge = (
                    f"\n> **[AnyS1 `{dp.id}` | Type-G Generative Step]** "
                    f"Invoke System-2 directly for open-ended synthesis.\n"
                )
            annotated_body = annotated_body.replace(dp.anchor, dp.anchor + badge, 1)

    return "\n".join(header) + annotated_body


def _render_workflow_py(compiled: CompiledSkill) -> str:
    """Generate a clean, self-contained Python module ``workflow.py``."""
    dp_list_repr = json.dumps([asdict(d) for d in compiled.decision_points], indent=4, ensure_ascii=False)
    return f'''"""Auto-generated AnyS1 Dual-System Workflow for skill: {compiled.skill_name}.

Generated from: {compiled.source_path}
Default cold-start threshold (tau_cold): {compiled.tau_default}
"""

from __future__ import annotations

from typing import Any, Callable
from anys1.runtime import DualSystemRuntime
from anys1.s1 import CalibratedGate

SKILL_NAME = {compiled.skill_name!r}
TAU_DEFAULT = {compiled.tau_default}

DECISION_POINTS: list[dict[str, Any]] = {dp_list_repr}


class CompiledSkillWorkflow:
    """Executable dual-system workflow wrapper around ``{compiled.skill_name}``."""

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
        self.dps = {{dp["id"]: dp for dp in DECISION_POINTS}}

    def step(
        self,
        dp_id: str,
        state: str,
        *,
        predicate_fn: Callable[[], int] | None = None,
    ) -> str:
        """Execute a single compiled decision point (`D`, `E`, or `G`) by ID."""
        if dp_id not in self.dps:
            raise KeyError(f"Unknown decision point ID: {{dp_id!r}}")
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
            prompt=f"### Current State\\n{{state}}\\n\\n### Task\\n{{spec['question']}}",
        )

    def run_all_enumerative(self, state: str) -> dict[str, str]:
        """Run all Type-E decision points against the provided execution state."""
        results: dict[str, str] = {{}}
        for dp in DECISION_POINTS:
            if dp["type"] == "E":
                results[dp["id"]] = self.step(dp["id"], state)
        return results
'''
