"""Four-stage SkillDP offline synthesis pipeline (§3.2).

1. Grounded D/E/G decision-point extraction & self-consistent merging (`anys1.compiler.SkillCompiler`).
2. Skill-conditioned container task & workspace synthesis (`synthesize_container_tasks`).
3. Post-hoc trajectory decision-point state location & de-leaked prefix assembly (`locate_trajectory_states`).
4. Order-debiased 2x teacher consensus labeling & 3x choice-order permutation expansion (`consensus_label_state`, `expand_choice_permutations`).
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from anys1.compiler import DecisionPointSpec
from anys1.llm import System2Client


TASK_SYNTH_PROMPT = """You are generating realistic, executable sandbox tasks for a procedural agent skill.
Given the skill specification and its extracted Type-E decision points, synthesize {n_tasks} diverse user tasks and workspace setup scripts that naturally exercise different candidate options across the skill's decision points.

Return JSON:
{{
  "tasks": [
    {{
      "task_id": "t_01",
      "target_dp_ids": ["dp_01", "dp_02"],
      "user_prompt": "...",
      "setup_files": {{"path/to/file": "file contents"}}
    }}
  ]
}}"""


LOCATOR_PROMPT = """Given a completed agent execution trajectory (list of steps with tool calls and observations) and a list of Type-E decision points from the skill, identify the exact step index `t` right BEFORE the agent makes each decision point's choice (so that step `t` contains the observation prefix up to that moment, and step `t+1` contains the chosen action).

Return JSON:
{{
  "hits": [
    {{"dp_id": "dp_01", "step_index": 3, "rationale": "..."}}
  ]
}}"""


def synthesize_container_tasks(
    skill_markdown: str,
    decision_points: list[DecisionPointSpec],
    *,
    s2: System2Client | None = None,
    n_tasks: int = 3,
) -> list[dict[str, Any]]:
    """Synthesize concrete user tasks and initial workspace files conditioned on a skill."""
    client = s2 or System2Client()
    e_dps = [
        {"id": dp.id, "question": dp.question, "options": dp.options}
        for dp in decision_points
        if dp.type == "E"
    ]
    raw = client.complete(
        f"### SKILL.md\n{skill_markdown}\n\n### Enumerative Decision Points\n{json.dumps(e_dps, indent=2)}",
        system=TASK_SYNTH_PROMPT.format(n_tasks=n_tasks),
        role="synth_container_tasks",
        max_tokens=3072,
    )
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start : end + 1]).get("tasks", [])
        except Exception:
            pass
    return []


def locate_trajectory_states(
    trajectory_steps: list[dict[str, Any]],
    decision_points: list[DecisionPointSpec],
    *,
    s2: System2Client | None = None,
) -> list[dict[str, Any]]:
    """Locate decision points along an execution trajectory and extract de-leaked prefixes.

    Crucially truncates the trajectory strictly at ``step_index`` (before the agent's
    chosen action at ``step_index + 1``) so the resulting state contains zero action leakage.
    """
    client = s2 or System2Client()
    e_dps = {dp.id: dp for dp in decision_points if dp.type == "E"}
    compact_steps = [
        {
            "step_index": idx,
            "tool": st.get("tool", ""),
            "observation_preview": str(st.get("observation", ""))[:400],
        }
        for idx, st in enumerate(trajectory_steps)
    ]
    raw = client.complete(
        f"### Trajectory Steps\n{json.dumps(compact_steps, indent=2)}\n\n"
        f"### Decision Points\n{json.dumps([{'id': k, 'question': v.question, 'options': v.options} for k, v in e_dps.items()], indent=2)}",
        system=LOCATOR_PROMPT,
        role="locate_states",
        max_tokens=2048,
    )
    hits: list[dict[str, Any]] = []
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            hits = json.loads(raw[start : end + 1]).get("hits", [])
        except Exception:
            hits = []

    states: list[dict[str, Any]] = []
    for h in hits:
        dp_id = h.get("dp_id")
        step_idx = int(h.get("step_index", -1))
        if dp_id not in e_dps or not (0 <= step_idx < len(trajectory_steps)):
            continue
        dp = e_dps[dp_id]
        # Build de-leaked prefix state strictly up to step_idx (inclusive of observation, excluding next action)
        prefix_lines = []
        for s in trajectory_steps[: step_idx + 1]:
            if s.get("user_prompt"):
                prefix_lines.append(f"[User Task]\n{s['user_prompt']}")
            if s.get("action"):
                prefix_lines.append(f"[Action]\n{s['action']}")
            if s.get("observation"):
                prefix_lines.append(f"[Observation]\n{s['observation']}")
        states.append(
            {
                "dp_id": dp_id,
                "step_index": step_idx,
                "state": "\n\n".join(prefix_lines),
                "question": dp.question,
                "options": list(dp.options),
            }
        )
    return states


def consensus_label_state(
    state: str,
    question: str,
    options: list[str],
    *,
    s2: System2Client | None = None,
) -> int | None:
    """Label a state with 2x order-debiased teacher consensus (forward + reversed option order).

    Returns the consensus option index ``y* in {0, ..., K-1}`` if and only if the
    forward pass and reversed-option pass select the exact same underlying option;
    otherwise returns ``None`` (discarding position-biased or ambiguous states).
    """
    client = s2 or System2Client()
    fwd_idx, _ = client.decide_option(state, question, options, role="label_fwd")
    rev_options = list(reversed(options))
    rev_idx, _ = client.decide_option(state, question, rev_options, role="label_rev")
    mapped_rev_idx = len(options) - 1 - rev_idx
    if fwd_idx == mapped_rev_idx:
        return fwd_idx
    return None


def expand_choice_permutations(
    record: dict[str, Any],
    n_permutations: int = 3,
) -> list[dict[str, Any]]:
    """Expand a consensus-labeled state into ``n_permutations`` deterministic option orderings (`_x3`)."""
    state = record["state"]
    q0 = record["questions"][0]
    instr = q0["instr"]
    options = list(q0["options"])
    gold_opt = options[int(q0["label"])]
    K = len(options)

    seed_int = int.from_bytes(
        hashlib.sha256(f"{state[:256]}::{instr}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = random.Random(seed_int)

    expanded: list[dict[str, Any]] = []
    seen_orders: list[tuple[int, ...]] = []

    # Permutation 0: original order
    base_order = tuple(range(K))
    seen_orders.append(base_order)

    for _ in range(n_permutations * 4):
        if len(seen_orders) >= n_permutations:
            break
        perm = list(range(K))
        rng.shuffle(perm)
        t_perm = tuple(perm)
        if t_perm not in seen_orders:
            seen_orders.append(t_perm)

    while len(seen_orders) < n_permutations:
        seen_orders.append(seen_orders[-1])

    for p_idx, order in enumerate(seen_orders[:n_permutations]):
        perm_opts = [options[i] for i in order]
        new_label = perm_opts.index(gold_opt)
        expanded.append(
            {
                "state": state,
                "questions": [
                    {
                        "instr": instr,
                        "options": perm_opts,
                        "label": new_label,
                        "src": f"{q0.get('src', 'skilldp')}:p{p_idx}",
                    }
                ],
            }
        )
    return expanded
