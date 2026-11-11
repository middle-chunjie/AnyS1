"""Command-line interface for AnyS1 (`anys1`)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from anys1.compiler import SkillCompiler
from anys1.mcp_bridge import AnyS1MCPServer
from anys1.s1 import TAU_COLD_DEFAULT, CalibratedGate


def cmd_compile(args: argparse.Namespace) -> int:
    compiler = SkillCompiler(n_passes=args.passes)
    compiled = compiler.compile_file(
        skill_path=args.skill_md,
        output_dir=args.output,
        tau=args.tau,
        offline_heuristic=args.offline,
    )
    summary = compiled.to_dict()["counts"]
    print(
        f"[anys1] Compiled '{compiled.skill_name}' -> {Path(args.output).resolve()}\n"
        f"  Decision points: {summary['total']} total "
        f"(D={summary['D']}, E={summary['E']}, G={summary['G']})\n"
        f"  Default threshold: tau = {compiled.tau_default:.3f}\n"
        f"  Generated files:\n"
        f"    1. decisions.json  (structured D/E/G specification)\n"
        f"    2. SKILL.anys1.md  (drop-in dual-system skill for Claude Code / OpenHands / Cursor)\n"
        f"    3. workflow.py     (executable Python DualSystemRuntime workflow)"
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    gate, report = CalibratedGate.fit_from_jsonl(
        args.shadow,
        target_agreement=args.target_agr,
        min_coverage=args.min_cov,
    )
    out_data = asdict(report)
    print(json.dumps(out_data, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    srv = AnyS1MCPServer(
        compiled_dir=args.compiled,
        s1_url=args.s1_url,
        tau_override=args.tau,
    )
    res = srv.handle_decide(dp_id=args.dp_id, state=args.state)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    srv = AnyS1MCPServer(
        compiled_dir=args.compiled,
        s1_url=args.s1_url,
        tau_override=args.tau,
    )
    srv.serve_stdio()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anys1",
        description="AnyS1: Compile procedural SKILL.md files into calibrated dual-system workflows.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # anys1 compile
    p_comp = sub.add_parser(
        "compile",
        help="Compile a SKILL.md into dual-system artifacts (decisions.json, SKILL.anys1.md, workflow.py)",
    )
    p_comp.add_argument("skill_md", help="Path to input SKILL.md file")
    p_comp.add_argument("-o", "--output", required=True, help="Output directory for compiled skill")
    p_comp.add_argument(
        "--tau",
        type=float,
        default=TAU_COLD_DEFAULT,
        help=f"Initial deferral threshold tau (default: {TAU_COLD_DEFAULT} = AnyS1_cold)",
    )
    p_comp.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Number of grounded extraction passes for self-consistent merging (default: 3)",
    )
    p_comp.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic markdown structural parser without calling a System-2 API",
    )
    p_comp.set_defaults(func=cmd_compile)

    # anys1 calibrate
    p_cal = sub.add_parser(
        "calibrate",
        help="Fit warm threshold tau_warm from shadow execution logs (shadow.jsonl) for >=95% agreement",
    )
    p_cal.add_argument("--shadow", required=True, help="Path to shadow.jsonl log file")
    p_cal.add_argument(
        "--target-agr",
        type=float,
        default=0.95,
        help="Minimum required System-1 / System-2 agreement (default: 0.95)",
    )
    p_cal.add_argument(
        "--min-cov",
        type=float,
        default=0.10,
        help="Minimum required System-1 coverage (default: 0.10)",
    )
    p_cal.add_argument("-o", "--output", help="Optional path to save calibration report JSON")
    p_cal.set_defaults(func=cmd_calibrate)

    # anys1 decide
    p_dec = sub.add_parser(
        "decide",
        help="One-shot CLI tool call to evaluate a compiled decision point against AnyS1-4B",
    )
    p_dec.add_argument("--compiled", required=True, help="Directory containing compiled decisions.json")
    p_dec.add_argument("--dp-id", required=True, help="Decision point ID (e.g. dp_01)")
    p_dec.add_argument("--state", required=True, help="Execution state string")
    p_dec.add_argument("--s1-url", default=None, help="AnyS1-4B server URL (default: http://127.0.0.1:8895/decide)")
    p_dec.add_argument("--tau", type=float, default=None, help="Optional override for threshold tau")
    p_dec.set_defaults(func=cmd_decide)

    # anys1 mcp
    p_mcp = sub.add_parser(
        "mcp",
        help="Start stdio MCP tool server exposing `anys1_decide` for Claude Code / OpenHands / Cursor",
    )
    p_mcp.add_argument("--compiled", required=True, help="Directory containing compiled decisions.json")
    p_mcp.add_argument("--s1-url", default=None, help="AnyS1-4B server URL (default: http://127.0.0.1:8895/decide)")
    p_mcp.add_argument("--tau", type=float, default=None, help="Optional override for threshold tau")
    p_mcp.set_defaults(func=cmd_mcp)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
