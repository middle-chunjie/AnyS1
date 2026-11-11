"""Model Context Protocol (MCP) JSON-RPC bridge for compiled AnyS1 skills.

Allows open-ended coding agents (Claude Code, Cursor, OpenHands, Codex) reading
``SKILL.anys1.md`` to call ``anys1_decide(dp_id, state)`` directly as an MCP tool.
When ``p_max >= tau``, the tool returns the System-1 option immediately in ~15ms;
when ``p_max < tau``, it returns ``DEFER_TO_SYSTEM2`` with the candidate options
so the host agent performs deliberate System-2 reasoning only on hard states.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from anys1.s1 import TAU_COLD_DEFAULT, CalibratedGate, System1Client


class AnyS1MCPServer:
    """Minimal zero-dependency stdio JSON-RPC 2.0 MCP server for a compiled skill."""

    def __init__(
        self,
        compiled_dir: str | Path,
        s1_url: str | None = None,
        tau_override: float | None = None,
    ) -> None:
        self.compiled_dir = Path(compiled_dir)
        manifest_path = self.compiled_dir / "decisions.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing decisions.json in {self.compiled_dir}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.dps = {dp["id"]: dp for dp in self.manifest.get("decision_points", [])}
        tau = tau_override if tau_override is not None else float(self.manifest.get("tau_default", TAU_COLD_DEFAULT))
        self.gate = CalibratedGate(tau=tau, mode="warm" if tau_override is not None else "cold")
        self.s1 = System1Client(url=s1_url)

    def handle_decide(self, dp_id: str, state: str) -> dict[str, Any]:
        if dp_id not in self.dps:
            return {"error": f"Unknown decision point id: {dp_id}"}
        dp = self.dps[dp_id]
        if dp["type"] != "E":
            return {
                "dp_id": dp_id,
                "type": dp["type"],
                "status": "NON_ENUMERATIVE",
                "instruction": dp["question"],
            }
        pred = self.s1.predict(
            state=state,
            question=dp["question"],
            options=dp["options"],
        )
        if self.gate.should_accept(pred):
            return {
                "dp_id": dp_id,
                "status": "ACCEPTED_SYSTEM1",
                "chosen_letter": pred.choice_letter,
                "chosen_option": pred.choice_text,
                "p_max": round(pred.p_max, 4),
                "tau": self.gate.tau,
                "latency_ms": pred.latency_ms,
            }
        return {
            "dp_id": dp_id,
            "status": "DEFER_TO_SYSTEM2",
            "reason": f"p_max ({pred.p_max:.4f}) < tau ({self.gate.tau:.4f})",
            "question": dp["question"],
            "options": dp["options"],
            "s1_tentative_option": pred.choice_text,
            "p_max": round(pred.p_max, 4),
            "tau": self.gate.tau,
        }

    def serve_stdio(self) -> None:
        """Run the MCP JSON-RPC 2.0 loop over stdin/stdout."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception:
                continue
            req_id = req.get("id")
            method = req.get("method", "")
            params = req.get("params") or {}

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "anys1-mcp", "version": "0.1.0"},
                    },
                }
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "anys1_decide",
                                "description": (
                                    f"Query the calibrated AnyS1-4B System-1 pointer head (tau={self.gate.tau:.3f}) "
                                    f"for an enumerative decision point in skill '{self.manifest.get('skill_name')}'."
                                ),
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "dp_id": {
                                            "type": "string",
                                            "description": "Decision point ID (e.g. 'dp_01') from SKILL.anys1.md",
                                        },
                                        "state": {
                                            "type": "string",
                                            "description": "Current execution context / tool observations",
                                        },
                                    },
                                    "required": ["dp_id", "state"],
                                },
                            }
                        ]
                    },
                }
            elif method == "tools/call":
                args = params.get("arguments") or {}
                out = self.handle_decide(
                    dp_id=str(args.get("dp_id", "")),
                    state=str(args.get("state", "")),
                )
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(out, indent=2, ensure_ascii=False),
                            }
                        ]
                    },
                }
            else:
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}

            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
