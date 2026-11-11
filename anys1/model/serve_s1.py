"""HTTP pointer server for AnyS1-4B (`/decide` and `/v1/models`).

Loads `Qwen/Qwen3.5-4B-Base` + LoRA adapter (`adapter_model.safetensors`) +
`OptionInteractionHead` (`head.pt`) from a local directory or Hugging Face Hub
repo (`middle-chunjie/AnyS1-4B`) and serves single-pass pointer probabilities.
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from anys1.model.head import HEADS

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def _user_tokens(tok: Any, text: str) -> list[int]:
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


class AnyS1PointerEngine:
    """Loads an AnyS1-4B checkpoint (`adapter_model.safetensors` + `head.pt`) for inference."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str = "cuda",
        dtype: str = "bf16",
        max_state: int = 8192,
    ) -> None:
        self.ckpt_path = Path(checkpoint_dir)
        self.device = device
        self.max_state = max_state
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

        meta = torch.load(self.ckpt_path / "head.pt", map_location="cpu", weights_only=False)
        base_name = meta.get("base", "Qwen/Qwen3.5-4B-Base")
        base_rev = meta.get("base_revision")
        head_type = meta.get("head_type", "oc")
        head_dim = int(meta.get("head_dim", 256))

        self.tok = AutoTokenizer.from_pretrained(str(self.ckpt_path))
        base_lm = AutoModelForCausalLM.from_pretrained(
            base_name,
            revision=base_rev,
            torch_dtype=torch_dtype,
        ).model
        self.lm = PeftModel.from_pretrained(base_lm, str(self.ckpt_path)).to(device).eval()

        self.head = HEADS[head_type](self.lm.config.hidden_size, dp=head_dim)
        self.head.load_state_dict(meta["head"])
        self.head.temperature = float(meta.get("temperature", 1.0))
        self.head.to(device=device, dtype=torch_dtype).eval()

    @torch.inference_mode()
    def score_questions(
        self,
        state: str,
        questions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        s_toks = _user_tokens(self.tok, state)[: self.max_state - 1]
        s_ids = [self.tok.convert_tokens_to_ids(SPECIAL[0])] + s_toks
        q_id, o_id, c_id, d_id = (self.tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])

        results: list[dict[str, Any]] = []
        for q in questions:
            instr_text = q.get("instruction") or q.get("instr") or ""
            options = q["options"]
            instr_ids = [q_id] + _user_tokens(self.tok, instr_text)
            spans = [[o_id] + _user_tokens(self.tok, str(o)) + [c_id] for o in options]
            br = instr_ids + [t for sp in spans for t in sp] + [d_id]

            ends: list[int] = []
            cursor = len(s_ids) + len(instr_ids)
            for sp in spans:
                cursor += len(sp)
                ends.append(cursor - 1)
            decide_idx = len(s_ids) + len(br) - 1

            input_ids = torch.tensor([s_ids + br], dtype=torch.long, device=self.device)
            hidden = self.lm(input_ids=input_ids).last_hidden_state[0]
            h_decide = hidden[decide_idx]
            h_opts = hidden[ends]
            logits = self.head(h_decide, h_opts).float()
            probs = F.softmax(logits, dim=-1).cpu().tolist()
            best = int(max(range(len(probs)), key=probs.__getitem__))
            results.append(
                {
                    "choice_idx": best,
                    "choice": chr(ord("A") + best),
                    "p_max": round(float(probs[best]), 6),
                    "probs": [round(float(p), 6) for p in probs],
                }
            )
        return results


def serve_http(engine: AnyS1PointerEngine, host: str = "127.0.0.1", port: int = 8895) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/models":
                body = json.dumps({"data": [{"id": "AnyS1-4B", "object": "model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            if self.path not in ("/decide", "/v1/systemone"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            decisions = engine.score_questions(
                state=str(payload.get("state", "")),
                questions=list(payload.get("questions", [])),
            )
            body = json.dumps({"decisions": decisions}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[AnyS1-4B] Serving pointer head on http://{host}:{port}/decide")
    server.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description="Serve AnyS1-4B pointer head over HTTP.")
    p.add_argument("--checkpoint", required=True, help="Path to AnyS1-4B checkpoint dir (with head.pt + adapter_model.safetensors)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8895)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    engine = AnyS1PointerEngine(
        checkpoint_dir=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    serve_http(engine, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
