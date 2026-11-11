"""Training recipe for AnyS1-4B (`Qwen/Qwen3.5-4B-Base` + LoRA r=16 + OptionInteractionHead).

Implements the exact training objective described in §3.3 of the paper:
- Backbone: `Qwen/Qwen3.5-4B-Base` (with Gated DeltaNet + attention projections in LoRA r=16, alpha=32)
- Readout: `OptionInteractionHead` (`--head_type oc`, d_p=256, 4 heads, zero-init residuals)
- Loss: Label-smoothed cross-entropy (epsilon = 0.1) over 3x choice-order permutations (`SkillDP`)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from anys1.model.head import HEADS
from anys1.model.serve_s1 import SPECIAL, _user_tokens

LORA_TARGETS_HYBRID = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
]


def smoothed_ce_loss(logits: torch.Tensor, target: int, epsilon: float = 0.1) -> torch.Tensor:
    """Label-smoothed cross-entropy loss (§3.3):
    L_cal = -(1 - eps) log p(y|x) - (eps / K) sum_j log p(j|x).
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs[target]
    smooth = -log_probs.mean()
    return (1.0 - epsilon) * nll + epsilon * smooth


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    base_lm = AutoModelForCausalLM.from_pretrained(
        args.base,
        revision=args.base_revision,
        torch_dtype=dtype,
    ).model

    lora_cfg = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=args.lora,
        lora_alpha=2 * args.lora,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS_HYBRID,
    )
    lm = get_peft_model(base_lm, lora_cfg).to(args.device)
    if args.checkpointing:
        lm.gradient_checkpointing_enable()

    head = HEADS[args.head_type](lm.config.hidden_size, dp=args.head_dim).to(
        device=args.device, dtype=torch.float32
    )

    params = [p for p in lm.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    records: list[dict[str, Any]] = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    s0_id = tok.convert_tokens_to_ids(SPECIAL[0])

    lm.train()
    head.train()
    step = 0
    opt.zero_grad()

    for ep in range(args.epochs):
        random.shuffle(records)
        for idx, rec in enumerate(records):
            s_toks = _user_tokens(tok, rec["state"])[: args.max_state - 1]
            s_ids = [s0_id] + s_toks
            q = rec["questions"][0]
            instr_ids = [q_id] + _user_tokens(tok, q["instr"])
            spans = [[o_id] + _user_tokens(tok, str(o)) + [c_id] for o in q["options"]]
            br = instr_ids + [t for sp in spans for t in sp] + [d_id]

            ends: list[int] = []
            cursor = len(s_ids) + len(instr_ids)
            for sp in spans:
                cursor += len(sp)
                ends.append(cursor - 1)
            decide_idx = len(s_ids) + len(br) - 1

            input_ids = torch.tensor([s_ids + br], dtype=torch.long, device=args.device)
            hidden = lm(input_ids=input_ids).last_hidden_state[0].float()
            logits = head(hidden[decide_idx], hidden[ends])
            loss = smoothed_ce_loss(logits, int(q["label"]), epsilon=args.label_smoothing)
            (loss / (args.batch * args.accum)).backward()

            if (idx + 1) % (args.batch * args.accum) == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"[train_s1] ep={ep+1}/{args.epochs} step={step} loss={loss.item():.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    torch.save(
        {
            "base": args.base,
            "base_revision": args.base_revision,
            "lora": args.lora,
            "head_dim": args.head_dim,
            "head_type": args.head_type,
            "weights_dtype": args.dtype,
            "temperature": 1.0,
            "head": head.state_dict(),
        },
        out_dir / "head.pt",
    )
    print(f"[train_s1] Saved AnyS1 checkpoint to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train AnyS1-4B pointer head on SkillDP.")
    p.add_argument("--data", required=True, help="Path to SkillDP train_x3.jsonl")
    p.add_argument("--base", default="Qwen/Qwen3.5-4B-Base")
    p.add_argument("--base_revision", default="1001bb4d826a52d1f399e183466143f4da7b741b")
    p.add_argument("--head_type", default="oc", choices=["oc", "pointer"])
    p.add_argument("--head_dim", type=int, default=256)
    p.add_argument("--lora", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--max_state", type=int, default=7552)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--checkpointing", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="Output checkpoint directory")
    train(p.parse_args())


if __name__ == "__main__":
    main()
