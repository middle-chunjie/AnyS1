# AnyS1: Compiling Procedural Agent Skills into Calibrated Dual-System Workflows

<p align="center">
  <a href="https://middle-chunjie.github.io/AnyS1/"><b>🌐 Project Page</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/datasets/middle-chunjie/SkillDP"><b>📦 SkillDP Dataset (HF)</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/middle-chunjie/AnyS1-4B"><b>🤗 AnyS1-4B Model (HF)</b></a>
</p>

**AnyS1** is an open-source framework and compiler that transforms natural-language procedural agent skills (`SKILL.md` files and standard operating procedures) into **calibrated dual-system (`System-1` + `System-2`) workflows**.

Instead of invoking a heavy frontier `System-2` model at every routine check and routing branch, **AnyS1** decomposes a skill into three decision types (shares over the 17,865 decision points mined from 1,072 public skills):
- **`D` (Deterministic — 21.3%)**: Surface checks (regex, file existence, numeric thresholds) compiled into zero-token Python predicates.
- **`E` (Enumerable — 64.5%)**: Bounded routing, triage, tool selection, and action choice over a closed option set, delegated to **`AnyS1-4B`** — a 4B single-forward-pass pointer model (`Qwen3.5-4B-Base` + LoRA + permutation-equivariant `OptionInteractionHead`) guarded by a calibrated confidence gate $p_{\max}(x) \ge \tau$ that falls back to `System-2` when uncertain.
- **`G` (Generative — 14.2%)**: Open-ended code writing and free-form synthesis routed directly to `System-2`.

---

## ✨ Key Results

On three benchmarks unseen during training (**SOPBench**, **ALFWorld**, **WebShop**), each compiled into a D/E/G workflow, compared with the same workflow where `System-2` answers every decision (`All-S2`):

| System-2 | Configuration | SOPBench pass | ALFWorld SR | WebShop score | Avg. Δ vs All-S2 | Avg. cost saving |
|---|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | All-S2 | 83.2 | 96.3 | 66.9 | – | – |
| | `AnyS1_warm` | 84.3 | 95.5 | 68.1 | +0.5 | 38% |
| | `AnyS1_cold` | 83.2 | 95.5 | 67.9 | +0.1 | 29% |
| GPT-6 Sol | All-S2 | 82.1 | 96.3 | 70.6 | – | – |
| | `AnyS1_warm` | 82.5 | 95.5 | 70.4 | −0.2 | 36% |
| | `AnyS1_cold` | 82.5 | 94.0 | 70.1 | −0.8 | 31% |

- `AnyS1_warm` fits $\tau$ on 50 unlabeled development runs of the target workflow; `AnyS1_cold` transfers $\tau_{\text{cold}} = 0.918$ with **zero** target-workflow runs. The same `AnyS1-4B` checkpoint is used with both `System-2` backbones, without retraining.
- `AnyS1_warm` cuts episode cost by 30–44% and critical-path model time by 38–48% across benchmarks and backbones.
- `AnyS1-4B` answers one decision in 22–32 ms (median, one H100 GPU).

---

## 🚀 Quickstart: Compile Any `SKILL.md` into Dual-System Form

### 1. Installation

```bash
git clone https://github.com/middle-chunjie/AnyS1.git
cd AnyS1
pip install -e .
# Optional: install PyTorch / Transformers / PEFT for local AnyS1-4B serving & training
pip install -e ".[train]"
```

### 2. Serve `AnyS1-4B` Locally

Download the **`AnyS1-4B`** checkpoint (`adapter_model.safetensors` + `head.pt`) from Hugging Face and launch the lightweight HTTP pointer server:

```bash
python -m anys1.model.serve_s1 \
  --checkpoint middle-chunjie/AnyS1-4B \
  --host 127.0.0.1 \
  --port 8895 \
  --device cuda \
  --dtype bf16
```

### 3. Compile Your Own `SKILL.md`

Point `anys1 compile` at any procedural `SKILL.md`:

```bash
# Uses 3x grounded System-2 extraction + self-consistent merging (requires OPENAI_API_KEY)
anys1 compile examples/code_review_skill/SKILL.md -o examples/code_review_skill/compiled

# Or run offline with the deterministic markdown structural parser (zero API calls):
anys1 compile examples/code_review_skill/SKILL.md -o examples/code_review_skill/compiled --offline
```

`anys1 compile` generates **three ready-to-use artifacts** inside the output folder:

1. **`decisions.json`**: Structured specification of all extracted `D`, `E`, and `G` decision points, their candidate options, and the default cold-start threshold ($\tau_{\text{cold}} = 0.918$).
2. **`SKILL.anys1.md` (Mode A — Drop-In Agent Skill + MCP Hook)**:
   An annotated version of your `SKILL.md` for open-ended coding agents (**Claude Code**, **OpenHands**, **Cursor**, **Codex**). At every `E` decision point, `SKILL.anys1.md` instructs the agent to call the `anys1_decide` MCP tool first:
   ```bash
   # Attach as an MCP stdio server in Claude Code / Cursor / OpenHands:
   anys1 mcp --compiled examples/code_review_skill/compiled --s1-url http://127.0.0.1:8895/decide
   ```
   When $p_{\max} \ge \tau$, `anys1_decide` returns `ACCEPTED_SYSTEM1` with the resolved option in one forward pass; when $p_{\max} < \tau$, it returns `DEFER_TO_SYSTEM2` so the host agent deliberates only on hard states.
3. **`workflow.py` (Mode B — Programmable Python SDK)**:
   A clean Python class (`CompiledSkillWorkflow`) wired into `DualSystemRuntime`:
   ```python
   from examples.code_review_skill.compiled.workflow import CompiledSkillWorkflow

   wf = CompiledSkillWorkflow(tau=0.918, mode="anys1")
   choice = wf.step(
       "dp_02",
       state="Modified files: auth/jwt_session.py (+42 -11). Added bearer token expiry check.",
   )
   print("Chosen branch:", choice)
   print("Execution summary:", wf.runtime.summary())
   ```

### 4. One-Command Warm Calibration (`AnyS1_cold` $\rightarrow$ `AnyS1_warm`)

Start in cold-start mode ($\tau_{\text{cold}} = 0.918$) or log shadow executions (`mode="shadow"`, which records both `System-1` and `System-2` choices to `shadow.jsonl`), then fit a workflow-specific $\tau_{\text{warm}}$ targeting $\ge 95\%$ agreement:

```bash
anys1 calibrate --shadow shadow.jsonl --target-agr 0.95 -o calibration_warm.json
```

---

## 🗂️ Repository Structure

```text
AnyS1/
├── anys1/
│   ├── cli.py                # Unified CLI: `anys1 compile`, `anys1 calibrate`, `anys1 decide`, `anys1 mcp`
│   ├── compiler.py           # Compiles SKILL.md -> decisions.json + SKILL.anys1.md + workflow.py
│   ├── runtime.py            # DualSystemRuntime (D/E/G dispatch, cold/warm gating, shadow logging)
│   ├── s1.py                 # System1Client & CalibratedGate (tau_cold=0.918, fit_from_shadow)
│   ├── llm.py                # OpenAI-compatible System-2 wrapper
│   ├── mcp_bridge.py         # JSON-RPC 2.0 stdio MCP tool server (`anys1_decide`)
│   ├── mining/               # Offline SkillDP data synthesis pipeline (§3.2)
│   │   └── pipeline.py       # 3x extraction, container task synthesis, trajectory locator, 2x consensus
│   └── model/                # System-1 architecture, training & serving (§3.3)
│       ├── head.py           # PointerHead & OptionInteractionHead (permutation-equivariant `oc` head)
│       ├── train_s1.py       # Qwen3.5-4B-Base + LoRA (r=16) + OptionInteractionHead + label smoothing 0.1
│       └── serve_s1.py       # Fast HTTP pointer server (/decide, /v1/models)
└── examples/
    └── code_review_skill/    # Self-contained SKILL.md compilation & dual-system execution example
```

---

## 🏋️ Training `AnyS1-4B` on `SkillDP`

To reproduce `AnyS1-4B` from `Qwen/Qwen3.5-4B-Base` on the **SkillDP** dataset (`train_x3.jsonl`, 77,958 permutation-expanded instances across 25,986 consensus states from 1,072 skills):

```bash
python -m anys1.model.train_s1 \
  --data /path/to/SkillDP/train_x3.jsonl \
  --base Qwen/Qwen3.5-4B-Base \
  --head_type oc \
  --lora 16 \
  --lr 1e-4 \
  --epochs 1 \
  --batch 4 \
  --accum 2 \
  --max_state 7552 \
  --label_smoothing 0.1 \
  --dtype bf16 \
  --out ./checkpoints/AnyS1-4B
```

---

## 📄 License

This project is released under the [Apache-2.0 License](LICENSE).
