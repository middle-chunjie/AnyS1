# AnyS1: Compiling Procedural Agent Skills into Calibrated Dual-System Workflows

<p align="center">
  <a href="https://research.smsquirrel.top/"><b>🌐 Project Page</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/datasets/middle-chunjie/SkillDP"><b>📦 SkillDP Dataset (HF)</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/middle-chunjie/AnyS1-4B"><b>🤗 AnyS1-4B Model (HF)</b></a>
</p>

**AnyS1** is an open-source framework and compiler that transforms natural-language procedural agent skills (`SKILL.md` files and standard operating procedures) into **calibrated dual-system (`System-1` + `System-2`) workflows**.

Instead of invoking a heavy frontier `System-2` model (`DeepSeek-V4.1-Flash`, `GPT-6 Sol`, `Claude`) at every routine check and routing branch, **AnyS1** decomposes a skill into three decision types:
- **`D` (Deterministic — 13.7%)**: Surface checks (regex, file existence, numeric thresholds) compiled into zero-token Python predicates.
- **`E` (Enumerative — 72.1%)**: Bounded routing, triage, tool selection, and action choice over 2–6 candidate options, delegated to **`AnyS1-4B`** — a 4B single-forward-pass pointer network (`Qwen3.5-4B-Base` + LoRA + permutation-equivariant `OptionInteractionHead`) guarded by a calibrated confidence gate $p_{\max}(x) \ge \tau$ that automatically falls back to `System-2` when uncertain.
- **`G` (Generative — 14.2%)**: Open-ended code writing and free-form synthesis routed directly to `System-2`.

---

## ✨ Key Results

Across five agent benchmarks (**SkillsBench-A**, **SkillsBench-B**, **SOP-Bench**, **$\tau^2$-bench**, and **ALFWorld**):
- **With `DeepSeek-V4.1-Flash` (`System-2`)**:
  - **`AnyS1_warm`** ($\tau$ calibrated on development runs for $\ge 95\%$ agreement) delegates **62.4%** of enumerative decisions to `AnyS1-4B`, reducing **System-2 tokens by 50.6%** and **end-to-end wall-clock latency by 44.3%** while **improving task success by +0.6 points** (`84.2% -> 84.8%`).
  - **`AnyS1_cold`** ($\tau_{\text{cold}} = 0.918$ transferred directly from held-out **SkillDP** states with **zero** target-workflow runs) delegates **41.8%** of enumerative decisions, cutting **System-2 tokens by 35.1%** and **latency by 30.8%** (`84.6%` task success).
- **With `GPT-6 Sol` (`System-2`)**:
  - The exact same **`AnyS1-4B`** checkpoint (without any retraining) saves **34.7% (`AnyS1_cold`)** to **49.8% (`AnyS1_warm`)** of `GPT-6 Sol` tokens while maintaining **`87.1%–87.2%`** task success (`All-S2`: `87.1%`).

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
   When $p_{\max} \ge \tau$, `anys1_decide` returns `ACCEPTED_SYSTEM1` with the resolved option in ~15ms; when $p_{\max} < \tau$, it returns `DEFER_TO_SYSTEM2` so the host agent deliberates only on hard states.
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
