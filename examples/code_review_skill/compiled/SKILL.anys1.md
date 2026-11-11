<!-- ===================================================================== -->
<!-- Compiled by AnyS1 (tau_cold = 0.918)                          -->
<!-- Dual-System Execution Protocol:                                       -->
<!--   [D] Deterministic : Execute via local check / script (0 LLM tokens) -->
<!--   [E] Enumerative   : Call `anys1_decide(dp_id, state)` tool first;   -->
<!--                       accepts System-1 if p_max >= 0.918, else System-2 -->
<!--   [G] Generative    : Execute directly with System-2 reasoning        -->
<!-- ===================================================================== -->

## AnyS1 Compiled Decision Table

| ID | Type | Section | Decision Question | Candidate Options / Rule |
|:---|:---:|:---|:---|:---|
| `dp_01` | **D** | Step 1: Pre-Flight Diff Check | In section 'Step 1: Pre-Flight Diff Check', how should the agent proceed at: "Check whether the pull request diff is empty or only touches markdown documentation files (`.md`)."? | Rule: `bool(state_context.get('condition_met', False))` → **Option A**: Skip test execution and approve documentation-only diff immediately. / **Option B**: Proceed to security and test triage. |
| `dp_02` | **E** | Step 2: Security Risk Classification | In section 'Step 2: Security Risk Classification', how should the agent proceed at: "Triage the modified files and diff hunks to classify the primary security risk surface:"? | `(A)` **Option A**: Authentication, session tokens, or permission boundary changes (High Risk). / `(B)` **Option B**: Database schema, SQL query construction, or serialization changes (Medium Risk). / `(C)` **Option C**: Pure UI styling, frontend layout, or non-sensitive business logic (Low Risk). |
| `dp_03` | **E** | Step 3: Verification Suite Selection | In section 'Step 3: Verification Suite Selection', how should the agent proceed at: "Select one of the verification suites based on the diff scope and risk surface:"? | `(A)` **Option A**: Run full security + integration + unit test suite (`pytest tests/security tests/integration`). / `(B)` **Option B**: Run targeted unit tests for modified modules only (`pytest -k modified`). / `(C)` **Option C**: Run static linter and type-check only (`ruff check && mypy`). |
| `dp_04` | **G** | Step 4: Final Review Synthesis | In section 'Step 4: Final Review Synthesis', how should the agent proceed at: "Write a comprehensive pull request review summary explaining the security findings, test outcomes, and concrete line-lev"? | *Open-ended System-2 generation* |

---
---
name: pr-security-and-test-review
description: Procedural skill for reviewing pull request diffs, triaging risk level, selecting verification suites, and drafting review feedback.
---

# PR Security and Test Review Skill

## Step 1: Pre-Flight Diff Check
- Check whether the pull request diff is empty or only touches markdown documentation files (`.md`).
> **[AnyS1 `dp_01` | Type-D Deterministic Check]** Evaluate deterministic rule `bool(state_context.get('condition_met', False))` without LLM tokens.

  - **Option A**: Skip test execution and approve documentation-only diff immediately.
  - **Option B**: Proceed to security and test triage.

## Step 2: Security Risk Classification
- Triage the modified files and diff hunks to classify the primary security risk surface:
> **[AnyS1 `dp_02` | Type-E Enumerative Decision]** Call `anys1_decide(dp_id="dp_02", state=...)` over options ["**Option A**: Authentication, session tokens, or permission boundary changes (High Risk).", "**Option B**: Database schema, SQL query construction, or serialization changes (Medium Risk).", "**Option C**: Pure UI styling, frontend layout, or non-sensitive business logic (Low Risk)."] (gate $\tau=0.918$).

  - **Option A**: Authentication, session tokens, or permission boundary changes (High Risk).
  - **Option B**: Database schema, SQL query construction, or serialization changes (Medium Risk).
  - **Option C**: Pure UI styling, frontend layout, or non-sensitive business logic (Low Risk).

## Step 3: Verification Suite Selection
- Select one of the verification suites based on the diff scope and risk surface:
> **[AnyS1 `dp_03` | Type-E Enumerative Decision]** Call `anys1_decide(dp_id="dp_03", state=...)` over options ["**Option A**: Run full security + integration + unit test suite (`pytest tests/security tests/integration`).", "**Option B**: Run targeted unit tests for modified modules only (`pytest -k modified`).", "**Option C**: Run static linter and type-check only (`ruff check && mypy`)."] (gate $\tau=0.918$).

  - **Option A**: Run full security + integration + unit test suite (`pytest tests/security tests/integration`).
  - **Option B**: Run targeted unit tests for modified modules only (`pytest -k modified`).
  - **Option C**: Run static linter and type-check only (`ruff check && mypy`).

## Step 4: Final Review Synthesis
- Write a comprehensive pull request review summary explaining the security findings, test outcomes, and concrete line-level recommendations for the author.
> **[AnyS1 `dp_04` | Type-G Generative Step]** Invoke System-2 directly for open-ended synthesis.

