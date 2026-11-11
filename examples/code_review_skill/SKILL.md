---
name: pr-security-and-test-review
description: Procedural skill for reviewing pull request diffs, triaging risk level, selecting verification suites, and drafting review feedback.
---

# PR Security and Test Review Skill

## Step 1: Pre-Flight Diff Check
- Check whether the pull request diff is empty or only touches markdown documentation files (`.md`).
  - **Option A**: Skip test execution and approve documentation-only diff immediately.
  - **Option B**: Proceed to security and test triage.

## Step 2: Security Risk Classification
- Triage the modified files and diff hunks to classify the primary security risk surface:
  - **Option A**: Authentication, session tokens, or permission boundary changes (High Risk).
  - **Option B**: Database schema, SQL query construction, or serialization changes (Medium Risk).
  - **Option C**: Pure UI styling, frontend layout, or non-sensitive business logic (Low Risk).

## Step 3: Verification Suite Selection
- Select one of the verification suites based on the diff scope and risk surface:
  - **Option A**: Run full security + integration + unit test suite (`pytest tests/security tests/integration`).
  - **Option B**: Run targeted unit tests for modified modules only (`pytest -k modified`).
  - **Option C**: Run static linter and type-check only (`ruff check && mypy`).

## Step 4: Final Review Synthesis
- Write a comprehensive pull request review summary explaining the security findings, test outcomes, and concrete line-level recommendations for the author.
