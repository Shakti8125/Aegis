# Orchestration Plan: Aegis Bug Resolution, Library Verification, and Colab Notebook Fixes

## Objectives
1. Identify and resolve all bugs across simulator, marl, encoder, ops_layer, graph, backend, and demo.
2. Verify API usage of core libraries (PyTorch Geometric, PettingZoo, PyTorch, Neo4j, FastAPI) against official documentation and correct any misuse.
3. Fix the training Jupyter notebook (`notebooks/aegis_training.ipynb`) to be fully compatible with Google Colab, ensuring clear step-by-step instructions and structural validity.
4. Verify the entire project using `pytest tests/` and independent review/challenge, ensuring zero regressions and clean forensic integrity.

## Milestone Plan
### Milestone 1: Exploration, Bug Resolution, and Core Library Verification
- Explorer phase: 3 explorers analyze the codebase for bugs, architectural deviations, library misuse (PyG, PettingZoo, PyTorch, Neo4j, FastAPI), and failing tests.
- Worker phase: 1 worker implements code fixes and library corrections.
- Reviewer phase: 2 reviewers evaluate code correctness, completeness, and documentation alignment.
- Challenger phase: 2 challengers run empirical verification and stress testing.
- Forensic Auditor phase: 1 auditor performs integrity forensics.
- Gate evaluation.

### Milestone 2: Colab Notebook Fixes & Instructions
- Explorer phase: 3 explorers analyze notebook cells, Colab dependency installation, environment setup, execution flow, error points, and instruction clarity.
- Worker phase: 1 worker fixes notebook cells, installs, imports, training loops, and writes step-by-step markdown guidance.
- Reviewer phase: 2 reviewers verify notebook structure, JSON validity, cell formatting, and Colab compatibility.
- Challenger phase: 2 challengers evaluate notebook execution logic and instructions.
- Forensic Auditor phase: 1 auditor performs integrity forensics.
- Gate evaluation.

### Milestone 3: Full Test Suite & Integrity Verification
- Reviewer/Challenger execution of `pytest tests/` across all test suites.
- Verify zero regressions and passing test suite.
- Prepare completion report for Sentinel.
