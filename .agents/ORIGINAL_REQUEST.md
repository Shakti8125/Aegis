# Original User Request

## 2026-08-18T19:30:20Z

# Teamwork Project Prompt

Review the Aegis project for bugs, verify library usage against official documentation, and fix the Colab training Jupyter notebook.

Working directory: C:\Users\Shakti\Documents\Aegis
Integrity mode: demo

## Requirements

### R1. Autonomous Bug Resolution
Explore the codebase to identify and fix bugs. The agent team has full autonomy to decide which areas to focus on based on their initial exploration and analysis.

### R2. Core Library Verification
Verify the API usage of the most complex and core libraries (such as PyTorch Geometric and PettingZoo) against their official documentation, and correct any misuse.

### R3. Colab Notebook Fixes
Fix the existing Jupyter notebook used for training models so that it is fully compatible with Google Colab. Provide clear, step-by-step instructions within the notebook on how to run it.

## Verification Resources
- Use the existing test suite (`pytest tests/`) to ensure no regressions are introduced and that core functionality remains intact.

## Acceptance Criteria

### Testing & Stability
- [ ] Running `pytest tests/` succeeds without any failures after all changes are made.

### Notebook Evaluation
- [ ] An independent agent acting as a judge confirms that the Colab notebook's instructions are clear and that its cells are structurally sound for a Colab environment.
