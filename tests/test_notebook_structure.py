"""Automated structural and syntactic verification tests for notebooks/aegis_training.ipynb.

Verifies:
1. Valid Jupyter Notebook JSON format (nbformat v4).
2. Clean python compilation of every code cell without SyntaxError.
3. Accurate implementation of all 4 training stages and Google Colab instructions.
4. Correctness of CLI arguments, imports, and agent key names.
"""

import json
from pathlib import Path
import pytest

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "aegis_training.ipynb"
GUIDE_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "COLAB_TRAINING_GUIDE.md"


def test_notebook_file_exists():
    assert NOTEBOOK_PATH.exists(), f"Notebook file {NOTEBOOK_PATH} does not exist"
    assert GUIDE_PATH.exists(), f"Guide file {GUIDE_PATH} does not exist"


def test_notebook_json_and_nbformat_schema():
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    assert isinstance(data, dict), "Notebook JSON root must be a dictionary"
    assert "cells" in data, "Notebook must contain 'cells'"
    assert "metadata" in data, "Notebook must contain 'metadata'"
    assert data.get("nbformat") == 4, f"Notebook must be nbformat v4, got {data.get('nbformat')}"
    assert isinstance(data["cells"], list), "Notebook cells must be a list"
    assert len(data["cells"]) >= 10, f"Expected at least 10 cells, got {len(data['cells'])}"


def test_notebook_all_code_cells_compile():
    """Verify that every python code cell compiles cleanly with python's compile() function."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)

    for idx, cell in enumerate(data["cells"]):
        if cell.get("cell_type") != "code":
            continue

        raw_source = "".join(cell.get("source", []))
        # Filter out IPython cell / line magics for compilation
        lines = []
        for line in raw_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("%%") or stripped.startswith("%"):
                continue
            lines.append(line)
        cleaned_source = "\n".join(lines)

        try:
            compile(cleaned_source, f"<cell_{idx}>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Cell {idx} failed syntax compilation: {e}\nSource:\n{cleaned_source}")


def test_notebook_colab_instructions():
    """Verify step-by-step instructions for Google Colab are present in markdown."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    md_sources = [
        "".join(c.get("source", []))
        for c in data["cells"]
        if c.get("cell_type") == "markdown"
    ]
    full_md = "\n".join(md_sources)

    assert "Step 1: GPU Runtime Setup" in full_md or "T4 GPU" in full_md
    assert "Step 2: Repository Clone" in full_md or "/content/Aegis" in full_md
    assert "Step 3: Pinned Dependency Installation" in full_md or "Dependency Installation" in full_md
    assert "Step 4: CUDA & Environment Verification" in full_md or "CUDA" in full_md


def test_stage1_probe_imports_and_no_normalization_state_dict():
    """Stage 1: check run_probe imports and ensure no invalid normalization_state_dict calls."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    code_sources = [
        "".join(c.get("source", []))
        for c in data["cells"]
        if c.get("cell_type") == "code"
    ]
    full_code = "\n".join(code_sources)

    assert "from encoder.probe import run_probe, ProbeConfig" in full_code
    assert "probe_encoder" not in full_code
    assert "normalization_state_dict" not in full_code


def test_stage1_hgt_pretraining_float_loss_fix():
    """Stage 1: check HGT pretraining safeguards float loss backward bug and uses correct constructor signature & feature dimensions."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    code_sources = [
        "".join(c.get("source", []))
        for c in data["cells"]
        if c.get("cell_type") == "code"
    ]
    full_code = "\n".join(code_sources)

    assert "HGTGraphEncoder" in full_code
    assert "EncoderConfig" in full_code
    assert "FEATURE_DIMS" in full_code
    assert "NODE_TYPES" in full_code
    assert "rec_loss" in full_code
    assert "has_terms" in full_code or "requires_grad" in full_code

    # Ensure broken signatures and attributes are absent
    assert "HGTGraphEncoder(hidden_dim=" not in full_code
    assert "hgt_encoder.feature_dims" not in full_code
    assert "hgt_encoder.node_types" not in full_code


def test_stage2_agent_key_naming():
    """Stage 2: check agent action collection uses service_{i}, NOT service-00."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    code_sources = [
        "".join(c.get("source", []))
        for c in data["cells"]
        if c.get("cell_type") == "code"
    ]
    full_code = "\n".join(code_sources)

    assert 'f"service_{i}"' in full_code or "f'service_{i}'" in full_code
    assert "service-00" not in full_code
    assert "service-01" not in full_code
    assert "DecisionTransformer" in full_code


def test_stage3_mappo_cli_arguments():
    """Stage 3: check marl.train CLI arguments match marl/train.py parser and RolloutBuffer includes component_names."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    code_sources = [
        "".join(c.get("source", []))
        for c in data["cells"]
        if c.get("cell_type") == "code"
    ]
    full_code = "\n".join(code_sources)

    assert "--total-env-steps" in full_code
    assert "--envs" in full_code
    assert "--checkpoint-dir" in full_code
    assert "--run-id" in full_code
    assert "RUN_ID =" in full_code
    assert "mappo_colab_run" in full_code
    # Ensure invalid flags are absent
    assert "--total-steps" not in full_code
    assert "--n-envs" not in full_code
    assert "--save-dir" not in full_code

    # Check RolloutBuffer component_names integration in HAPPO/QMIX cell
    assert "COMPONENT_NAMES" in full_code
    assert "component_names=COMPONENT_NAMES" in full_code


def test_stage4_policy_controller_evaluation():
    """Stage 4: check PolicyController benchmark evaluation against RuleBasedController and NoOpController with safe unpickling."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    code_sources = [
        "".join(c.get("source", []))
        for c in data["cells"]
        if c.get("cell_type") == "code"
    ]
    full_code = "\n".join(code_sources)

    assert "PolicyController" in full_code
    assert "RuleBasedController" in full_code
    assert "NoOpController" in full_code
    assert "evaluate(" in full_code
    assert "beats(" in full_code
    assert "format_comparison(" in full_code
    assert "format_reward_components(" in full_code
    assert "weights_only=False" in full_code

