## 2026-08-19T01:20:28+05:30
You are Worker 2 for Milestone 2 (Aegis Colab Training Notebook Fixes & Instructions).

Working Directory: c:\Users\Shakti\Documents\Aegis\.agents\worker_m2\
Project Root: c:\Users\Shakti\Documents\Aegis\
Project Spec: c:\Users\Shakti\Documents\Aegis\PROJECT.md
Architecture Skill: c:\Users\Shakti\Documents\Aegis\.agents\skills\aegis-architecture\SKILL.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope & Required Fixes in `notebooks/aegis_training.ipynb`:
1. **Google Colab Environment & Step-by-Step Instructions**:
   - Provide clear, professional markdown cells at the top of the notebook detailing how to run it in Google Colab:
     - Step 1: GPU Runtime Setup (Runtime -> Change runtime type -> T4 GPU).
     - Step 2: Repository Clone / Working Directory Setup (`/content/Aegis` or local path, `sys.path.insert(0, ...)`).
     - Step 3: Pinned Dependency Installation (`gymnasium`, `pettingzoo`, `torch-geometric`, `torch`, `fastapi`, `pydantic`, `matplotlib`, `pandas`).
     - Step 4: CUDA & Environment Verification.
2. **Stage 1 (PyG HeteroData Graph Extraction & GNN Pretraining)**:
   - Fix imports: Replace any broken `from encoder.probe import probe_encoder` with `from encoder.probe import run_probe, ProbeConfig`.
   - Fix state dict saving: Remove non-existent `encoder.normalization_state_dict()` and save `encoder.state_dict()` (which already contains all registered normalization buffers).
   - Fix float loss bug in HGT pretraining: Ensure `rec_loss = torch.tensor(0.0, device=device, requires_grad=True)` or accumulate losses safely with a terms check so `.backward()` never fails on a float.
3. **Stage 2 (Decision Transformer Trajectory Collection & Training)**:
   - Fix agent key naming: `simulator/cluster_env.py` has agents named `service_0`, `service_1`, ..., `service_11` (`f"service_{i}"`), NOT `service-00`. Replace `act_dict.get(f"service-{i:02d}", 0)` with `act_dict.get(f"service_{i}", 0)` so actual actions are recorded instead of defaulting to 0 (NOOP).
4. **Stage 3 (MAPPO Training)**:
   - Fix CLI arguments for `marl.train` invocation to match `marl/train.py` argument parser:
     `--total-env-steps`, `--envs`, `--checkpoint-dir`, `--run-id`, `--device`, `--train-scenario`.
   - Ensure `RUN_ID = "mappo_colab_run"` is explicitly defined in the cell before use.
5. **Stage 4 (Comprehensive Benchmark & Evaluation)**:
   - Load the trained MAPPO checkpoint into a `PolicyController` and benchmark against `RuleBasedController` and `NoOpController`.
   - Compare TTR, SLA violations, and separate reward components.
6. **Notebook Schema & Structural Integrity**:
   - Ensure `notebooks/aegis_training.ipynb` conforms to valid Jupyter Notebook format (nbformat v4) with valid JSON syntax.
   - Also update/synchronize `notebooks/COLAB_TRAINING_GUIDE.md` if applicable.
   - Add/run automated structural tests (e.g. validating notebook JSON parsing, code cell compilation via `compile(code, "<string>", "exec")`, and schema integrity).
   - Run the full test suite (`.venv\Scripts\pytest tests/`) to ensure no regressions.

Output:
Write a comprehensive completion handoff to `c:\Users\Shakti\Documents\Aegis\.agents\worker_m2\handoff.md` detailing all cells updated, instructions added, structural checks performed, and test outputs. Then send a completion message.
