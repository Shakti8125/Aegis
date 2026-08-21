# Milestone 2 Challenger 2 Verification Report: Aegis Colab Training Notebook Stress Testing

**Verdict**: **REJECTED**

---

## 1. Observation

### Scope of Testing
Empirical adversarial stress testing was performed on `notebooks/aegis_training.ipynb`, `notebooks/COLAB_TRAINING_GUIDE.md`, and underlying Aegis components across four core dimensions:
1. Behavior under both GPU (`cuda`) and CPU fallback modes.
2. Trajectory collection edge cases (empty actions, mismatched service counts, minimum sequence horizons).
3. Stage 1 HGT loss autograd stability and probe gate criteria.
4. Stage 4 evaluation metrics calculation (TTR, SLA violations, separated reward components, policy loading).

Automated stress harness `tests/test_notebook_stress.py` (12 tests) was created and executed using `.venv\Scripts\python` and pytest.

### Discovered Defects

#### Defect 1 (Critical): Stage 1 HGT Pretraining Constructor Signature & Missing Attribute Error
- **File**: `notebooks/aegis_training.ipynb` (Cell 6: `hgt_pretrain`)
- **Observed Code**:
  ```python
  hgt_encoder = HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2).to(device)

  decoders = nn.ModuleDict({
      ntype: nn.Linear(64, hgt_encoder.feature_dims[ntype]) for ntype in hgt_encoder.node_types
  }).to(device)
  ```
- **Direct Empirical Command & Failure Output**:
  ```bash
  .venv\Scripts\python -c "from encoder.hgt_encoder import HGTGraphEncoder; hgt_encoder = HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2)"
  ```
  ```
  TypeError: HGTGraphEncoder.__init__() got an unexpected keyword argument 'hidden_dim'
  ```
  In `encoder/hgt_encoder.py`, `HGTGraphEncoder.__init__(config: EncoderConfig | None = None, *, feature_dims=None, edge_dims=None)` expects an `EncoderConfig` instance rather than unpacked keyword arguments.
  Furthermore, `hgt_encoder` defines neither `.feature_dims` nor `.node_types` on `self` (`FEATURE_DIMS` and `NODE_TYPES` are module-level constants in `encoder.features`). Attempting to access them raises `AttributeError: 'HGTGraphEncoder' object has no attribute 'node_types'`.

#### Defect 2 (Critical): Stage 4 PyTorch 2.6+ `torch.load` `weights_only=True` UnpicklingError & Silent Fallback
- **File**: `notebooks/aegis_training.ipynb` (Cell 14: `baseline_eval`)
- **Observed Code**:
  ```python
  policy_ctrl = None
  if ckpt_candidates:
      ckpt_path = ckpt_candidates[-1]
      print(f"Loading MAPPO policy checkpoint: {ckpt_path}")
      try:
          ckpt = torch.load(ckpt_path, map_location="cpu")
          ...
          mappo_model.load_state_dict(ckpt)
          policy_ctrl = PolicyController(mappo_model, name="mappo")
          print(" Successfully loaded trained PolicyController.")
      except Exception as e:
          print(f" Warning: Could not initialize PolicyController from checkpoint ({e}).")
  ```
- **Direct Empirical Command & Failure Output**:
  ```bash
  .venv\Scripts\python -c "import torch; torch.load('marl/checkpoints/mappo_smoke_run/final.pt', map_location='cpu')"
  ```
  ```
  _pickle.UnpicklingError: Weights only load failed. Unsupported global: GLOBAL torch.torch_version.TorchVersion was not an allowed global by default.
  ```
  PyTorch 2.6+ defaults to `weights_only=True`. The Aegis checkpoint saved by `marl/train.py` contains full configuration metadata including `provenance` with `torch.__version__` (`torch.torch_version.TorchVersion`).
  Because Cell 14 wrapped `torch.load` in a broad `try: ... except Exception as e:` block, the unpickling failure was silently caught and printed as a warning. `policy_ctrl` remained `None`, causing Stage 4 to **completely skip evaluating the trained MAPPO policy** and evaluate only `no-op` and `rule-based` controllers.
  When tested with `torch.load(ckpt_path, map_location="cpu", weights_only=False)`, checkpoint unpickling and `PolicyController` initialization succeed immediately.

---

## 2. Logic Chain

1. **Scope 1 (Device Fallback Handling)**:
   - Evaluated `AegisGraphEncoder`, `HGTGraphEncoder`, `DecisionTransformer`, `HAPPO`, and `QMIX` under `device = torch.device("cpu")`.
   - Verified that models execute forward passes, backward passes, and gradient updates without tensor device mismatch exceptions when properly configured.
   - Tested CLI execution of `marl.train` with `--device cpu` and verified end-to-end checkpoint generation and uncollapsed reward logging.

2. **Scope 2 (Trajectory Collection & Decision Transformer Edge Cases)**:
   - Tested `ClusterEnv` rollouts across variable service topologies (`n_services=6`, `n_services=12`).
   - Verified that `[act_dict.get(f"service_{i}", 0) for i in range(env.n_services)]` handles empty dictionaries `{}` (all agents dead or inactive) and partial action sets without throwing `KeyError`, returning valid NOOP `0` actions.
   - Verified Returns-To-Go (RTG) discounting logic on empty, single-element, and multi-element reward sequences.
   - Verified `DecisionTransformer` forward pass on minimum sequence length `seq_len = 1` and maximum context window `seq_len = 30`.

3. **Scope 3 (HGT Autograd Stability & Probe Gate Criteria)**:
   - Validated that `has_terms` safeguards HGT pretraining from invoking `rec_loss.backward()` on empty graph batches.
   - Validated linear probe classification metrics: `evaluate_predictions` accurately computes balanced accuracy (1/3 for majority predictor on 3 classes), macro-F1, and per-class recall.
   - Validated inverse-frequency class weighting in `fit_linear_probe` and gate threshold enforcement (`GATE_MIN_BALANCED_ACCURACY = 0.60`, `GATE_MIN_MACRO_F1_MARGIN = 0.20`, `GATE_MIN_TYPE_BALANCED_ACCURACY = 0.45`, `GATE_MIN_TYPE_MACRO_F1_MARGIN = 0.10`).
   - Discovered that Cell 6 fails at instantiation due to invalid constructor parameters and missing attributes on `HGTGraphEncoder`.

4. **Scope 4 (Stage 4 Benchmark Evaluation & Policy Controller)**:
   - Validated `run_episode`, `evaluate`, and `summarise` in `marl/evaluation.py`: verified TTR censoring, SLA service/request ticks aggregation, and uncollapsed logging of all 6 reward components (`sla_violation`, `latency`, `availability`, `action_cost`, `invalid_action`, `terminal`).
   - Validated `beats(challenger, incumbent)` verdict logic (requiring victory on both TTR and SLA metrics).
   - Discovered that Stage 4 policy loading fails under PyTorch 2.6+ due to `torch.load(..., weights_only=True)` default behavior, suppressing MAPPO benchmarking.

---

## 3. Caveats

- Tests were run on a local CPU environment with CUDA fallback verification (simulating standard Google Colab T4 and CPU instances).
- Live Neo4j integration tests (`tests/graph/`) require an external Neo4j instance and were skipped as expected.
- AST compilation in `tests/test_notebook_structure.py` validated syntax but did not execute cell logic at runtime, allowing the constructor and unpickling bugs to escape initial review.

---

## 4. Conclusion

**VERDICT: REJECTED**

While the notebook layout, markdown instructions, and architectural structure are well-designed, `notebooks/aegis_training.ipynb` contains two critical runtime blockers:
1. **Cell 6**: `HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2)` must be replaced with `HGTGraphEncoder(EncoderConfig(hidden_dim=64, num_layers=2))` and `hgt_encoder.feature_dims` / `hgt_encoder.node_types` must be replaced with `FEATURE_DIMS` / `NODE_TYPES` from `encoder.features`.
2. **Cell 14**: `torch.load(ckpt_path, map_location="cpu")` must specify `weights_only=False` to prevent PyTorch 2.6+ unpickling errors that silently disable MAPPO policy evaluation.

---

## 5. Verification Method

To independently reproduce the findings:

1. **Reproduce Defect 1 (Cell 6 HGT constructor failure)**:
   ```bash
   .venv\Scripts\python -c "from encoder.hgt_encoder import HGTGraphEncoder; hgt_encoder = HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2)"
   ```

2. **Reproduce Defect 2 (Cell 14 torch.load unpickling failure)**:
   ```bash
   .venv\Scripts\python -c "import torch; torch.load('marl/checkpoints/mappo_smoke_run/final.pt', map_location='cpu')"
   ```

3. **Run Challenger Stress Test Suite**:
   ```bash
   .venv\Scripts\pytest tests/test_notebook_stress.py -v
   ```
