# 🛡️ Aegis — Google Colab Training & Deployment Guide

This guide provides step-by-step instructions for running the complete **Aegis** model training pipeline in Google Colab (GPU-accelerated) and deploying the resulting trained weights into the local backend services.

---

## 1. Google Colab Environment Setup

### 1.1 GPU Hardware Selection
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload or open [`notebooks/aegis_training.ipynb`](notebooks/aegis_training.ipynb).
3. Navigate to **Runtime -> Change runtime type**.
4. Under **Hardware accelerator**, select **T4 GPU** (or **A100 GPU** if using Colab Pro).
5. Click **Save**.

### 1.2 Google Drive Mounting & Workspace Directory
To ensure training progress and trained weights persist across session disconnections, mount Google Drive in Cell 1:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The workspace directory will be initialized at `/content/Aegis`, with persistent checkpoints saved to `/content/drive/MyDrive/Aegis_Checkpoints/`.

### 1.3 Dependency Installation
The notebook automatically installs all required dependencies:

```bash
pip install -q gymnasium pettingzoo torch-geometric torch fastapi uvicorn pydantic matplotlib pandas pytest neo4j
```

---

## 2. Repository Map & Artifact File Locations

### 2.1 Core Training Scripts
| Script / Path | Module / Purpose | Description |
| :--- | :--- | :--- |
| `encoder/pretrain.py` | `encoder.pretrain` | Self-supervised pretraining for GraphSAGE encoder (masked feature reconstruction + link prediction). |
| `encoder/probe.py` | `encoder.probe` | Phase 3 validation gate: evaluates linear probe classification on frozen embeddings (`run_probe`, `ProbeConfig`). |
| `encoder/gnn_model.py` | `AegisGraphEncoder` | PyTorch Geometric GraphSAGE architecture and feature norm adapters. |
| `encoder/hgt_encoder.py` | `HGTGraphEncoder` | Heterogeneous Graph Transformer with typed relation attention. |
| `marl/train.py` | `marl.train` | MAPPO training entrypoint with separate reward component logging and baseline comparison. |
| `marl/mappo.py` | `MAPPO`, `RolloutBuffer` | CTDE PPO implementation, GAE(lambda) advantage calculation, and observation encoders. |
| `marl/decision_transformer.py` | `DecisionTransformer` | Causal Decision Transformer sequence model for offline RL. |
| `marl/happo.py` | `HAPPO` | Heterogeneous-Agent Proximal Policy Optimization with sequential updates. |
| `marl/qmix.py` | `QMIX` | Monotonic value decomposition hypernetwork. |
| `marl/baseline.py` | `RuleBasedController` | Heuristic threshold controller and automated hyperparameter tuner (`tune_baseline`). |
| `marl/evaluation.py` | `evaluate`, `beats`, `PolicyController` | Multi-scenario benchmark evaluator comparing MAPPO, Baseline, and No-Op. |

### 2.2 Output Checkpoint Locations
Saved model weights and training logs are generated under `encoder/checkpoints/` and `marl/checkpoints/`:

```
Aegis/
├── encoder/
│   └── checkpoints/
│       ├── gnn_graphsage_pretrained.pt    # Pretrained GraphSAGE encoder weights (state_dict + normalization buffers)
│       └── hgt_encoder_pretrained.pt      # Pretrained Heterogeneous Graph Transformer weights
└── marl/
    └── checkpoints/
        ├── offline_trajectories.pkl        # Offline incident trajectory log dataset
        ├── decision_transformer_pretrained.pt # Pretrained Causal Decision Transformer model
        ├── happo_qmix_policy.pt            # HAPPO & QMIX monotonic value mixer weights
        └── mappo_colab_run/                # MAPPO run directory (defined via RUN_ID)
            ├── config.json                 # Complete run parameters & git provenance
            ├── metrics.jsonl                # Uncollapsed per-reward-component metrics
            ├── final.pt / update_*.pt       # Policy & critic neural network weights
            └── comparison.json             # MAPPO vs Baseline benchmark verdict report
```

---

## 3. Step-by-Step Execution Workflow

The training pipeline executes in 5 sequential stages inside [`notebooks/aegis_training.ipynb`](notebooks/aegis_training.ipynb):

```
+-----------------------------------------------------------------------------------+
|                        AEGIS COLAB TRAINING WORKFLOW                              |
+-----------------------------------------------------------------------------------+
|  STAGE 1: Inductive GNN State Encoder Pretraining                                 |
|  - Self-supervised GraphSAGE pretraining (masked node recon + link prediction)    |
|  - Pretrain Heterogeneous Graph Transformer (HGT) on cluster subgraphs           |
|  - Validate frozen embeddings via linear probe gate (`run_probe(ProbeConfig())`)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  STAGE 2: Decision Transformer Pretraining on Offline Incident Logs               |
|  - Collect (R_t, s_t, a_t) trajectories with correct agent IDs (service_0..11)     |
|  - Pretrain Causal Decision Transformer sequence model for offline RL             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  STAGE 3: Multi-Agent RL Training Loop (MAPPO / HAPPO / QMIX)                     |
|  - MAPPO CTDE training with CLI flags matching marl/train.py parser               |
|  - Separate reward component tracking (SLA, latency, availability, action cost)   |
|  - HAPPO sequential policy updates & QMIX value decomposition hypernetwork        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  STAGE 4: Model Evaluation against Baseline                                       |
|  - Load trained checkpoint into PolicyController                                  |
|  - Benchmark Policy vs RuleBasedController and NoOpController                     |
|  - Confirm victory condition: Beat baseline on BOTH TTR and SLA violation count   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  STAGE 5: Checkpoint Export & Sync to Google Drive                                |
|  - Copy all .pt, .json, and .pkl artifacts to /content/drive/MyDrive/             |
+-----------------------------------------------------------------------------------+
```

### Step 1: Inductive GNN State Encoder Pretraining
Run Cell 5 & 6 in the notebook or execute via Python:

```python
from encoder.probe import run_probe, ProbeConfig
from encoder.pretrain import PretrainConfig

cfg = ProbeConfig(pretrain=PretrainConfig(epochs=15, batch_size=16, lr=3e-3))
encoder, probe_report = run_probe(cfg, verbose=True)
```

> **Validation Gate Requirement**: The linear probe on frozen embeddings must achieve pooled balanced accuracy $\ge 0.60$ and macro-F1 margin $\ge +0.20$ above majority-class baselines across both training and held-out cluster sizes before proceeding.

### Step 2: Decision Transformer Pretraining
Run Cell 8 & 9 to collect trajectory logs from simulator rollouts (using `f"service_{i}"` keys) and pretrain the Causal Decision Transformer model matching target Returns-to-Go ($R_t$).

### Step 3: Multi-Agent RL Training (MAPPO / HAPPO / QMIX)
Run Cell 11 & 12 in the notebook or execute via CLI using the supported argument flags:

```bash
python -m marl.train \
    --total-env-steps 50000 \
    --envs 4 \
    --lr 5e-4 \
    --checkpoint-dir marl/checkpoints \
    --run-id mappo_colab_run \
    --train-scenario mixed \
    --device cuda
```

### Step 4: Model Evaluation against Baseline
Run Cell 14 to evaluate the trained policy in `PolicyController` against `RuleBasedController` and `NoOpController`:
- **Time-to-Recovery (TTR)**: Average steps required to return cluster health to 100%.
- **SLA Violation Ticks**: Number of ticks where cluster availability drops below SLA thresholds.
- **Separate Reward Logging**: Verify each reward component (`sla_violation`, `latency`, `availability`, `action_cost`, `invalid_action`, `terminal`) is reported individually.

### Step 5: Checkpoint Packaging & Sync
Run Cell 16 to export all trained weights from `encoder/checkpoints/` and `marl/checkpoints/` to Google Drive `/content/drive/MyDrive/Aegis_Checkpoints/`.

---

## 4. Deploying Trained Checkpoints for Local Backend Services

Once training completes on Colab, follow these steps to place the model weights into your local Aegis repository for backend deployment:

### 4.1 Download Checkpoints from Google Drive
Download the `Aegis_Checkpoints/` folder from Google Drive to your local machine.

### 4.2 Place Weights into Repository Directories
Copy the downloaded files into their respective local directories under your repository root:

```bash
# 1. Place GNN Encoder weights
cp Aegis_Checkpoints/encoder/gnn_graphsage_pretrained.pt encoder/checkpoints/
cp Aegis_Checkpoints/encoder/hgt_encoder_pretrained.pt encoder/checkpoints/

# 2. Place MARL & Decision Transformer weights
cp -r Aegis_Checkpoints/marl/mappo_colab_run marl/checkpoints/
cp Aegis_Checkpoints/marl/decision_transformer_pretrained.pt marl/checkpoints/
cp Aegis_Checkpoints/marl/happo_qmix_policy.pt marl/checkpoints/
```

### 4.3 Verify Backend Deployment Readiness
Run the integration verification suite to ensure the backend service correctly loads the model checkpoints:

```bash
# Verify test suite
pytest tests/ -v
```

The trained checkpoints are now ready for live streaming inference and real-time self-healing orchestration!
