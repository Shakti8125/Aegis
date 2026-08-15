# 🛡️ Aegis — Google Colab Training & Deployment Guide

This guide provides step-by-step instructions for running the complete **Aegis** model training pipeline in Google Colab (GPU-accelerated) and deploying the resulting trained weights into the local backend services.

---

## 1. Google Colab Environment Setup

### 1.1 GPU Hardware Selection
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload or open [`notebooks/aegis_training.ipynb`](file:///c:/Users/Shakti/Documents/Aegis/notebooks/aegis_training.ipynb).
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
pip install -q gymnasium pettingzoo pytest torch-geometric torch-scatter torch-sparse transformers
```

---

## 2. Repository Map & Artifact File Locations

### 2.1 Core Training Scripts
| Script / Path | Module / Purpose | Description |
| :--- | :--- | :--- |
| [`encoder/pretrain.py`](file:///c:/Users/Shakti/Documents/Aegis/encoder/pretrain.py) | `encoder.pretrain` | Self-supervised pretraining for GraphSAGE encoder (masked feature reconstruction + link prediction). |
| [`encoder/probe.py`](file:///c:/Users/Shakti/Documents/Aegis/encoder/probe.py) | `encoder.probe` | Phase 3 validation gate: evaluates linear probe classification on frozen embeddings. |
| [`encoder/gnn_model.py`](file:///c:/Users/Shakti/Documents/Aegis/encoder/gnn_model.py) | `AegisGraphEncoder` | PyTorch Geometric GraphSAGE architecture and feature norm adapters. |
| [`marl/train.py`](file:///c:/Users/Shakti/Documents/Aegis/marl/train.py) | `marl.train` | MAPPO training entrypoint with separate reward component logging and baseline comparison. |
| [`marl/mappo.py`](file:///c:/Users/Shakti/Documents/Aegis/marl/mappo.py) | `MAPPO`, `RolloutBuffer` | CTDE PPO implementation, GAE(lambda) advantage calculation, and observation encoders. |
| [`marl/baseline.py`](file:///c:/Users/Shakti/Documents/Aegis/marl/baseline.py) | `RuleBasedController` | Heuristic threshold controller and automated hyperparameter tuner (`tune_baseline`). |
| [`marl/evaluation.py`](file:///c:/Users/Shakti/Documents/Aegis/marl/evaluation.py) | `evaluate`, `beats` | Multi-scenario benchmark evaluator comparing MAPPO, Baseline, and No-Op. |

### 2.2 Output Checkpoint Locations
Saved model weights and training logs are generated under `encoder/checkpoints/` and `marl/checkpoints/`:

```
Aegis/
├── encoder/
│   └── checkpoints/
│       ├── gnn_graphsage_pretrained.pt    # Pretrained GraphSAGE encoder weights
│       └── hgt_encoder_pretrained.pt      # Pretrained Heterogeneous Graph Transformer weights
└── marl/
    └── checkpoints/
        ├── offline_trajectories.pkl        # Offline incident trajectory log dataset
        ├── decision_transformer_pretrained.pt # Pretrained Causal Decision Transformer model
        ├── happo_qmix_policy.pt            # HAPPO & QMIX monotonic value mixer weights
        └── <run-id>/                       # MAPPO run directory (e.g. aegis-mappo-colab)
            ├── config.json                 # Complete run parameters & git provenance
            ├── metrics.jsonl                # Uncollapsed per-reward-component metrics
            ├── update_00400.pt / final.pt   # Policy & critic neural network weights
            └── comparison.json             # MAPPO vs Baseline benchmark verdict report
```

---

## 3. Step-by-Step Execution Workflow

The training pipeline executes in 5 sequential stages inside [`notebooks/aegis_training.ipynb`](file:///c:/Users/Shakti/Documents/Aegis/notebooks/aegis_training.ipynb):

```
+-----------------------------------------------------------------------------------+
|                        AEGIS COLAB TRAINING WORKFLOW                              |
+-----------------------------------------------------------------------------------+
|  STAGE 1: Inductive GNN State Encoder Pretraining                                 |
|  - Self-supervised GraphSAGE pretraining (masked node recon + link prediction)    |
|  - Pretrain Heterogeneous Graph Transformer (HGT) on cluster subgraphs           |
|  - Validate frozen embeddings via linear probe gate (`python -m encoder.probe`)   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  STAGE 2: Decision Transformer Pretraining on Offline Incident Logs               |
|  - Collect (R_t, s_t, a_t) trajectories from PettingZoo simulator rollouts         |
|  - Pretrain Causal Decision Transformer sequence model for offline RL             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  STAGE 3: Multi-Agent RL Training Loop (MAPPO / HAPPO / QMIX)                     |
|  - 400 updates over 8 parallel environments on CUDA GPU                           |
|  - Separate reward component tracking (SLA, latency, availability, action cost)   |
|  - HAPPO sequential policy updates & QMIX value decomposition hypernetwork        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  STAGE 4: Model Evaluation against Baseline                                       |
|  - Evaluate trained MAPPO / HAPPO policy against tuned `RuleBasedController`      |
|  - Benchmark across 5 fault scenarios: pod_crash, node_drain, cpu_hog, network, mixed|
|  - Confirm victory condition: Beat baseline on BOTH TTR and SLA violation count   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  STAGE 5: Checkpoint Export & Sync to Google Drive                                |
|  - Copy all `.pt`, `.json`, and `.pkl` artifacts to `/content/drive/MyDrive/`     |
+-----------------------------------------------------------------------------------+
```

### Step 1: Inductive GNN State Encoder Pretraining
Run Cell 5 & 6 in the notebook or execute via CLI:

```bash
python -m encoder.probe
```

> **Validation Gate Requirement**: The linear probe on frozen embeddings must achieve pooled balanced accuracy $\ge 0.60$ and macro-F1 margin $\ge +0.20$ above majority-class baselines across both training and held-out cluster sizes before proceeding.

### Step 2: Decision Transformer Pretraining
Run Cell 8 & 9 to collect trajectory logs from simulator rollouts and pretrain the Causal Decision Transformer model matching target Returns-to-Go ($R_t$).

### Step 3: Multi-Agent RL Training (MAPPO / HAPPO / QMIX)
Run Cell 11 & 12 in the notebook or execute via CLI:

```bash
python -m marl.train \
    --updates 400 \
    --rollout-steps 128 \
    --envs 8 \
    --device cuda \
    --train-scenario mixed \
    --eval-every 50 \
    --checkpoint-every 50 \
    --tune-baseline \
    --run-id aegis-mappo-colab
```

### Step 4: Model Evaluation against Baseline
Run Cell 14 to evaluate policy metrics against `marl/baseline.py`:
- **Time-to-Recovery (TTR)**: Average steps required to return cluster health to 100%.
- **SLA Violation Ticks**: Number of ticks where cluster availability drops below SLA thresholds.

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
cp -r Aegis_Checkpoints/marl/aegis-mappo-colab marl/checkpoints/
cp Aegis_Checkpoints/marl/decision_transformer_pretrained.pt marl/checkpoints/
cp Aegis_Checkpoints/marl/happo_qmix_policy.pt marl/checkpoints/
```

### 4.3 Verify Backend Deployment Readiness
Run the integration verification suite to ensure the backend service correctly loads the model checkpoints:

```bash
# Verify backend WebSocket & inference pipeline
pytest tests/integration/test_marl_baseline.py -v
```

The trained checkpoints are now fully wired into the FastAPI backend service (`backend/main.py`) for live streaming inference and real-time self-healing orchestration!
