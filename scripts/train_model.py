import sys, os, subprocess

# FIX: this cell previously did NOT mount Google Drive or clone the repo,
# despite the section header promising both ("Mount Google Drive, setup
# repository workspace"). On a fresh Colab runtime, os.path.exists("/content/Aegis")
# is False, chdir never happens, and every later `from encoder...` / `from marl...`
# import fails with ModuleNotFoundError.

# 1. Mount Google Drive (needed later for Stage 5 checkpoint sync)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not running in Colab -- skipping Drive mount.")

# 2. Clone the Aegis repo if it isn't already present
REPO_URL = "https://github.com/Shakti8125/Aegis.git"  # <-- set this to your actual repo
if not os.path.exists("/content/Aegis") and not os.path.exists("Aegis"):
    subprocess.check_call(["git", "clone", REPO_URL, "/content/Aegis"])

# 3. Ensure working directory is Aegis repository root
if os.path.exists("/content/Aegis"):
    os.chdir("/content/Aegis")
elif os.path.exists("Aegis"):
    os.chdir("Aegis")

print(" Working Directory:", os.getcwd())
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
print(f" PyTorch Version: {torch.__version__}")
print(f" CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f" GPU Device Name: {torch.cuda.get_device_name(0)}")

%%time
import sys, os, subprocess

# Install essential dependencies fast without slow C++ builds
deps = ["pettingzoo", "gymnasium", "torch-geometric", "neo4j", "fastapi", "uvicorn", "pydantic", "pytest"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + deps)

# Verify PyTest test suite
env = os.environ.copy()
env["PYTHONPATH"] = "."
res = subprocess.run([sys.executable, "-m", "pytest", "tests/"], env=env, capture_output=True, text=True)
print(res.stdout[-400:] if len(res.stdout) > 400 else res.stdout)


%%time
import os, torch
from encoder.pretrain import PretrainConfig
from encoder.probe import run_probe, ProbeConfig, format_report

print("1 & 2 & 3. GraphSAGE Pretraining & Linear Probe Validation Gate...")
cfg = ProbeConfig(pretrain=PretrainConfig(epochs=15, batch_size=16, lr=3e-3))
# Run the end-to-end probe: collects data, pretrains, freezes, fits linear probe, and scores.
encoder, probe_report = run_probe(cfg, verbose=True)

os.makedirs("encoder/checkpoints", exist_ok=True)
sage_ckpt_path = "encoder/checkpoints/graphsage_encoder_pretrained.pt"
torch.save({
    "model_type": "GraphSAGE",
    "state_dict": encoder.state_dict(),
    "normalization": encoder.normalization_state_dict()
}, sage_ckpt_path)
print(f" Saved GraphSAGE Encoder checkpoint to {sage_ckpt_path} ({os.path.getsize(sage_ckpt_path):,} bytes)")

print(f"\nLinear Probe Passed: {probe_report.passed}")


%%time
import os, torch
import torch.nn as nn
from encoder.hgt_encoder import HGTGraphEncoder
from encoder.dataset import TRAIN_SIZES, collect_sized_dataset, iter_all

print("Pretraining Heterogeneous Graph Transformer (HGT)...")
train_dataset = collect_sized_dataset(TRAIN_SIZES)
train_graphs = list(iter_all(train_dataset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
hgt_encoder = HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2).to(device)

decoders = nn.ModuleDict({
    ntype: nn.Linear(64, hgt_encoder.feature_dims[ntype]) for ntype in hgt_encoder.node_types
})
dec_params = list(hgt_encoder.parameters()) + list(decoders.parameters())
opt = torch.optim.AdamW(dec_params, lr=1e-3, weight_decay=1e-4)

hgt_encoder.train()
for epoch in range(1, 11):
    total_loss = 0.0
    for g in train_graphs[:16]:
        opt.zero_grad()
        out = hgt_encoder(g)
        # FIX: rec_loss used to start as a plain float 0.0. If a graph had zero
        # nodes across every node type, the inner loop below would never touch
        # it, and rec_loss.backward() would crash with
        # AttributeError: 'float' object has no attribute 'backward'.
        # Guard with has_terms and skip graphs that contribute nothing.
        rec_loss = torch.tensor(0.0, device=device)
        has_terms = False
        for ntype, emb in out.node_embeddings.items():
            if g[ntype].num_nodes > 0:
                rec = decoders[ntype](emb)
                rec_loss = rec_loss + nn.functional.mse_loss(rec, g[ntype].x)
                has_terms = True
        if not has_terms:
            continue
        rec_loss.backward()
        opt.step()
        total_loss += rec_loss.item()
    if epoch % 2 == 0 or epoch == 10:
        print(f"  [HGT Epoch {epoch:2d}/10] MSE Reconstruction Loss: {total_loss/16:.4f}")

os.makedirs("encoder/checkpoints", exist_ok=True)
hgt_ckpt_path = "encoder/checkpoints/hgt_encoder_pretrained.pt"
torch.save({
    "model_type": "HGT",
    "hidden_dim": 64,
    "state_dict": hgt_encoder.state_dict()
}, hgt_ckpt_path)
print(f" Saved HGT Encoder checkpoint to {hgt_ckpt_path} ({os.path.getsize(hgt_ckpt_path):,} bytes)")

%%time
# ============================================================
# 1. Collect Offline Trajectory Dataset from PettingZoo Simulator
# ============================================================
import os, pickle
import numpy as np
from simulator.cluster_env import ClusterConfig, ClusterEnv
from marl.vec_env import DEFAULT_EVAL_SCENARIOS, scenario_overrides
from marl.reward import RewardConfig, RewardShaper

print("Collecting offline trajectory logs across 5 fault scenarios...")
trajectories = []
reward_shaper = RewardShaper(RewardConfig())

for sc in DEFAULT_EVAL_SCENARIOS:
    cfg_kwargs = scenario_overrides(sc, max_cycles=100)
    env = ClusterEnv(config=ClusterConfig(**cfg_kwargs))
    for ep in range(3):
        obs, infos = env.reset(seed=ep * 100 + 42)
        states, actions, ep_rewards = [], [], []
        done = False
        step = 0
        
        while not done and step < 100:
            state_vec = env.state()
            act_dict = {agent: env.action_space(agent).sample() for agent in env.agents}
            next_obs, raw_rews, terms, truncs, infos = env.step(act_dict)
            r_scalar, _ = reward_shaper.shape_infos(infos, env.possible_agents)
            
            states.append(state_vec)
            actions.append([act_dict.get(f"service-{i:02d}", 0) for i in range(12)])
            ep_rewards.append(float(r_scalar.mean()))
            
            done = any(terms.values()) or any(truncs.values())
            step += 1
            
        # Compute Returns-To-Go (RTG)
        returns_to_go = []
        discounted_sum = 0.0
        for r in reversed(ep_rewards):
            discounted_sum = r + 0.99 * discounted_sum
            returns_to_go.insert(0, discounted_sum)
            
        trajectories.append({
            "scenario": sc,
            "timesteps": np.arange(len(states)),
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(ep_rewards, dtype=np.float32),
            "returns_to_go": np.array(returns_to_go, dtype=np.float32)
        })
        env.close()

os.makedirs("marl/checkpoints", exist_ok=True)
dt_data_path = "marl/checkpoints/offline_trajectories.pkl"
with open(dt_data_path, "wb") as f:
    pickle.dump(trajectories, f)

print(f" Saved {len(trajectories)} offline trajectories ({os.path.getsize(dt_data_path):,} bytes)")


%%time
# ============================================================
# 2. Decision Transformer (DT) Model & Offline Training
# ============================================================
import torch
import torch.nn as nn
from marl.decision_transformer import DecisionTransformer, DecisionTransformerConfig

# Pretrain Decision Transformer Model using codebase module
state_dim = trajectories[0]["states"].shape[1]
n_agents = trajectories[0]["actions"].shape[1]
n_actions = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
dt_model = DecisionTransformer(state_dim=state_dim, n_actions=n_actions).to(device)
optimizer = torch.optim.AdamW(dt_model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

print(f"Pretraining Decision Transformer (state_dim={state_dim}, n_actions={n_actions})...")
dt_model.train()
for epoch in range(1, 16):
    total_loss = 0.0
    for traj in trajectories:
        seq_len = min(30, len(traj["states"]))
        optimizer.zero_grad()
        
        # Format tensors matching DecisionTransformer input dimensions (B, seq_len, ...)
        rtg = torch.tensor(traj["returns_to_go"][:seq_len], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        st = torch.tensor(traj["states"][:seq_len], dtype=torch.float32).unsqueeze(0)
        ts = torch.tensor(traj["timesteps"][:seq_len], dtype=torch.long).unsqueeze(0)
        
        # Train over primary agent action sequence
        act = torch.tensor(traj["actions"][:seq_len, 0], dtype=torch.long).unsqueeze(0)
        
        logits = dt_model(st, act, rtg, ts)
        loss = loss_fn(logits.reshape(-1, n_actions), act.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if epoch % 3 == 0 or epoch == 15:
        print(f"  [DT Epoch {epoch:2d}/15] Cross-Entropy Loss: {total_loss/len(trajectories):.4f}")

dt_ckpt_path = "marl/checkpoints/decision_transformer_pretrained.pt"
torch.save({
    "model_type": "DecisionTransformer",
    "state_dim": state_dim,
    "state_dict": dt_model.state_dict()
}, dt_ckpt_path)
print(f" Saved Decision Transformer checkpoint to {dt_ckpt_path} ({os.path.getsize(dt_ckpt_path):,} bytes)")


%%time
# ============================================================
# 1. Train MAPPO (Multi-Agent PPO with CTDE & GAE)
# ============================================================
import sys, os, subprocess

# FIX: RUN_ID previously didn't exist anywhere in the notebook, but Stage 4
# referenced it (`marl/checkpoints/{RUN_ID}`), causing a NameError. Define it
# once here as the single source of truth and reuse it below in Stage 4.
RUN_ID = "mappo_run"

print("Launching MAPPO RL Training run on PettingZoo cluster simulator...")
env = os.environ.copy()
env["PYTHONPATH"] = "."
res = subprocess.run([
    sys.executable, "-m", "marl.train",
    "--total-steps", "50000",
    "--n-envs", "4",
    "--lr", "5e-4",
    "--save-dir", f"marl/checkpoints/{RUN_ID}"
], env=env, capture_output=True, text=True)

print(res.stdout[-1200:] if len(res.stdout) > 1200 else res.stdout)
if res.returncode != 0:
    print("stderr:", res.stderr[-500:])
    # FIX: previously a failed training run was only printed, never raised,
    # so later cells (Stage 4 eval) would silently run against a missing/stale
    # checkpoint. Fail loudly instead.
    raise RuntimeError("marl.train subprocess failed -- see stderr above.")

%%time
# ============================================================
# 2. HAPPO & QMIX Multi-Agent RL Policy & Value Decomposition
# ============================================================
import os, torch
import numpy as np
from marl.happo import HAPPO, HAPPOConfig
from marl.qmix import QMIX, QMixer, QMIXConfig
from marl.mappo import RolloutBuffer

print("Initializing HAPPO and QMIX Multi-Agent RL algorithms...")

obs_dim = 38
state_dim = 143
n_agents = 12
n_actions = 6

# Instantiate HAPPO & QMIX
happo_trainer = HAPPO(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=n_actions)
qmix_trainer = QMIX(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=n_actions)

# Demonstrate HAPPO act and update pipeline
dummy_obs = np.random.randn(2, n_agents, obs_dim).astype(np.float32)
dummy_state = np.random.randn(2, state_dim).astype(np.float32)

actions, logprobs, values = happo_trainer.act(dummy_obs, dummy_state)

buffer = RolloutBuffer(n_steps=4, n_envs=2, n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim)
for step in range(4):
    buffer.add(
        obs=dummy_obs,
        state=dummy_state,
        action=actions,
        logprob=logprobs,
        value=values,
        reward=np.zeros((2, n_agents), dtype=np.float32),
        components={k: np.zeros((2, n_agents), dtype=np.float32) for k in ("sla_violation", "latency", "availability", "action_cost", "invalid_action", "terminal")},
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        final_state=dummy_state,
    )

adv = np.random.randn(4, 2, n_agents).astype(np.float32)
ret = adv + buffer.values
happo_stats = happo_trainer.update(buffer, adv, ret, progress=0.1)
print("HAPPO update step stats:", happo_stats)

# Demonstrate QMIX loss computation and target net update
obs_t = torch.randn(4, n_agents, obs_dim)
states_t = torch.randn(4, state_dim)
act_t = torch.randint(0, n_actions, (4, n_agents))
rews_t = torch.randn(4, 1)
next_obs_t = torch.randn(4, n_agents, obs_dim)
next_states_t = torch.randn(4, state_dim)
dones_t = torch.zeros(4, 1)

qmix_loss = qmix_trainer.compute_loss(obs_t, states_t, act_t, rews_t, next_obs_t, next_states_t, dones_t)
qmix_trainer.optimizer.zero_grad()
qmix_loss.backward()
qmix_trainer.optimizer.step()
qmix_trainer.update_target_nets()
print(f"QMIX TD Loss: {qmix_loss.item():.4f}")

happo_ckpt_path = "marl/checkpoints/happo_qmix_policy.pt"
os.makedirs("marl/checkpoints", exist_ok=True)
torch.save({
    "model_type": "HAPPO_QMIX",
    "happo_state": happo_trainer.state_dict(),
    "qmix_state": qmix_trainer.state_dict(),
}, happo_ckpt_path)
print(f" Saved HAPPO/QMIX checkpoint to {happo_ckpt_path} ({os.path.getsize(happo_ckpt_path):,} bytes)")


%%time
# ============================================================
# Stage 4: Benchmark Evaluation (evaluate & beats)
# ============================================================
import json
from pathlib import Path
from marl.evaluation import evaluate, beats, PolicyController, NoOpController
from marl.baseline import RuleBasedController
from marl.vec_env import scenario_overrides
from simulator.cluster_env import ClusterConfig, ClusterEnv

# FIX: RUN_ID was referenced below but never defined anywhere -> NameError.
# It must match the --save-dir used in the MAPPO training cell above. If you
# run this cell standalone (not right after training), set it manually here.
RUN_ID = globals().get("RUN_ID", "mappo_run")

print("Executing Stage 4 Benchmark Evaluation harness...")

# 1. Run direct benchmark evaluation on pod_crash scenario
def make_env():
    return ClusterEnv(config=ClusterConfig(**scenario_overrides("pod_crash", max_cycles=60)))

eval_seeds = [900001, 900002, 900003]

noop_ctrl = NoOpController()
rule_ctrl = RuleBasedController()

report_noop, _ = evaluate(make_env, noop_ctrl, eval_seeds, scenario="pod_crash")
report_rule, _ = evaluate(make_env, rule_ctrl, eval_seeds, scenario="pod_crash")

verdict = beats(report_rule, report_noop)
print(f"\nRule-Based Baseline vs No-Op (pod_crash):")
print(f"  TTR Delta: {verdict['ttr_delta']:.1f} ticks | SLA Delta: {verdict['sla_delta']:.1f}")
print(f"  Beats Both? {verdict['beats_both']}")

# FIX: PolicyController was imported but never used, so the actual trained
# MAPPO policy was never benchmarked here -- only rule-vs-noop was compared,
# contradicting this section's own header ("Compare trained MAPPO / HAPPO
# policy against the rule-based controller baseline"). Load the trained
# checkpoint and evaluate it too.
# NOTE: confirm PolicyController(...)'s constructor signature against your
# actual marl/evaluation.py -- adjust if it expects different args.
run_dir = Path(f"marl/checkpoints/{RUN_ID}")
policy_ckpt_candidates = sorted(run_dir.glob("*.pt")) if run_dir.exists() else []
if policy_ckpt_candidates:
    try:
        trained_ctrl = PolicyController(str(policy_ckpt_candidates[-1]))
        report_policy, _ = evaluate(make_env, trained_ctrl, eval_seeds, scenario="pod_crash")
        verdict_vs_rule = beats(report_policy, report_rule)
        print(f"\nTrained MAPPO Policy vs Rule-Based Baseline (pod_crash):")
        print(f"  TTR Delta: {verdict_vs_rule['ttr_delta']:.1f} ticks | SLA Delta: {verdict_vs_rule['sla_delta']:.1f}")
        print(f"  Beats Both? {verdict_vs_rule['beats_both']}")
    except Exception as e:
        print(f"Could not evaluate trained policy checkpoint ({policy_ckpt_candidates[-1]}): {e}")
else:
    print(f"\nNo trained policy checkpoint found under {run_dir} -- run the MAPPO training cell first.")

# 2. Inspect full training comparison output if present
comp_file = run_dir / "comparison.json"

if comp_file.exists():
    with open(comp_file, "r") as f:
        comp_data = json.load(f)
        
    print("\n" + "=" * 72)
    print("      AEGIS MARL BENCHMARK EVALUATION vs RULE-BASED BASELINE")
    print("=" * 72)
    print(f"{'Scenario':<18} | {'TTR Delta':>10} | {'SLA Delta':>10} | {'Beats Both?':>12}")
    print("-" * 72)
    for v in comp_data.get("verdicts", []):
        print(f"{v['scenario']:<18} | {v['ttr_delta']:>10.1f} | {v['sla_delta']:>10.1f} | {str(v['beats_both']):>12}")
    print("=" * 72)
    won = comp_data.get("scenarios_won_on_both", 0)
    total = comp_data.get("scenarios_total", 5)
    print(f"\nFinal Verdict: MAPPO beat Baseline on both metrics in {won}/{total} scenarios.")
    if won >= 2:
        print(" SUCCESS: MARL policy meets Phase 4 requirement!")
    else:
        print(" Policy beat baseline in < 2 scenarios. More training updates recommended.")
else:
    print("\nTraining comparison file pending full run completion.")

# ============================================================
# Sync Trained Checkpoints to Google Drive
# ============================================================
import shutil
import os

drive_dst = "/content/drive/MyDrive/Aegis_Checkpoints"

# FIX: this cell previously assumed Drive was already mounted (per Stage 1's
# markdown) but Stage 1 never actually called drive.mount(...), so
# os.makedirs() would silently create an ordinary local folder tree instead
# of syncing to Drive -- checkpoints looked "saved" but were lost when the
# Colab runtime recycled. Fail loudly if Drive isn't actually mounted.
if not os.path.ismount("/content/drive"):
    raise RuntimeError(
        "Google Drive is not mounted at /content/drive. "
        "Re-run the Stage 1 setup cell (which now calls drive.mount(...)) before syncing checkpoints."
    )

os.makedirs(os.path.join(drive_dst, "encoder"), exist_ok=True)
os.makedirs(os.path.join(drive_dst, "marl"), exist_ok=True)

print("Synchronizing trained model weights to Google Drive...\n")

# Copy Encoder Checkpoints
enc_src = "encoder/checkpoints"
if os.path.exists(enc_src):
    for f in os.listdir(enc_src):
        sp = os.path.join(enc_src, f)
        if os.path.isfile(sp):
            dp = os.path.join(drive_dst, "encoder", f)
            shutil.copy2(sp, dp)
            print(f"  [Encoder] Synced {f:<32s} ({os.path.getsize(sp):,} bytes)")

# Copy MARL & Decision Transformer Checkpoints
marl_src = "marl/checkpoints"
if os.path.exists(marl_src):
    for root, dirs, files in os.walk(marl_src):
        for f in files:
            if f.endswith(".pt") or f.endswith(".json") or f.endswith(".pkl"):
                sp = os.path.join(root, f)
                rel_dir = os.path.relpath(root, marl_src)
                dp_dir = os.path.join(drive_dst, "marl", rel_dir)
                os.makedirs(dp_dir, exist_ok=True)
                dp = os.path.join(dp_dir, f)
                shutil.copy2(sp, dp)
                print(f"  [MARL]    Synced {os.path.join(rel_dir, f):<32s} ({os.path.getsize(sp):,} bytes)")

print(f"\n\U0001F389 All model weights & artifacts exported to Google Drive:")
print(f"   {drive_dst}")
