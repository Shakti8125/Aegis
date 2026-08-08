---
name: train-episode
description: Run one MAPPO training iteration and report reward curves against the baseline. Use to kick off or check on training.
context: fork
agent: rl-trainer
disable-model-invocation: true
---

Run one training iteration:
1. Launch marl/train.py with the current config
2. Wait for it to finish (or hit the configured episode budget)
3. Report each reward component separately, wall-clock time, and the
   comparison against marl/baseline.py
4. Save a checkpoint and note its path
