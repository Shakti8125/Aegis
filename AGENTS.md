# Aegis — Multi-Agent RL for Cluster Self-Healing

## What this is
Multi-agent RL that learns to diagnose and heal failures in a simulated
(later real, demo-only) Kubernetes-like cluster. Full design: PLAN.md.

## Stack
- Simulator: pure Python, PettingZoo ParallelEnv API
- Graph: Neo4j (schema: graph/schema.cypher)
- Encoder: PyTorch Geometric, GraphSAGE
- RL: hand-rolled MAPPO (CTDE + GAE), PyTorch
- LLM ops layer: LLMClient protocol (ops_layer/llm_client.py) —
  Ollama by default, Gemini API via GEMINI_API_KEY for the demo build
- Backend: FastAPI + WebSocket
- Frontend: React + Vite + Tailwind — NOT Streamlit, see PLAN.md §9

## Non-negotiable conventions
- Every reward component is logged separately (marl/reward.py) — never
  collapse them into one scalar.
- Every MAPPO run is compared against marl/baseline.py. Not "done"
  until it beats the baseline on recovery time and SLA violations.
- LLM narrations must cite only real graph facts passed into the
  prompt — no invented causes.
- Cypher migrations are numbered files under graph/migrations/.
- Run `pytest tests/` before considering any phase done.

## Where to look
- Full phased plan: PLAN.md
- Reward design rationale: marl/reward.py docstring
- Graph schema: graph/schema.cypher
