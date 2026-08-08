# Aegis

Multi-agent RL for cluster self-healing: Neo4j knowledge graph → GNN state
encoder → MAPPO cooperative agents → LLM ops layer, with a custom React
dashboard on top.

Full design and phased roadmap: [PLAN.md](PLAN.md).

## Layout

| Path | Phase | Owner |
|---|---|---|
| `simulator/` | 1 — simulated cluster (PettingZoo `ParallelEnv`) | `sim-engineer` |
| `graph/` | 2 — Neo4j schema, migrations, ingestion | `graph-engineer` |
| `encoder/` | 3 — GraphSAGE encoder + linear probe | `gnn-architect` |
| `marl/` | 4 — MAPPO (CTDE + GAE), reward, baseline | `rl-trainer` |
| `ops_layer/` | 5 — log parsing, narration, safety veto | `ops-llm-layer` |
| `backend/` | 6 — FastAPI + WebSocket | main session |
| `frontend/` | 7 — React + Vite + Tailwind dashboard | `frontend-builder` |
| `demo/` | 8 — kind + Chaos Mesh demo (recording only) | main session |
