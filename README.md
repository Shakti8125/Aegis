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

## Local dev

```bash
cp .env.example .env     # then set NEO4J_PASSWORD
docker compose up -d     # local Neo4j at bolt://localhost:7687
pytest tests/
```

## Status

Phase 0 (scaffolding) complete. Phases 1–8 not started.

<!-- TODO: architecture summary, baseline-comparison numbers, demo video link -->
