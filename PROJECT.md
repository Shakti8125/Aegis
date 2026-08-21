# Project: Aegis Bug Resolution, Library Verification, and Colab Notebook Fixes

## Architecture
Aegis is a multi-agent reinforcement learning system that learns to diagnose and heal failures in a simulated Kubernetes-like cluster.
- Data Flow: Simulator -> Neo4j Graph -> GNN Encoder -> MAPPO Agents -> Actions -> LLM Ops Layer -> FastAPI Backend -> React/Vite Frontend.
- Core Libraries: PettingZoo ParallelEnv, PyTorch Geometric (GraphSAGE), PyTorch (MAPPO), Neo4j (Cypher), FastAPI (WebSocket / REST).
- Notebooks: Google Colab training workflow in `notebooks/aegis_training.ipynb`.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Bug Resolution + Library Verification | Identify and fix bugs across simulator, marl, encoder, ops_layer, graph, backend, and demo; verify API usage of PyG, PettingZoo, PyTorch, Neo4j, FastAPI against documentation | none | DONE |
| 2 | Colab Notebook Fixes & Instructions | Fix Colab training notebook (`notebooks/aegis_training.ipynb`) for full compatibility, structural integrity, step-by-step instructions, and Colab runtime dependencies | M1 | DONE |
| 3 | Comprehensive Test Suite & Integrity Verification | Ensure full test suite passes (`pytest tests/`) with zero failures or regressions, verifying core functionality and notebook integrity | M1, M2 | DONE |

## Interface Contracts
### Simulator <-> Graph
- `ClusterEnv` steps produce state snapshots, synced to Neo4j graph nodes (Pod, Node, Service) and relationships (HOSTED_ON, DEPENDS_ON).
### Graph <-> Encoder
- Cypher queries extract heterogeneous subgraph into PyG `Data`/`HeteroData` object with node feature tensors and edge indices.
### Encoder <-> MARPO
- GNN encoder outputs node/agent embeddings passed into MAPPO Actor-Critic networks.
### MARPO <-> Ops Layer
- Actions selected by MAPPO agents trigger simulated remediation actions and generate LLM diagnostic explanations citing only real graph facts.
### Ops Layer <-> Backend
- FastAPI exposes endpoints and WebSocket streams for cluster metrics, graph state, agent rewards, and LLM logs.

## Code Layout
- `simulator/`: Cluster simulation, failure injection, PettingZoo environment (`cluster_env.py`).
- `graph/`: Neo4j driver, Cypher migrations (`schema.cypher`, `graph/migrations/`).
- `encoder/`: PyTorch Geometric GNN encoder (`gnn_encoder.py`, `graph_dataset.py`).
- `marl/`: MAPPO implementation (`mappo.py`, `actor_critic.py`, `reward.py`, `baseline.py`, `replay_buffer.py`).
- `ops_layer/`: LLMClient protocol (`llm_client.py`), prompt engine, action executor.
- `backend/`: FastAPI app, REST routes, WebSocket server (`main.py`, `routes/`, `ws/`).
- `demo/`: End-to-end demo scripts and runner.
- `notebooks/`: Jupyter notebooks for Colab training (`aegis_training.ipynb`).
- `tests/`: Pytest test suite covering all modules.
