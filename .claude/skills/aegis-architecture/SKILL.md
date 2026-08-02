---
name: aegis-architecture
description: Background on Aegis's architecture and conventions — layer order, reward-logging rules, and the LLMClient adapter pattern. Relevant anywhere in this repo.
---

Data-flow order: simulator -> Neo4j graph -> GNN encoder -> MAPPO
agents -> actions -> LLM ops layer -> backend -> frontend.

Conventions:
- Reward components are always logged separately (marl/reward.py).
- All LLM calls go through the LLMClient protocol
  (ops_layer/llm_client.py) so Ollama and the Gemini API are
  interchangeable.
- Cypher migrations are numbered files in graph/migrations/.
- Never propose Streamlit for the frontend.
