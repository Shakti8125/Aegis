---
name: generate-incident-narrative
description: Generate a human-readable incident narrative for a completed training episode, grounded in its actual action log and graph state.
context: fork
agent: ops-llm-layer
disable-model-invocation: true
---

Take the episode ID passed as $ARGUMENTS:
1. Pull the episode's action log and the graph state at each action
2. For each action, write a 1-2 sentence explanation grounded only in
   the real dependency edges and metrics at that point — no invented
   causes
3. Assemble the narratives into a single incident timeline
