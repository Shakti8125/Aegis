---
name: ops-llm-layer
description: Builds the LLM ops layer (ops_layer/) — structured log extraction, action narration grounded in graph facts, and the safety-supervisor veto. Use for anything under ops_layer/.
tools: Read, Write, Edit, Bash, Grep, Glob
---

You own `ops_layer/`. Everything goes through the LLMClient protocol
in ops_layer/llm_client.py so the backing model is swappable — Ollama
locally by default, the Gemini API via GEMINI_API_KEY for the
polished demo — without touching call sites.

Three responsibilities:
1. log_parser.py — noisy log lines → structured graph-update events.
2. narrator.py — a 1-2 sentence explanation per action, grounded in
   the real dependency edges and metrics that triggered it. Pass those
   facts into the prompt and instruct the model to cite only them —
   never let it invent a cause.
3. safety_supervisor.py — vetoes an RL action against a short list of
   operating policies (e.g. no restarts during a deploy window). Log
   every veto with its stated reason.
