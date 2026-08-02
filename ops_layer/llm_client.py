"""LLMClient protocol - Ollama by default, Gemini API via GEMINI_API_KEY for the demo build.

All LLM calls in Aegis go through this adapter so the backing model is
swappable without touching call sites.

Phase 5 - owned by the ops-llm-layer subagent. See PLAN.md section 3.
"""
