"""Aegis Phase 6 — FastAPI backend.

Endpoints:
* ``GET  /health``                — health check
* ``GET  /api/scenarios``         — available fault scenarios
* ``GET  /api/runs``              — available training runs
* ``GET  /api/metrics``           — latest training curves
* ``GET  /api/metrics/{run_id}``  — specific run's training curves
* ``WS   /ws/live``               — live simulation stream
"""
