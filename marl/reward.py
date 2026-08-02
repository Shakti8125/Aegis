"""Reward terms for cluster self-healing.

Every component (SLA violations, action cost, terminal recovery bonus) is
logged separately and never collapsed into a single scalar - separated logs
are how reward hacking gets caught early.

Phase 4 - owned by the rl-trainer subagent. See PLAN.md section 3.
"""
