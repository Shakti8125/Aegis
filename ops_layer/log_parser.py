"""Noisy log lines → structured graph-update events.

Phase 5 — owned by the ops-llm-layer subagent. See PLAN.md section 3.

Two parsing paths
-----------------
1. **Regex fast-path** — the simulator emits log lines in a known format
   (``[TICK nnn] SERVICE svc-XX: event_type details``). This path handles
   the common structured logs with pure regex at zero latency, no LLM call.

2. **LLM fallback** — truly unstructured or ambiguous lines get sent to the
   :class:`LLMClient` for extraction. This is the path that would matter in
   production (real Kubernetes logs are messy); in the simulator the regex
   path covers everything.

Output
------
Every parser method returns a list of :class:`GraphEvent` — each one maps
directly to a single ``MERGE … SET`` in the ingestion pipeline. The events
carry only data that appeared in the log line; the consumer decides whether
to apply them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

from ops_layer.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)


# ==========================================================================
# Output types
# ==========================================================================
class EventType(str, Enum):
    """The kind of graph mutation a parsed log line implies."""

    NODE_UPDATE = "node_update"       # Update properties on an existing node
    EDGE_UPDATE = "edge_update"       # Update properties on an existing edge
    STATUS_CHANGE = "status_change"   # Pod/Service status transition
    FAULT_DETECTED = "fault_detected" # A fault has been identified
    ACTION_TAKEN = "action_taken"     # An agent action was executed
    METRIC_UPDATE = "metric_update"   # A telemetry reading


@dataclass
class GraphEvent:
    """One atomic graph mutation extracted from a log line.

    Attributes
    ----------
    event_type : EventType
        What kind of mutation this represents.
    entity_type : str
        ``"Service"``, ``"Pod"``, or ``"Node"``.
    entity_id : str
        The simulator's own ID (e.g. ``"svc-03-mid"``).
    properties : dict
        Key-value pairs to ``SET`` on the entity.
    source_line : str
        The original log line, for audit.
    tick : int | None
        Simulation tick if parseable.
    """

    event_type: EventType
    entity_type: str
    entity_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    source_line: str = ""
    tick: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


# ==========================================================================
# Regex patterns for structured simulator logs
# ==========================================================================
# [TICK 42] SERVICE svc-03-mid: health=0.45 cpu=0.82 latency=312.5
_TICK_RE = re.compile(r"\[TICK\s+(\d+)\]")
_ENTITY_RE = re.compile(
    r"(SERVICE|POD|NODE)\s+([\w\-]+):\s*(.*)", re.IGNORECASE
)
_KV_RE = re.compile(r"([\w_]+)=([\w.\-]+)")

# Fault lines: [TICK 8] FAULT pod_crash on svc-05-back pod 2 duration=30
_FAULT_RE = re.compile(
    r"FAULT\s+([\w_]+)\s+on\s+([\w\-]+)(?:\s+pod\s+(\d+))?\s*(.*)",
    re.IGNORECASE,
)

# Action lines: [TICK 15] ACTION restart on svc-03-mid (agent service_3)
_ACTION_RE = re.compile(
    r"ACTION\s+([\w_]+)\s+on\s+([\w\-]+)(?:\s*\(agent\s+([\w_]+)\))?\s*(.*)",
    re.IGNORECASE,
)

_ENTITY_TYPE_MAP = {"SERVICE": "Service", "POD": "Pod", "NODE": "Node"}


# ==========================================================================
# Parser
# ==========================================================================
class LogParser:
    """Extracts :class:`GraphEvent` instances from log lines.

    Parameters
    ----------
    llm_client : LLMClient | None
        If provided, lines that the regex fast-path cannot parse will be
        sent to the LLM for structured extraction.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client
        self._llm_parse_count = 0
        self._regex_parse_count = 0
        self._unparsed_count = 0

    # ---------------------------------------------------------------- public
    def parse(self, lines: str | Sequence[str]) -> list[GraphEvent]:
        """Parse one or more log lines into graph events.

        Accepts either a single multi-line string or a sequence of strings.
        """
        if isinstance(lines, str):
            lines = [l for l in lines.splitlines() if l.strip()]
        events: list[GraphEvent] = []
        for line in lines:
            events.extend(self._parse_line(line.strip()))
        return events

    @property
    def stats(self) -> dict[str, int]:
        return {
            "regex_parsed": self._regex_parse_count,
            "llm_parsed": self._llm_parse_count,
            "unparsed": self._unparsed_count,
        }

    # ---------------------------------------------------------------- internal
    def _parse_line(self, line: str) -> list[GraphEvent]:
        if not line:
            return []

        tick = self._extract_tick(line)

        # Try fault pattern first (more specific)
        events = self._try_fault(line, tick)
        if events:
            self._regex_parse_count += 1
            return events

        # Try action pattern
        events = self._try_action(line, tick)
        if events:
            self._regex_parse_count += 1
            return events

        # Try entity key-value pattern
        events = self._try_entity_kv(line, tick)
        if events:
            self._regex_parse_count += 1
            return events

        # Fall back to LLM if available
        if self.llm is not None:
            events = self._llm_parse(line, tick)
            if events:
                self._llm_parse_count += 1
                return events

        self._unparsed_count += 1
        logger.debug("Unparsed log line: %s", line)
        return []

    def _extract_tick(self, line: str) -> int | None:
        m = _TICK_RE.search(line)
        return int(m.group(1)) if m else None

    def _try_entity_kv(self, line: str, tick: int | None) -> list[GraphEvent]:
        m = _ENTITY_RE.search(line)
        if not m:
            return []
        entity_type_raw, entity_id, rest = m.group(1), m.group(2), m.group(3)
        entity_type = _ENTITY_TYPE_MAP.get(entity_type_raw.upper(), entity_type_raw)
        props = {}
        for km in _KV_RE.finditer(rest):
            key, val = km.group(1), km.group(2)
            # Attempt numeric conversion
            try:
                props[key] = int(val)
            except ValueError:
                try:
                    props[key] = float(val)
                except ValueError:
                    props[key] = val

        if not props:
            return []

        return [
            GraphEvent(
                event_type=EventType.METRIC_UPDATE,
                entity_type=entity_type,
                entity_id=entity_id,
                properties=props,
                source_line=line,
                tick=tick,
            )
        ]

    def _try_fault(self, line: str, tick: int | None) -> list[GraphEvent]:
        m = _FAULT_RE.search(line)
        if not m:
            return []
        fault_type, target_id = m.group(1), m.group(2)
        pod_idx = m.group(3)
        props: dict[str, Any] = {"fault_type": fault_type}
        if pod_idx is not None:
            props["pod_index"] = int(pod_idx)
        # Parse any trailing key=value pairs
        rest = m.group(4)
        for km in _KV_RE.finditer(rest):
            key, val = km.group(1), km.group(2)
            try:
                props[key] = int(val)
            except ValueError:
                try:
                    props[key] = float(val)
                except ValueError:
                    props[key] = val

        return [
            GraphEvent(
                event_type=EventType.FAULT_DETECTED,
                entity_type="Service",
                entity_id=target_id,
                properties=props,
                source_line=line,
                tick=tick,
            )
        ]

    def _try_action(self, line: str, tick: int | None) -> list[GraphEvent]:
        m = _ACTION_RE.search(line)
        if not m:
            return []
        action_name, target_id = m.group(1), m.group(2)
        agent_name = m.group(3)
        props: dict[str, Any] = {"action": action_name}
        if agent_name:
            props["agent"] = agent_name
        rest = m.group(4)
        for km in _KV_RE.finditer(rest):
            key, val = km.group(1), km.group(2)
            try:
                props[key] = int(val)
            except ValueError:
                try:
                    props[key] = float(val)
                except ValueError:
                    props[key] = val

        return [
            GraphEvent(
                event_type=EventType.ACTION_TAKEN,
                entity_type="Service",
                entity_id=target_id,
                properties=props,
                source_line=line,
                tick=tick,
            )
        ]

    def _llm_parse(self, line: str, tick: int | None) -> list[GraphEvent]:
        """Send an unparseable line to the LLM for structured extraction."""
        system = (
            "You are a Kubernetes log parser. Extract structured events from "
            "the log line. Return a JSON array of objects, each with keys: "
            '"event_type" (one of: node_update, edge_update, status_change, '
            'fault_detected, action_taken, metric_update), "entity_type" '
            '(Service, Pod, or Node), "entity_id" (the resource name), and '
            '"properties" (a dict of extracted key-value pairs). '
            "Return ONLY the JSON array, no other text."
        )
        try:
            response = self.llm.complete(system, line, temperature=0.0)
            # Strip markdown code fences if present
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"^```\w*\n?", "", response)
                response = re.sub(r"\n?```$", "", response)
            parsed = json.loads(response)
            if not isinstance(parsed, list):
                parsed = [parsed]

            events = []
            for item in parsed:
                events.append(
                    GraphEvent(
                        event_type=EventType(item.get("event_type", "metric_update")),
                        entity_type=item.get("entity_type", "Service"),
                        entity_id=item.get("entity_id", "unknown"),
                        properties=item.get("properties", {}),
                        source_line=line,
                        tick=tick,
                    )
                )
            return events
        except (LLMError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("LLM parse failed for line %r: %s", line, exc)
            return []
