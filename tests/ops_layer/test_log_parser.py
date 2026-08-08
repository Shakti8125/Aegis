"""Tests for ops_layer/log_parser.py — regex parsing and LLM fallback."""

from __future__ import annotations

import pytest

from ops_layer.llm_client import StubClient
from ops_layer.log_parser import EventType, GraphEvent, LogParser


# ==========================================================================
# Regex fast-path
# ==========================================================================
class TestRegexParsing:
    def test_entity_kv_line(self):
        parser = LogParser()
        events = parser.parse("[TICK 42] SERVICE svc-03-mid: health=0.45 cpu=0.82 latency=312.5")
        assert len(events) == 1
        e = events[0]
        assert e.event_type is EventType.METRIC_UPDATE
        assert e.entity_type == "Service"
        assert e.entity_id == "svc-03-mid"
        assert e.tick == 42
        assert e.properties["health"] == 0.45
        assert e.properties["cpu"] == 0.82
        assert e.properties["latency"] == 312.5

    def test_pod_kv_line(self):
        parser = LogParser()
        events = parser.parse("[TICK 10] POD pod-05-2: status=NotReady health=0.0 restart_count=3")
        assert len(events) == 1
        e = events[0]
        assert e.entity_type == "Pod"
        assert e.entity_id == "pod-05-2"
        assert e.properties["status"] == "NotReady"
        assert e.properties["health"] == 0.0
        assert e.properties["restart_count"] == 3

    def test_node_kv_line(self):
        parser = LogParser()
        events = parser.parse("[TICK 5] NODE node-2: cpu_pct=0.95 mem_pct=0.78")
        assert len(events) == 1
        e = events[0]
        assert e.entity_type == "Node"
        assert e.entity_id == "node-2"

    def test_fault_line(self):
        parser = LogParser()
        events = parser.parse("[TICK 8] FAULT pod_crash on svc-05-back pod 2 duration=30")
        assert len(events) == 1
        e = events[0]
        assert e.event_type is EventType.FAULT_DETECTED
        assert e.entity_id == "svc-05-back"
        assert e.properties["fault_type"] == "pod_crash"
        assert e.properties["pod_index"] == 2
        assert e.properties["duration"] == 30
        assert e.tick == 8

    def test_action_line(self):
        parser = LogParser()
        events = parser.parse("[TICK 15] ACTION restart on svc-03-mid (agent service_3)")
        assert len(events) == 1
        e = events[0]
        assert e.event_type is EventType.ACTION_TAKEN
        assert e.entity_id == "svc-03-mid"
        assert e.properties["action"] == "restart"
        assert e.properties["agent"] == "service_3"

    def test_multiline_parse(self):
        parser = LogParser()
        text = """\
[TICK 1] SERVICE svc-00-front: health=1.0 cpu=0.1
[TICK 2] SERVICE svc-01-mid: health=0.8 cpu=0.5
"""
        events = parser.parse(text)
        assert len(events) == 2
        assert events[0].entity_id == "svc-00-front"
        assert events[1].entity_id == "svc-01-mid"

    def test_empty_line_ignored(self):
        parser = LogParser()
        events = parser.parse("")
        assert events == []

    def test_no_tick_in_line(self):
        parser = LogParser()
        events = parser.parse("SERVICE svc-00-front: health=1.0")
        assert len(events) == 1
        assert events[0].tick is None

    def test_stats_tracking(self):
        parser = LogParser()
        parser.parse("[TICK 1] SERVICE svc-00-front: health=1.0")
        parser.parse("[TICK 2] FAULT pod_crash on svc-01 duration=10")
        parser.parse("some random garbage line")
        stats = parser.stats
        assert stats["regex_parsed"] == 2
        assert stats["unparsed"] == 1


# ==========================================================================
# LLM fallback
# ==========================================================================
class TestLLMFallback:
    def test_llm_called_for_unparseable_line(self):
        stub = StubClient(
            response='[{"event_type": "metric_update", "entity_type": "Service", '
                     '"entity_id": "svc-unknown", "properties": {"health": 0.5}}]'
        )
        parser = LogParser(llm_client=stub)
        events = parser.parse("completely unstructured log output from kubernetes")
        assert len(events) == 1
        assert events[0].entity_id == "svc-unknown"
        assert parser.stats["llm_parsed"] == 1

    def test_llm_not_called_for_parseable_line(self):
        stub = StubClient(response="should not be called")
        parser = LogParser(llm_client=stub)
        parser.parse("[TICK 1] SERVICE svc-00: health=1.0")
        assert len(stub._calls) == 0

    def test_llm_failure_returns_empty(self):
        stub = StubClient(response="not valid json {{{")
        parser = LogParser(llm_client=stub)
        events = parser.parse("unstructured line that llm fails on")
        assert events == []
        assert parser.stats["unparsed"] == 1

    def test_llm_handles_markdown_fences(self):
        stub = StubClient(
            response='```json\n[{"event_type": "fault_detected", '
                     '"entity_type": "Pod", "entity_id": "pod-01", '
                     '"properties": {"status": "CrashLoopBackOff"}}]\n```'
        )
        parser = LogParser(llm_client=stub)
        events = parser.parse("pod-01 is in CrashLoopBackOff")
        assert len(events) == 1
        assert events[0].event_type is EventType.FAULT_DETECTED


# ==========================================================================
# GraphEvent
# ==========================================================================
class TestGraphEvent:
    def test_as_dict(self):
        e = GraphEvent(
            event_type=EventType.METRIC_UPDATE,
            entity_type="Service",
            entity_id="svc-01",
            properties={"health": 0.5},
            tick=10,
        )
        d = e.as_dict()
        assert d["event_type"] == "metric_update"
        assert d["entity_type"] == "Service"
        assert d["entity_id"] == "svc-01"
        assert d["tick"] == 10
