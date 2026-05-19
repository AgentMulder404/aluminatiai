"""Tests for agent/error_tracker.py."""

import time
import pytest
from error_tracker import ErrorTracker, ErrorEntry


class TestErrorEntry:
    def test_to_dict_minimal(self):
        e = ErrorEntry(timestamp=1000.0, error_type="collection", message="fail")
        d = e.to_dict()
        assert d == {"timestamp": 1000.0, "error_type": "collection", "message": "fail"}

    def test_to_dict_with_gpu_and_stack(self):
        e = ErrorEntry(
            timestamp=1000.0,
            error_type="upload",
            message="timeout",
            gpu_index=2,
            stack_trace="Traceback ...",
        )
        d = e.to_dict()
        assert d["gpu_index"] == 2
        assert d["stack_trace"] == "Traceback ..."


class TestErrorTracker:
    def test_record_and_stats(self):
        et = ErrorTracker(max_entries=10)
        et.record("collection", "NVML error")
        stats = et.get_stats()
        assert stats["error_count_total"] == 1
        assert stats["error_count_last_hour"] == 1
        assert stats["last_error_message"] == "NVML error"
        assert stats["last_error_at"] is not None

    def test_ring_buffer_eviction(self):
        et = ErrorTracker(max_entries=3)
        for i in range(5):
            et.record("test", f"error {i}")
        assert et.total_count == 5
        recent = et.get_recent(10)
        assert len(recent) == 3
        assert recent[0].message == "error 2"
        assert recent[2].message == "error 4"

    def test_get_recent_limit(self):
        et = ErrorTracker()
        for i in range(10):
            et.record("test", f"e{i}")
        recent = et.get_recent(3)
        assert len(recent) == 3
        assert recent[0].message == "e7"

    def test_get_unsent(self):
        et = ErrorTracker()
        t0 = time.time()
        et.record("a", "old")
        t1 = time.time()
        et.record("b", "new")
        unsent = et.get_unsent(t1 - 0.001)
        assert len(unsent) >= 1
        assert any(e.message == "new" for e in unsent)

    def test_exception_captures_stack(self):
        et = ErrorTracker()
        try:
            raise ValueError("boom")
        except ValueError as exc:
            et.record("collection", str(exc), gpu_index=0, exc=exc)
        recent = et.get_recent(1)
        assert len(recent) == 1
        assert recent[0].stack_trace is not None
        assert "ValueError" in recent[0].stack_trace
        assert recent[0].gpu_index == 0

    def test_message_truncation(self):
        et = ErrorTracker()
        long_msg = "x" * 5000
        et.record("test", long_msg)
        assert len(et.get_recent(1)[0].message) == 2000

    def test_stack_truncation(self):
        et = ErrorTracker()
        try:
            raise RuntimeError("a" * 10000)
        except RuntimeError as exc:
            et.record("test", "err", exc=exc)
        entry = et.get_recent(1)[0]
        assert entry.stack_trace is not None
        assert len(entry.stack_trace) <= 4100

    def test_empty_stats(self):
        et = ErrorTracker()
        stats = et.get_stats()
        assert stats["error_count_total"] == 0
        assert stats["error_count_last_hour"] == 0
        assert stats["last_error_message"] is None
        assert stats["last_error_at"] is None

    def test_hourly_window(self):
        et = ErrorTracker()
        old = ErrorEntry(
            timestamp=time.time() - 7200,
            error_type="test",
            message="old",
        )
        et._buffer.append(old)
        et._total_count += 1
        et.record("test", "recent")
        stats = et.get_stats()
        assert stats["error_count_total"] == 2
        assert stats["error_count_last_hour"] == 1
