"""Startup routing/cancellation invariants; GPU/browser evidence is recorded separately."""
import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

import server
from unified_engine import MainEngine, WarmupCancelled


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.previous = (server._main_warmup_cancel, dict(server._main_warmup_status), server._active_mode)
        server._main_warmup_cancel = None
        server._main_warmup_status.update(status="idle", error=None)

    def tearDown(self):
        server._main_warmup_cancel, status, server._active_mode = self.previous
        server._main_warmup_status.clear()
        server._main_warmup_status.update(status)

    def test_startup_does_not_choose_a_model_for_the_browser(self):
        with patch.object(server, "warmup") as preload, patch.object(server.threading, "Thread") as thread:
            server.start_model_warmup()
        preload.assert_not_called()
        self.assertEqual(thread.call_args.kwargs["target"], server.monitor_model_cache)

    def test_fast_cancels_only_background_preload_and_keeps_ready_cache(self):
        event = threading.Event()
        server._main_warmup_cancel = event
        engine = Mock(ready=True)
        with patch.object(server, "_main_engine", engine):
            self.assertEqual(server.warmup("fast")["status"], "cancelling")
        self.assertTrue(event.is_set())
        engine.stop.assert_not_called()
        server._main_warmup_cancel = None  # Recognition has no preload token.
        self.assertFalse(server.cancel_main_preload())

    def test_main_review_and_aliases_still_preload_once(self):
        with patch.object(server, "missing_requirements", return_value=[]), patch.object(server, "_jobs", {}), \
                patch.object(server, "_main_engine", Mock(ready=False)), patch.object(server.threading, "Thread") as thread:
            for mode in ("main", "review", "conversation", "accurate"):
                self.assertEqual(server.warmup(mode)["status"], "loading")
            self.assertEqual(thread.call_count, 1)
            self.assertIs(thread.call_args.kwargs["args"][0], server._main_warmup_cancel)

    def test_preload_cannot_jump_a_queued_recognition(self):
        with patch.object(server, "missing_requirements", return_value=[]), \
                patch.object(server, "_main_engine", Mock(ready=False)), \
                patch.object(server, "_jobs", {"x": {"status": "queued"}}), patch.object(server.threading, "Thread") as thread:
            self.assertEqual(server.warmup("review")["status"], "task_pending")
        thread.assert_not_called()

    def test_later_fast_request_cannot_cancel_preparation_owned_by_a_main_job(self):
        event = threading.Event()
        server._main_warmup_cancel = event
        for mode in ("main", "review"):
            for status in ("queued", "running"):
                with patch.object(server, "_jobs", {"x": {"status": status, "mode": mode}}):
                    self.assertEqual(server.warmup("fast")["status"], "on_demand")
                    self.assertFalse(event.is_set())

    def test_cancellation_is_polled_even_when_child_emits_no_output(self):
        engine = MainEngine()
        event = threading.Event()
        timer = threading.Timer(.15, event.set)
        timer.start()
        started = time.monotonic()
        try:
            with patch.object(engine, "stop") as stop:
                with self.assertRaises(WarmupCancelled):
                    engine._receive(lambda e: None, 5, event)
                stop.assert_called_once()
            self.assertLess(time.monotonic()-started, 1)
        finally:
            timer.join()

    def test_cancelling_background_load_releases_lock_and_resets_state(self):
        event = threading.Event()
        server._main_warmup_cancel = event
        engine = Mock()
        engine.warmup.side_effect = WarmupCancelled()
        with patch.object(server, "_main_engine", engine), patch.object(server, "stop_moss_worker"), \
                patch.object(server, "stop_diarizen_worker"), patch.object(server, "release_local_models"):
            server.warmup_main_model(event)
        self.assertIsNone(server._main_warmup_cancel)
        self.assertEqual(server._main_warmup_status["status"], "idle")
        self.assertFalse(server._inference_lock.locked())
        self.assertFalse(server._warmup_lock.locked())
        self.assertIs(engine.warmup.call_args.kwargs["cancel_event"], event)

    def test_cancelled_before_gpu_acquisition_never_loads(self):
        event = threading.Event()
        event.set()
        server._main_warmup_cancel = event
        with patch.object(server, "_main_engine") as engine:
            server.warmup_main_model(event)
        engine.warmup.assert_not_called()
        self.assertIsNone(server._main_warmup_cancel)

    def test_missing_mode_or_invalid_mode_remains_an_explicit_error(self):
        with patch.object(server, "missing_requirements", return_value=["missing"]):
            with self.assertRaises(server.HTTPException) as caught:
                server.warmup("review")
            self.assertEqual(caught.exception.status_code, 503)
        with self.assertRaises(server.HTTPException) as caught:
            server.warmup("invalid")
        self.assertEqual(caught.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
