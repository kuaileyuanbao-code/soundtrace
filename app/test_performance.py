"""Observation invariants; no real ASR, long recordings, or GPU allocations."""
import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from performance import StageTimings, model_state, observe_call


class ObservationTests(unittest.TestCase):
    def test_class_level_instance_method_keeps_self_binding(self):
        class Module:
            def cuda(self, device=None):
                return self, device
        module = Module()
        original = Module.__dict__["cuda"]
        with observe_call(Module, "cuda", StageTimings(), "move_seconds"):
            self.assertEqual(module.cuda(0), (module, 0))
        self.assertIs(Module.__dict__["cuda"], original)

    def test_original_arguments_result_and_descriptor_survive_observation(self):
        class Loader:
            @classmethod
            def load(cls, value, *, option):
                return cls, value, option
        descriptor = Loader.__dict__["load"]
        timer = StageTimings()
        with observe_call(Loader, "load", timer, "load_seconds"):
            self.assertEqual(Loader.load("audio", option=3), (Loader, "audio", 3))
        self.assertIs(Loader.__dict__["load"], descriptor)
        self.assertGreaterEqual(timer.seconds()["load_seconds"], 0)

    def test_observed_exception_restores_inherited_method(self):
        class Worker:
            def infer(self):
                raise ValueError("original failure")
        worker = Worker()
        timer = StageTimings()
        with self.assertRaisesRegex(ValueError, "original failure"):
            with observe_call(worker, "infer", timer, "infer_seconds"):
                worker.infer()
        self.assertNotIn("infer", vars(worker))
        self.assertEqual(timer.calls["infer_seconds"], 1)

    def test_default_timer_does_not_synchronize_cuda(self):
        import torch
        with patch.dict(os.environ, {"SOUNDTRACE_PROFILE_SYNC": "0"}), \
             patch.object(torch.cuda, "synchronize") as synchronize:
            timer = StageTimings()
            with timer.measure("gpu_seconds", gpu=True):
                pass
            synchronize.assert_not_called()
        self.assertEqual(timer.info()["cuda_sync_calls"], 0)

    def test_state_uses_actual_tensors_including_buffers(self):
        class Tensor:
            def __init__(self, device):
                self.device = device
        class Model:
            resident = True  # Deliberately misleading policy flag.
            def parameters(self):
                return iter([Tensor("cpu")])
            def buffers(self):
                return iter([])
        model = Model()
        self.assertEqual(model_state(model)["state"], "cpu_loaded")
        model.buffers = lambda: iter([Tensor("cuda:0")])
        self.assertEqual(model_state(model)["state"], "mixed")
        self.assertEqual(model_state(None)["state"], "unloaded")

    def test_queue_lock_and_processing_are_distinct_and_output_unchanged(self):
        import server
        rows = [{"speaker": "spk1", "start_time": 0, "end_time": 1, "words": "你 好"}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.wav"
            source.write_bytes(b"test")
            job_id = "observation-test"
            received = time.perf_counter() - .5
            job = {"id": job_id, "source": str(source), "job_dir": str(folder), "session_id": "test",
                   "mode": "fast", "status": "queued", "progress": 0,
                   "_received_monotonic": received, "_queued_monotonic": received + .1,
                   "upload_receive_seconds": .1}
            with patch.dict(server._jobs, {job_id: job}), \
                 patch.object(server, "convert_to_wav"), patch.object(server, "stop_moss_worker"), \
                 patch.object(server, "stop_diarizen_worker"), patch.object(server._main_engine, "stop"), \
                 patch.object(server, "load_model", return_value=Mock()), \
                 patch.object(server, "infer_file", return_value=rows), \
                 patch.object(server, "model_snapshot", return_value={}):
                server.run_job(job_id)
                self.assertEqual(json.loads((folder / "result.json").read_text(encoding="utf-8")), rows)
                saved = json.loads((folder / "performance.json").read_text(encoding="utf-8"))
                timings = saved["timings"]
                self.assertGreater(timings["queue_wait_seconds"], .3)
                self.assertLess(timings["gpu_wait_seconds"], .1)
                self.assertGreaterEqual(timings["server_end_to_end_seconds"], .5)
                self.assertNotIn("_queued_monotonic", saved)
                self.assertEqual(saved["source_sha256"], server.SOURCE_SHA256)

    def test_client_clock_is_validated_and_does_not_change_result(self):
        import httpx
        import server
        async def check():
            with tempfile.TemporaryDirectory() as directory:
                job = {"id": "client-test", "status": "completed", "job_dir": directory}
                with patch.dict(server._jobs, {"client-test": job}):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                        data = dict(user_end_to_end_seconds=2, client_submit_seconds=.1, client_after_accept_seconds=1.9)
                        response = await client.post("/api/jobs/client-test/client-timing", json=data)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(job["client_timings"], data)
                        self.assertEqual(job["status"], "completed")
                        data["user_end_to_end_seconds"] = -1
                        response = await client.post("/api/jobs/client-test/client-timing", json=data)
                        self.assertEqual(response.status_code, 400)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
