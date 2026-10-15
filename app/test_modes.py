"""Real JS entry points -> multipart API -> real dispatcher, mocked model leaves."""
import asyncio
import copy
import base64
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import httpx
import server


ROWS = [{"session_id": "sample", "speaker": "spk1", "start_time": 0, "end_time": 1, "words": "测 试"}]


def main_payload():
    return {"rows": copy.deepcopy(ROWS), "timings": {}, "warnings": [], "duration_seconds": 1,
            "model_state_before_warmup": {}, "model_state_before": {}, "model_state_after": {},
            "load_observation": {}, "profiling": {}}


class ModeRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        node = shutil.which("node")
        if not node:
            raise RuntimeError("Node.js is required to verify the actual front-end mode entry points")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "frontend.json"
            completed = subprocess.run([node, str(server.APP_DIR / "test_modes_frontend.cjs"),
                                        "--fixtures", str(output)], cwd=server.ROOT,
                                       capture_output=True, text=True, encoding="utf-8", timeout=30)
            if completed.returncode:
                raise AssertionError(completed.stdout + completed.stderr)
            cls.fixtures = json.loads(output.read_text(encoding="utf-8"))
            print(completed.stdout.strip())

    def runtime(self, folder):
        stack = ExitStack()
        stack.enter_context(patch.object(server, "JOBS_DIR", Path(folder)))
        stack.enter_context(patch.dict(server._jobs, {}, clear=True))
        stack.enter_context(patch.dict(server._main_warmup_status, {"status": "idle", "error": None}, clear=True))
        stack.enter_context(patch.object(server._executor, "submit", side_effect=lambda call, job_id: call(job_id)))
        stack.enter_context(patch.object(server, "missing_requirements", return_value=[]))
        for name in ["stop_moss_worker", "stop_diarizen_worker", "release_local_models"]:
            stack.enter_context(patch.object(server, name))
        stack.enter_context(patch.object(server, "model_snapshot", return_value={}))
        stack.enter_context(patch.object(server, "apply_speaker_count_constraint", side_effect=lambda rows, *a: rows))
        stack.enter_context(patch.object(server._main_engine, "stop"))
        stack.enter_context(patch.object(server._main_engine, "park", return_value={}))
        main = stack.enter_context(patch.object(server._main_engine, "transcribe", side_effect=lambda *a: main_payload()))
        def text_payload(*args):
            payload = main_payload()
            payload.update(text="测 试", chunks=[], merge_boundaries=[])
            payload["rows"][0].update(speaker=None, speaker_analyzed=False, timestamp_kind="audio_extent")
            return payload
        text = stack.enter_context(patch.object(server._main_engine, "transcribe_text", side_effect=text_payload))
        review = stack.enter_context(patch.object(server, "review_recording", return_value=(
            copy.deepcopy(ROWS), 0, {"timings": {}, "firered": {"timings": {}}, "mover": {"timings": {}}})))
        fast_model = object()
        stack.enter_context(patch.object(server, "load_model", return_value=fast_model))
        fast = stack.enter_context(patch.object(server, "infer_file", return_value=copy.deepcopy(ROWS)))
        return stack, main, review, fast, fast_model, text

    def test_frontend_import_and_microphone_modes_reach_the_matching_execution_route(self):
        async def check():
            for fixture in self.fixtures:
                mode = fixture["selected_mode"]
                with self.subTest(origin=fixture["origin"], mode=mode), tempfile.TemporaryDirectory() as folder:
                    stack, main, review, fast, fast_model, text = self.runtime(folder)
                    with stack:
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                            response = await client.post("/api/jobs", data={
                                "mode": fixture["mode"], "speaker_count": fixture["speaker_count"],
                                "voiceprint_ids": fixture["voiceprint_ids"]}, files={"audio": (
                                    fixture["filename"], base64.b64decode(fixture["audio_base64"]), "audio/wav")})
                            self.assertEqual(response.status_code, 202, response.text)
                            job = response.json()
                            self.assertEqual((job["mode"], job["requested_mode"], job["status"]), (mode, mode, "completed"))
                            self.assertFalse(job["auto_routed"])
                            result = (await client.get(job["result_url"])).json()
                            self.assertEqual(result["mode"], mode)
                            self.assertEqual(result["rows"][0]["words"], "测 试")
                    self.assertEqual(main.call_count, int(mode in {"main", "review"}))
                    self.assertEqual(review.call_count, int(mode == "review"))
                    self.assertEqual(fast.call_count, int(mode == "fast"))
                    self.assertEqual(text.call_count, int(mode == "text"))
                    if mode == "fast":
                        self.assertIs(fast.call_args.args[0], fast_model)
        asyncio.run(check())

    def test_unavailable_mode_rejects_with_reason_and_never_schedules_another_route(self):
        async def check():
            with tempfile.TemporaryDirectory() as folder, patch.object(server, "JOBS_DIR", Path(folder)), \
                 patch.object(server._executor, "submit") as submit:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                    for mode in ["main", "review", "text"]:
                        with self.subTest(mode=mode), patch.object(server, "missing_requirements", return_value=["missing-model/config.json"]):
                            response = await client.post("/api/jobs", data={"mode": mode},
                                                         files={"audio": ("short.wav", b"test", "audio/wav")})
                            self.assertEqual(response.status_code, 503)
                            self.assertIn("纯文字" if mode == "text" else mode, response.json()["detail"])
                            self.assertIn("missing-model/config.json", response.json()["detail"])
                    submit.assert_not_called()
        asyncio.run(check())

    def test_runtime_failure_keeps_the_requested_mode_without_fallback(self):
        async def check():
            for mode in ["fast", "main", "review", "text"]:
                with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                    stack, main, review, fast, _, text = self.runtime(folder)
                    with stack:
                        failing = {"fast": fast, "main": main, "review": review, "text": text}[mode]
                        failing.side_effect = RuntimeError(f"{mode} 模型无法运行")
                        fixture = self.fixtures[0]
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                            response = await client.post("/api/jobs", data={"mode": mode}, files={"audio": (
                                "short.wav", base64.b64decode(fixture["audio_base64"]), "audio/wav")})
                            job = response.json()
                            self.assertEqual((job["mode"], job["status"]), (mode, "failed"))
                            self.assertIn(f"{mode} 模型无法运行", job["error"])
                        self.assertEqual(fast.call_count, int(mode == "fast"))
                        self.assertEqual(main.call_count, int(mode in {"main", "review"}))
                        self.assertEqual(review.call_count, int(mode == "review"))
                        self.assertEqual(text.call_count, int(mode == "text"))
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
