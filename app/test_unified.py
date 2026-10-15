"""Behavior checks for single-file routing and short/long recording boundaries."""
import asyncio
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_engine import validate_rows, window_plan, owned_tokens, map_chunk_speakers, compact_rows
from main_worker import cluster_turns, Runner


def row(speaker="spk1", start=0, end=1, words="你 好"):
    return dict(session_id="sample", speaker=speaker, start_time=start, end_time=end, words=words)


class SingleFileTests(unittest.TestCase):
    def test_gpu_memory_pressure_retries_without_failing_the_recording(self):
        import torch
        runner = Runner.__new__(Runner)
        runner.resident = True
        runner.batch_size = 2
        runner.qwen = Mock()
        runner.park = Mock()
        runner._transcribe = Mock(side_effect=[torch.cuda.OutOfMemoryError(),
                                               torch.cuda.OutOfMemoryError(), {"rows": [row()]}])
        request = {"id": "test"}
        with patch("main_worker.emit"):
            self.assertEqual(runner.transcribe(request)["rows"], [row()])
        self.assertFalse(runner.resident)
        self.assertEqual(runner.batch_size, 1)
        self.assertTrue(all(call.args == (request,) for call in runner._transcribe.call_args_list))

    def test_insufficient_memory_at_minimum_settings_does_not_loop_forever(self):
        import torch
        runner = Runner.__new__(Runner)
        runner.resident = False
        runner.batch_size = 1
        runner._transcribe = Mock(side_effect=torch.cuda.OutOfMemoryError())
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            runner.transcribe({"id": "test"})

    def test_no_duplicate_boundary_tokens_and_keeps_short_final_chunk(self):
        windows = window_plan(125, 60, 5)
        self.assertEqual(windows[-1][1], 125)
        owners = [(s, e) for _, _, s, e in windows]
        self.assertEqual(owners[0][0], 0)
        self.assertEqual(owners[-1][1], 125)
        for left, right in zip(owners, owners[1:]):
            self.assertEqual(left[1], right[0])
        candidates = [row(start=57, end=58, words="好")]
        first = owned_tokens(candidates, 0, 0, 57.5, "sample")
        second = owned_tokens([row(start=2, end=3, words="好")], 55, 57.5, 112.5, "sample")
        self.assertEqual(len(first) + len(second), 1)

    def test_deduplicate_exact_rows_but_preserve_real_overlap(self):
        result = validate_rows([row(), row(), row("spk2")], 2)
        self.assertEqual(len(result), 2)
        self.assertEqual({r["speaker"] for r in result}, {"spk1", "spk2"})
        for s, e in [(float("nan"), 1), (-1, 1), (0, 3), (1, 1), (2, 2.01)]:
            with self.assertRaises(ValueError):
                validate_rows([row(start=s, end=e)], 2)

    def test_readable_groups_preserve_words_and_other_speakers(self):
        result = compact_rows([row(start=0, end=.2, words="你"), row(start=.2, end=.4, words="好"),
                               row("spk2", .1, .3, "好"), row(start=2, end=3, words="再 见")])
        self.assertEqual([r["words"] for r in result], ["你 好", "好", "再 见"])
        self.assertEqual(result[0]["end_time"], .4)

    def test_short_turns_and_single_speaker_do_not_require_long_prototypes(self):
        self.assertEqual(cluster_turns([[0, .5, 0]], [[1, 0]])[0][2], 0)
        result = cluster_turns([[0, .5, 0], [.6, 1, 1]], [[1, 0], [0, 1]], 2)
        self.assertEqual(len({r[2] for r in result}), 2)
        single = cluster_turns([[0, 3, 0], [4, 7, 1]], [[1, 0], [0, 1]], 1)
        self.assertEqual({r[2] for r in single}, {0})

    def test_chunk_identities_follow_time_support_not_arrival_order(self):
        mapped = map_chunk_speakers([row("local1", 4, 5), row("local2", 0, 1)],
                                   [row("spk1", 0, 2), row("spk2", 3, 6)])
        self.assertEqual([r["speaker"] for r in mapped], ["spk2", "spk1"])

    def test_routes_accept_long_audio_and_legacy_modes(self):
        import httpx
        import numpy as np
        import soundfile as sf
        import server
        async def check():
            audio = io.BytesIO()
            sf.write(audio, np.zeros(125 * 16000, dtype="float32"), 16000, format="WAV")
            with tempfile.TemporaryDirectory() as folder, patch.object(server, "JOBS_DIR", Path(folder)), \
                 patch.object(server._executor, "submit"):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                    for requested, actual in [("main", "main"), ("review", "review"), ("conversation", "main"), ("accurate", "review")]:
                        response = await client.post("/api/jobs", data={"mode": requested, "speaker_count": "2"},
                                                     files={"audio": ("two_minutes.wav", audio.getvalue(), "audio/wav")})
                        self.assertEqual(response.status_code, 202, response.text)
                        self.assertEqual(response.json()["mode"], actual)
                        server._jobs.pop(response.json()["id"])
                    invalid = await client.post("/api/jobs", data={"mode": "bogus"}, files={"audio": ("x.wav", b"x")})
                    self.assertEqual(invalid.status_code, 400)
                    with patch.object(server, "read_voiceprints", return_value=[{"id": "test-profile", "name": "测试姓名"}]):
                        for mode in ["main", "review"]:
                            response = await client.post("/api/jobs", data={"mode": mode, "voiceprint_ids": "test-profile"},
                                                         files={"audio": ("two_minutes.wav", audio.getvalue(), "audio/wav")})
                            self.assertEqual(response.status_code, 202, response.text)
                            self.assertEqual(response.json()["speaker_count"], 1)
                            server._jobs.pop(response.json()["id"])
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
