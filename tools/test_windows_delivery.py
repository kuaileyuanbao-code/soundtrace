"""Focused deployment regression tests; no GPU or model loads."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portable_launcher import prepare, instance_id
import verify_windows_package
import hashlib
import json


class PortablePreparationTests(unittest.TestCase):
    def test_relocation_rebases_the_copied_envs_and_preserves_input_environment(self):
        with tempfile.TemporaryDirectory(prefix="soundtrace-package-") as folder:
            root = Path(folder)
            (root / "runtime/python").mkdir(parents=True)
            (root / "runtime/python/python.exe").touch()
            for relative in (".venv", "runtime/envs/main"):
                env = root / relative
                (env / "Scripts").mkdir(parents=True)
                (env / "Scripts/python.exe").touch()
                (env / "pyvenv.cfg").write_text("home = C:/original-private-python\n")
            with patch.dict(os.environ, {"PYTHONHOME": "C:/old", "PYTHONPATH": "C:/private"}):
                result = prepare(root)
                self.assertEqual(os.environ["PYTHONHOME"], "C:/old")
                self.assertNotIn("PYTHONHOME", result)
                self.assertNotIn("PYTHONPATH", result)
            self.assertEqual(result["SOUNDTRACE_INSTANCE_ID"], instance_id(root))
            self.assertEqual(result["HF_HUB_OFFLINE"], "1")
            self.assertTrue(Path(result["SOUNDTRACE_MAIN_PYTHON"]).is_relative_to(root))
            for relative in (".venv", "runtime/envs/main"):
                cfg = (root / relative / "pyvenv.cfg").read_text(encoding="utf-8")
                self.assertIn(str(root / "runtime/python"), cfg)
                self.assertNotIn("original-private-python", cfg)
            moved = root.with_name(root.name + "-moved")
            self.assertEqual(moved.resolve().parent, root.resolve().parent)
            self.assertFalse(moved.exists())
            root.rename(moved)
            try:
                second = prepare(moved)
                self.assertNotEqual(second["SOUNDTRACE_INSTANCE_ID"], result["SOUNDTRACE_INSTANCE_ID"])
                self.assertIn(str(moved / "runtime/python"), (moved / ".venv/pyvenv.cfg").read_text())
            finally:
                moved.rename(root)

    def test_missing_runtime_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "runtime/python/python.exe"):
                prepare(Path(folder))


class LocalModelPathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
        import server
        cls.server = server

    def test_developer_defaults_are_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.server.fast_model_path("SOUNDTRACE_FAST_ASR_MODEL", "iic/SenseVoiceSmall"), "iic/SenseVoiceSmall")

    def test_local_model_path_is_used_and_missing_path_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"SOUNDTRACE_FAST_ASR_MODEL": folder}):
                self.assertEqual(self.server.fast_model_path("SOUNDTRACE_FAST_ASR_MODEL", "hub-model"), str(Path(folder)))
            with patch.dict(os.environ, {"SOUNDTRACE_FAST_ASR_MODEL": str(Path(folder) / "missing")}):
                with self.assertRaisesRegex(FileNotFoundError, "缺少本地模型"):
                    self.server.fast_model_path("SOUNDTRACE_FAST_ASR_MODEL", "hub-model")

    def test_configured_model_uses_the_native_path_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "中文模型"
            model.mkdir()
            with patch.dict(os.environ, {"SOUNDTRACE_FAST_ASR_MODEL": str(model)}), \
                    patch.object(self.server, "native_model_path", return_value=Path("C:/ascii-model")) as adapter:
                self.assertEqual(self.server.fast_model_path("SOUNDTRACE_FAST_ASR_MODEL", "hub-model"), str(Path("C:/ascii-model")))
                adapter.assert_called_once_with(model)


class PackageVerificationTests(unittest.TestCase):
    def check(self, content, expected):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "model.bin"
            path.write_bytes(content)
            (root / "PACKAGE_MANIFEST.json").write_text(json.dumps({"files": [
                {"path": "model.bin", "size": len(expected), "sha256": hashlib.sha256(expected).hexdigest()}
            ]}))
            with patch.object(verify_windows_package, "ROOT", root):
                return verify_windows_package.main()

    def test_matching_package_passes(self):
        self.assertFalse(self.check(b"same bytes", b"same bytes"))

    def test_same_size_tampering_is_detected(self):
        self.assertTrue(self.check(b"different!", b"same bytes"))


if __name__ == "__main__":
    unittest.main()
