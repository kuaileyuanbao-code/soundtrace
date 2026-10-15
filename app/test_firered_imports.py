"""Check AED import isolation and the preserved FireRed recognition code."""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'packages/FireRedASR2S'
BEFORE = ROOT / 'app/validation/firered_20261014/before_sources'


class FireRedImportTests(unittest.TestCase):
    def child(self, source):
        command = 'import sys;sys.path.insert(0,' + repr(str(PACKAGE)) + ');' + source
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', command],
                                capture_output=True, text=True, encoding='utf8', timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_aed_import_does_not_import_unused_llm_or_system_components(self):
        names = self.child('from fireredasr2s.fireredasr2 import FireRedAsr2,FireRedAsr2Config;'
            'import json;print(json.dumps(list(sys.modules)))')
        for prefix in ('transformers', 'fireredasr2s.fireredasr2system',
                       'fireredasr2s.fireredasr2.models.fireredasr_llm',
                       'fireredasr2s.fireredasr2.tokenizer.llm_tokenizer',
                       'fireredasr2s.fireredpunc', 'fireredasr2s.fireredvad', 'fireredasr2s.fireredlid'):
            self.assertFalse(any(name == prefix or name.startswith(prefix+'.') for name in names), prefix)

    def test_system_public_exports_remain_accessible_and_cached(self):
        value = self.child('import fireredasr2s as p;'
            'from fireredasr2s import FireRedAsr2System,FireRedAsr2SystemConfig;'
            'import json;print(json.dumps([p.FireRedAsr2System is FireRedAsr2System,'
            'p.FireRedAsr2SystemConfig is FireRedAsr2SystemConfig]))')
        self.assertEqual(value, [True, True])

    def test_recognition_and_loading_code_only_moves_llm_imports(self):
        old = ast.parse((BEFORE / 'firered_asr.py').read_text(encoding='utf8'))
        new = ast.parse((PACKAGE / 'fireredasr2s/fireredasr2/asr.py').read_text(encoding='utf8'))

        class WithoutLlmImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.module and node.module.endswith(('models.fireredasr_llm', 'tokenizer.llm_tokenizer')):
                    return None
                return node

        transform = WithoutLlmImports()
        self.assertEqual(ast.dump(transform.visit(old)), ast.dump(transform.visit(new)))

    def test_aed_inference_config_and_result_serialization_unchanged(self):
        old = ast.parse((BEFORE / 'infer_fireredasr2_aed.py').read_text(encoding='utf8'))
        new = ast.parse((ROOT / 'scripts/infer_fireredasr2_aed.py').read_text(encoding='utf8'))
        def selected(tree):
            config = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name) and n.func.id == 'FireRedAsr2Config')
            serialization = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                                 and isinstance(n.target, ast.Name) and n.target.id == 'result')
            return ast.dump(config), ast.dump(serialization)
        self.assertEqual(selected(old), selected(new))

    def test_review_dispatch_and_resource_guards_unchanged(self):
        old = ast.parse((BEFORE / 'server.py').read_text(encoding='utf8'))
        new = ast.parse((ROOT / 'app/server.py').read_text(encoding='utf8'))
        def without_source_inventory(tree):
            tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'OBSERVED_SOURCES' for t in n.targets))]
            return ast.dump(tree)
        self.assertEqual(without_source_inventory(old), without_source_inventory(new))

    def test_missing_audio_fails_without_creating_a_result(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'raw.json'
            result = subprocess.run([sys.executable, '-X', 'utf8',
                str(ROOT / 'scripts/infer_fireredasr2_aed.py'), '--wav-dir', folder,
                '--output', str(output)], cwd=ROOT, capture_output=True,
                text=True, encoding='utf8', timeout=180)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No WAV files selected', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
