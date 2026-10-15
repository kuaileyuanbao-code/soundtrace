"""Regressions for missing short-speaker evidence; no model weights needed."""
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_worker import cluster_turns, _short_track_prototypes, _short_track_seglst


class SpeakerPrototypeTests(unittest.TestCase):
    def test_same_person_long_turns_do_not_become_two_prototypes(self):
        turns = [[0, 3, 0], [4, 4.8, 1], [6, 9, 0], [10, 11, 1]]
        vectors = [[1, 0], [0, 1], [.9, .1], [.1, .9]]
        diagnostics = {}
        result = cluster_turns(turns, vectors, 2, diagnostics)
        labels = [v[2] for v in result]
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[1], labels[3])
        self.assertNotEqual(labels[0], labels[1])
        self.assertEqual(diagnostics['strategy'], 'short_track_prototypes')

    def test_overlapping_tracks_are_not_merged_to_make_up_the_count(self):
        turns = [[0, 3, 0], [4, 7, 0], [5, 5.8, 2], [5.8, 7.8, 1]]
        vectors = np.asarray([[1, 0], [1, 0], [.3, .8], [0, 1]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        prototypes = _short_track_prototypes(turns, vectors, np.asarray([0, 1]), 2)
        self.assertIsNotNone(prototypes)
        labels = (vectors @ prototypes.T).argmax(axis=1)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[3])

    def test_inconsistent_track_is_not_trusted(self):
        turns = [[0, 3, 0], [4, 7, 0], [8, 9, 1]]
        vectors = np.asarray([[1, 0], [0, 1], [.7, .7]], dtype=np.float32)
        self.assertIsNone(_short_track_prototypes(turns, vectors, np.asarray([0, 1]), 2))

    def test_impossible_overlap_falls_back(self):
        turns = [[0, 3, 0], [4, 7, 0], [5, 6, 1], [5, 6, 2]]
        vectors = np.asarray([[1, 0], [1, 0], [0, 1], [.1, .9]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self.assertIsNone(_short_track_prototypes(turns, vectors, np.asarray([0, 1]), 2))

    def test_track_numbers_and_times_are_not_special_cases(self):
        turns = [[10, 13, 8], [14, 14.8, 2], [16, 19, 8], [20, 21, 2]]
        labels = [v[2] for v in cluster_turns(turns, [[1, 0], [0, 1], [.9, .1], [.1, .9]], 2)]
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[1], labels[3])
        self.assertNotEqual(labels[0], labels[1])

    def test_handoff_uses_supported_new_onset_not_weak_extra_track(self):
        turns = [[0, 3, 0], [2, 2.4, 2], [2.4, 4, 1], [5, 6, 1]]
        diarization = [[0, 3, 0], [2, 2.4, 1], [2.4, 4, 1], [5, 6, 1]]
        timestamps = [dict(text=word, start_time=s, end_time=e) for word, s, e in
                      [('first', 1.8, 1.9), ('still', 2.1, 2.2), ('next', 2.5, 2.6)]]
        rows = _short_track_seglst('sample', timestamps, turns, diarization)
        self.assertEqual([(r['speaker'], r['words']) for r in rows],
                         [('spk1', 'first still'), ('spk2', 'next')])
        self.assertEqual([(r['start_time'], r['end_time']) for r in rows],
                         [(1.8, 2.2), (2.5, 2.6)])

    def test_handoff_without_overlap_keeps_existing_rendering(self):
        from ifly_tcpwer.merge import make_seglst
        turns = [[0, 2, 4], [3, 4, 8]]
        diarization = [[0, 2, 0], [3, 4, 1]]
        timestamps = [dict(text='a', start_time=.1, end_time=.2),
                      dict(text='b', start_time=3.1, end_time=3.2)]
        self.assertEqual(_short_track_seglst('sample', timestamps, turns, diarization),
                         make_seglst('sample', {'timestamps': timestamps}, diarization))

    def test_preserved_gpu_tokens_follow_user_confirmed_handoff(self):
        path = Path(__file__).parent / 'validation/speaker_fix_20261015/gpu_comparison_with_tokens.json'
        if not path.exists():
            self.skipTest('Private GPU evidence is not a distribution fixture')
        records = json.loads(path.read_text(encoding='utf8'))['runs']
        record = next(x for x in records if x['route'] == 'after' and x['case'].startswith('e5f'))
        captured = record['captured']
        rows = _short_track_seglst('sample', captured['asr_timestamps'], captured['turns'],
                                  record['payload']['diarization'])
        self.assertEqual(' '.join(r['words'] for r in rows),
                         ' '.join(r['words'] for r in record['payload']['rows']))
        self.assertTrue(rows[3]['words'].endswith('就 个 人'))
        self.assertTrue(rows[4]['words'].startswith('说 什 么'))
        self.assertEqual(rows[3]['speaker'], rows[0]['speaker'])
        self.assertNotEqual(rows[3]['speaker'], rows[4]['speaker'])

    def test_cached_actual_recordings_and_legacy_branches(self):
        root = Path(__file__).resolve().parents[1]
        before = root / 'app/validation/speaker_fix_20261015/before_sources/main_worker.py'
        evidence = root / 'app/validation/two_speaker_issue_20261014'
        if not before.exists() or not (evidence / 'cpu_diagnosis.json').exists():
            self.skipTest('Private local recordings/evidence are not distribution fixtures')
        legacy = types.ModuleType('legacy_main_worker')
        legacy.__file__ = str(root / 'app/main_worker.py')
        exec(compile(before.read_text(encoding='utf8'), legacy.__file__, 'exec'), legacy.__dict__)
        records = json.loads((evidence / 'cpu_diagnosis.json').read_text(encoding='utf8'))['runs']
        for index, item in enumerate(records):
            turns = item['raw_turns']
            vectors = np.load(evidence / (item['job_id'] + '_embeddings_cpu.npy'))
            old = legacy.cluster_turns(turns, vectors.copy(), 2)
            diagnostics = {}
            new = cluster_turns(turns, vectors.copy(), 2, diagnostics)
            self.assertEqual(len({v[2] for v in new}), 2)
            if index == 0:
                labels = [v[2] for v in new]
                self.assertEqual(labels, [0, 1, 0, 0, 1, 1, 0])
                self.assertNotEqual(old, new)
                self.assertEqual(diagnostics['strategy'], 'short_track_prototypes')
            else:
                self.assertEqual(old, new)
                self.assertEqual(diagnostics['strategy'], 'long_turn_prototypes')
            for count in [0, 1, 3]:
                self.assertEqual(cluster_turns(turns, vectors.copy(), count),
                                 legacy.cluster_turns(turns, vectors.copy(), count))


if __name__ == '__main__':
    unittest.main()
