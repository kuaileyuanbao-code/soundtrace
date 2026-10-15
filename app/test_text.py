"""ASR-only isolation, honest schema, boundary stitching, and API dispatch."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np
import soundfile as sf
import server
from text_transcription import merge_windows, text_row, text_window_plan, TEXT_SINGLE_WINDOW_SECONDS
from unified_engine import MainEngine, missing_requirements


def chunks(a, b):
    return [{'start_time': 0, 'end_time': 14, 'text': a},
            {'start_time': 12, 'end_time': 26, 'text': b}]


class TextTests(unittest.TestCase):
    def test_short_recording_has_no_artificial_overlap(self):
        self.assertEqual(text_window_plan(17.152), [(0.0,17.152,0.0,17.152)])
        self.assertEqual(text_window_plan(TEXT_SINGLE_WINDOW_SECONDS), [(0.0,20.0,0.0,20.0)])

    def test_one_sample_above_ceiling_uses_exact_original_longer_plan(self):
        from unified_engine import window_plan
        for duration in [20+1/16000,24,30,35.98675,155]:
            self.assertEqual(text_window_plan(duration),window_plan(duration,14.0,2.0))

    def test_overlap_chinese_preserves_continuing_punctuation(self):
        text, boundaries, warnings = merge_windows(chunks('今天开会，我们讨论项目进度，', '项目进度。明天继续。'))
        self.assertEqual(text, '今天开会，我们讨论项目进度。明天继续。')
        self.assertEqual(boundaries[0]['matched_tokens'], 4)
        self.assertFalse(warnings)

    def test_english_overlap_uses_words_and_preserves_punctuation(self):
        text, _, warnings = merge_windows(chunks('We discuss the Project plan,', 'project plan. Next step.'))
        self.assertEqual(text, 'We discuss the Project plan. Next step.')
        self.assertFalse(warnings)

    def test_real_repetitions_inside_window_are_retained(self):
        text, _, _ = merge_windows(chunks('你好你好你好，今天开会。', '今天开会。继续你好你好。'))
        self.assertEqual(text, '你好你好你好，今天开会。继续你好你好。')

    def test_short_ambiguous_or_changed_boundary_does_not_drop_words(self):
        for a,b in [('今天很好','好，明天继续'), ('讨论项目进度','项目速度需要提高')]:
            text, boundaries, warnings = merge_windows(chunks(a,b))
            self.assertIn(a,text)
            self.assertIn(b,text)
            self.assertEqual(boundaries[0]['status'], 'unresolved')
            self.assertTrue(warnings)

    def test_nonoverlap_never_deduplicates_same_utterance(self):
        c = chunks('今天开会。','今天开会。')
        c[1]['start_time'] = 14
        text, _, _ = merge_windows(c)
        self.assertEqual(text, '今天开会。今天开会。')

    def test_english_unmatched_boundary_remains_readable(self):
        text, _, _ = merge_windows(chunks('We start.', 'Then continue.'))
        self.assertEqual(text, 'We start. Then continue.')

    def test_blank_window_does_not_match_across_silence(self):
        c = chunks('今天开会。','') + [{'start_time':24,'end_time':36,'text':'今天开会。'}]
        text, _, _ = merge_windows(c)
        self.assertEqual(text,'今天开会。今天开会。')

    def test_long_unrelated_same_transcript_not_wholly_removed(self):
        a='一二三四五六七八九十甲乙丙丁戊己庚辛壬癸'
        text, _, warnings = merge_windows(chunks(a,a))
        self.assertEqual(text,a+a)
        self.assertTrue(warnings)

    def test_honest_row_and_requirements_exclude_main_dependencies(self):
        row=text_row('session','你好。',6)
        self.assertIsNone(row['speaker'])
        self.assertEqual(row['timestamp_kind'],'audio_extent')
        self.assertFalse(row['speaker_analyzed'])
        with patch.object(Path,'exists',return_value=False):
            missing=missing_requirements(text_only=True)
        self.assertFalse(any('ForcedAligner' in p or 'sortformer_4spk' in p or 'campplus' in p for p in missing))

    def test_existing_full_worker_reused_without_prepare_main(self):
        engine=MainEngine();engine.ready=True;engine.process=Mock();engine.process.poll.return_value=None
        engine.model_info={'text_only':False}
        engine.warmup(text_only=True)
        engine.process.stdin.write.assert_not_called()

    def test_text_worker_can_be_upgraded_without_restart(self):
        engine=MainEngine();engine.ready=True;engine.process=Mock();engine.process.poll.return_value=None
        engine.model_info={'text_only':True}
        def response(*args):
            request=json.loads(engine.process.stdin.write.call_args.args[0])
            self.assertEqual(request['command'],'prepare_main')
            return {'type':'result','id':request['id'],'text_only':False,'model_state_after':{}}
        with patch('unified_engine.missing_requirements',return_value=[]), patch.object(engine,'_receive',side_effect=response), patch.object(engine,'stop') as stop:
            engine.warmup()
        stop.assert_not_called()
        self.assertFalse(engine.model_info['text_only'])

    def test_missing_main_dependencies_do_not_claim_ready_or_destroy_text_cache(self):
        engine=MainEngine();engine.ready=True;engine.process=Mock();engine.process.poll.return_value=None
        engine.model_info={'text_only':True}
        with patch('unified_engine.missing_requirements',return_value=['missing aligner']), patch.object(engine,'stop') as stop:
            with self.assertRaisesRegex(RuntimeError,'missing aligner'):engine.warmup()
        engine.process.stdin.write.assert_not_called();stop.assert_not_called()
        self.assertTrue(engine.model_info['text_only'])

    def test_partial_asr_cpu_cache_obeys_existing_memory_release_guard(self):
        engine=MainEngine();engine.ready=True
        engine.state=Mock(return_value={'models':{'qwen':{'state':'cpu_loaded'},'aligner':{'state':'unloaded'},
                                                'sortformer':{'state':'unloaded'},'campp':{'state':'unloaded'}}})
        with patch.object(server,'_main_engine',engine), patch.object(engine,'stop') as stop, \
             patch.object(server,'_active_mode','fast'), patch.object(server,'memory_budget',return_value={'can_retain':False}), \
             patch.object(server,'cache_note'):
            server.discard_inactive_cache()
        stop.assert_called_once()

    def test_text_api_uses_only_text_dispatch_and_does_not_invent_speaker(self):
        async def check():
            wav=io.BytesIO();sf.write(wav,np.zeros(16000,dtype=np.float32),16000,format='WAV')
            payload={'rows':[text_row('test','你好。',1)],'text':'你好。','chunks':[], 'merge_boundaries':[],
                     'timings':{},'warnings':[], 'duration_seconds':1,'model_state_before_warmup':{},
                     'model_state_before':{},'model_state_after':{},'load_observation':{},'profiling':{}}
            with tempfile.TemporaryDirectory() as folder, patch.object(server,'JOBS_DIR',Path(folder)), \
                 patch.dict(server._jobs,{},clear=True), patch.object(server,'missing_requirements',return_value=[]), \
                 patch.object(server._executor,'submit',side_effect=lambda fn,jid: fn(jid)), \
                 patch.object(server,'stop_moss_worker'),patch.object(server,'stop_diarizen_worker'), \
                 patch.object(server,'release_local_models'),patch.object(server,'model_snapshot',return_value={}), \
                 patch.object(server,'discard_inactive_cache'), patch.object(server._main_engine,'transcribe_text',return_value=payload) as text, \
                 patch.object(server._main_engine,'transcribe') as main,patch.object(server,'infer_file') as fast, \
                 patch.object(server,'review_recording') as review,patch.object(server,'normalize_speakers') as speakers:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://test') as client:
                    response=await client.post('/api/jobs',data={'mode':'text'},files={'audio':('test.wav',wav.getvalue(),'audio/wav')})
                    self.assertEqual(response.status_code,202,response.text)
                    job=response.json();self.assertEqual(job['status'],'completed');self.assertEqual(job['speakers'],0)
                    result=(await client.get(job['result_url'])).json()
                    self.assertIsNone(result['rows'][0]['speaker']);self.assertEqual(result['text'],'你好。')
                    self.assertFalse(result['speaker_analyzed']);self.assertEqual(result['timestamp_kind'],'audio_extent')
                    for params in [{'speaker_count':'1'},{'voiceprint_ids':'someone'}]:
                        response=await client.post('/api/jobs',data={'mode':'text',**params},files={'audio':('test.wav',wav.getvalue(),'audio/wav')})
                        self.assertEqual(response.status_code,400)
                text.assert_called_once()
                for fn in [main,fast,review,speakers]:fn.assert_not_called()
        asyncio.run(check())


if __name__=='__main__':unittest.main()
