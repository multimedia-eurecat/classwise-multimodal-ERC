import csv
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from aimara.optimized_evaluation import run_evaluation as runner
with patch.dict(os.environ, EVAL_VIDEO_DIR="", EVAL_EMB_DIR=""):
 from aimara.optimized_evaluation.visual_cache import aligned_audio_window,window_bounds,_frame_indices
from aimara.optimized_evaluation.au_features import extract_crop_aus,pool_aus
from aimara.optimized_evaluation.eval_metrics import check_embeddings

class SelectionTests(unittest.TestCase):
 def test_only_required_branches(self):
  for name,extra in [('iemocap-base',[]),('meld-base',[]),('iemocap-full',['visual_landmarks']),('meld-full',['action_units'])]:
   with self.subTest(name=name):
    stages=runner.profile_stages(name,'sdt','au',Path('/tmp/run'),Path('/tmp/model.pt'))
    self.assertEqual([s[0] for s in stages],['text','audio','visual_vit']+extra+['merge','erc'])
    self.assertEqual(stages[-1][1].count('--checkpoint'),1)
 def test_dry_run_writes_nothing(self):
  with tempfile.TemporaryDirectory() as d:
   with patch('builtins.print'):
    self.assertEqual(runner.main(['/nonexistent','--profile','meld-base','--use-predefined-csv','--output-root',d,'--dry-run']),0)
   self.assertEqual(list(Path(d).iterdir()),[])

class AnnotationTests(unittest.TestCase):
 def check_row(self,start='0',end='1',name='clip.mp4',more=()):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'clip.mp4').touch();source=root/'input.csv'
   with source.open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['video_file','start','end','speaker','text','emotion'])
    writer.writerow([name,start,end,'0','quoted, text','anger'])
    writer.writerows(more)
   return runner.read_annotations(source,root)
 def test_quoted_csv(self):self.assertEqual(self.check_row()[0]['text'],'quoted, text')
 def test_invalid_intervals(self):
  for start,end in [('nan','2'),('0','inf'),('-1','1'),('2','1'),('1','1')]:
   with self.subTest(start=start,end=end),self.assertRaises(ValueError):self.check_row(start,end)
 def test_traversal(self):
  with self.assertRaises(ValueError):self.check_row(name='../clip.mp4')
 def test_overlapping_speakers_allowed(self):
  self.assertEqual(len(self.check_row(end='2',more=[['clip.mp4','1','3','1','other','neutral']])),2)

class AlignmentTests(unittest.TestCase):
 def test_zero_lookahead_contains_every_sampled_frame(self):
  for start,end in [(0,3.3),(3.3,4.94),(4.94,8)]:
   offset,stop=window_bounds(192,24,start,end,0)
   indices=_frame_indices(192,24,start,end)
   self.assertGreaterEqual(int(min(indices)),offset)
   self.assertLess(int(max(indices)),stop)
 def test_audio_matches_video_window_not_remainder(self):
  audio=np.arange(8*16000,dtype=np.float32)
  window=aligned_audio_window(audio,0,127,24)
  self.assertEqual(len(window),round(127*16000/24))
  np.testing.assert_array_equal(window,audio[:len(window)])
 def test_onset_is_aligned_to_decoded_frame(self):
  audio=np.arange(8*16000,dtype=np.float32)
  window=aligned_audio_window(audio,79,48,24)
  self.assertEqual(window[0],round(79*16000/24))
  self.assertEqual(len(window),32000)
 def test_short_audio_padded_with_silence(self):
  window=aligned_audio_window(np.ones(16000),12,24,24)
  np.testing.assert_array_equal(window[:8000],1)
  np.testing.assert_array_equal(window[8000:],0)

class AUTests(unittest.TestCase):
 def test_path_api_and_cleanup(self):
  seen=[]
  class Detector:
   def detect_image(self,paths,batch_size):
    self_path=Path(paths[0]);seen.append(self_path)
    if not self_path.is_file() or batch_size!=1:raise AssertionError('Bad API')
    with Image.open(self_path) as image:
     if image.size!=(16,16):raise AssertionError('Bad crop')
    return pd.DataFrame([{f'AU{i:02}':i/20 for i in range(20)}])
  values=extract_crop_aus(np.ones((16,16,3),dtype=np.uint8),Detector())
  self.assertEqual(values.shape,(20,));self.assertFalse(seen[0].exists())
 def test_operational_errors_surface(self):
  class Detector:
   def detect_image(self,*a,**kw):raise RuntimeError('failed detector')
  with self.assertRaisesRegex(RuntimeError,'failed detector'):
   extract_crop_aus(np.zeros((16,16,3),dtype=np.uint8),Detector())
 def test_no_face_is_missing_feature(self):
  class Detector:
   def detect_image(self,*a,**kw):return pd.DataFrame()
  self.assertIsNone(extract_crop_aus(np.zeros((16,16,3),dtype=np.uint8),Detector()))
 def test_pool_temporal_order(self):
  vector=pool_aus([np.ones(20),np.ones(20)*3])
  np.testing.assert_allclose(vector[:20],2)
  np.testing.assert_allclose(vector[20:40],1)
  np.testing.assert_allclose(vector[40:],2)
 def test_all_zero_extraction_rejected(self):
  with self.assertRaisesRegex(ValueError,'Every embedding is zero'):check_embeddings({0:{0:np.zeros(60)}},60)
 def test_non_finite_rejected(self):
  with self.assertRaises(ValueError):check_embeddings({0:{0:np.ones(60)*np.nan}},60)

class ResumeTests(unittest.TestCase):
 def run_stage(self,root,output,code,resume=False,changed=False):
  return runner.execute_stage('test',[sys.executable,'-c',code],[output],dict(os.environ),root,'fingerprint',resume,changed)
 def test_content_validation_and_downstream_invalidation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'logs').mkdir();output=root/'artifact'
   code=f'from pathlib import Path; Path({str(output)!r}).write_text("good")'
   row,changed=self.run_stage(root,output,code);self.assertEqual(row['status'],'passed')
   row,changed=self.run_stage(root,output,'raise Exception()',True);self.assertEqual(row['status'],'reused');self.assertFalse(changed)
   output.write_text('corrupted')
   row,changed=self.run_stage(root,output,code,True);self.assertEqual(row['status'],'passed');self.assertEqual(output.read_text(),'good')
   row,changed=self.run_stage(root,output,'raise SystemExit(3)',True,True)
   self.assertEqual(row['status'],'failed');self.assertFalse((root/'logs/test.receipt.json').exists())
 def test_missing_output_cannot_succeed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'logs').mkdir()
   row,_=self.run_stage(root,root/'missing','pass');self.assertEqual(row['status'],'failed')


class ReportTests(unittest.TestCase):
 def exercise(self,fail=False,resume=False):
  directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
  root=Path(directory.name);video=root/'clip.mp4';video.touch()
  source=root/'input.csv';source.write_text('video_file,start,end,speaker,text,emotion\nclip.mp4,0,1,0,hello,anger\n')
  run=root/'output'/'fixture'/'iemocap-base'
  def stages(*args):
   output=run/'result.json'
   code='raise SystemExit(4)' if fail else f'import json; from pathlib import Path; Path({str(output)!r}).write_text(json.dumps(dict(overall_accuracy=100,n_correct=1,n_total=1)))'
   return [('erc',[sys.executable,'-c',code],[output])]
  assets=root/'assets'
  (assets/'modelscope/iic/emotion2vec_plus_large').mkdir(parents=True)
  (assets/'light-asd').mkdir()
  for path in [assets/'modelscope/iic/emotion2vec_plus_large'/n for n in ['model.pt','config.yaml','configuration.json']]+[assets/'light-asd/pretrain_AVA_CVPR.model',root/'checkpoint.pt']:
   path.touch()
  args=['--assets-root',str(assets),'--audio-model',str(assets/'modelscope/iic/emotion2vec_plus_large'),'--text-model',str(assets),'--vit-model',str(assets),'--checkpoint',str(root/'checkpoint.pt'),'--device','cpu',str(root),'--profile' ,'iemocap-base','--csv',str(source),'--run-name','fixture','--output-root',str(root/'output')]
  with patch.object(runner,'digest',return_value='hash'),patch.object(runner,'environment_identity',return_value={'cuda_available':False}),patch.object(runner,'profile_stages',side_effect=stages),patch.object(runner.subprocess,'check_output',return_value='{"format":{"duration":"8"}}'),patch('builtins.print'):
   if fail:
    with self.assertRaises(RuntimeError):runner.main(args)
   else:
    runner.main(args)
    if resume:runner.main(args+['--resume'])
  return json.loads((run/'timing_report.json').read_text())
 def test_failed_run_has_no_rtf(self):
  report=self.exercise(fail=True);self.assertEqual(report['status'],'failed');self.assertIsNone(report['pipeline_rtf'])
 def test_reused_outputs_have_no_benchmark_rtf(self):
  report=self.exercise(resume=True);self.assertEqual(report['status'],'resumed');self.assertIsNone(report['pipeline_rtf'])
 def test_fresh_run_uses_actual_media_duration(self):
  report=self.exercise();self.assertEqual(report['media_seconds'],8)
  self.assertAlmostEqual(report['pipeline_rtf'],report['executed_stage_seconds']/8)


class PortabilityTests(unittest.TestCase):
 def test_source_has_no_workstation_paths_or_bytecode(self):
  import ast
  package=Path(runner.__file__).parent
  for path in package.rglob('*.py'):
   for node in ast.walk(ast.parse(path.read_text())):
    if isinstance(node,ast.Constant) and isinstance(node.value,str):
     self.assertFalse(node.value.startswith(('/media/','/home/','/tmp/')),(path,node.value))
  self.assertFalse([p for p in package.rglob('*.pyc') if '__pycache__' not in p.parts])
 def test_inference_uses_repository_model_sources(self):
  from aimara.optimized_evaluation import run_inference
  self.assertEqual(run_inference.Transformer_Based_Model.__module__,'multimodal.model')
  self.assertEqual(run_inference.IEMOCAPDataset.__module__,'multimodal.dataloader')
 def test_missing_assets_fail_before_creating_outputs(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'clip.mp4').touch()
   source=root/'annotations.csv';source.write_text('video_file,start,end,speaker,text,emotion\nclip.mp4,0,1,0,hello,anger\n')
   with patch.object(runner.subprocess,'check_output',return_value='{"format":{"duration":"8"}}'),patch.object(runner,'environment_identity') as environment:
    with self.assertRaisesRegex(FileNotFoundError,'ERC checkpoint not found'):
     runner.main([str(root),'--csv',str(source),'--profile','iemocap-base','--checkpoint',str(root/'absent.pt'),'--output-root',str(root/'outputs')])
    environment.assert_not_called()
   self.assertFalse((root/'outputs').exists())
 def test_relocated_checkout_and_shadowing_cwd(self):
  import shutil
  import subprocess
  repo=Path(runner.__file__).parents[2]
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);checkout=root/'extracted ERC';checkout.mkdir()
   for name in ['aimara','unimodal','multimodal','scripts']:
    shutil.copytree(repo/name,checkout/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
   shadow=root/'caller';(shadow/'aimara').mkdir(parents=True)
   (shadow/'aimara/__init__.py').write_text('raise RuntimeError("Wrong checkout imported")\n')
   env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
   for profile in runner.PROFILES:
    completed=subprocess.run([sys.executable,'-B',str(checkout/'scripts/run_optimized_evaluation.py'),'--profile',profile,'--use-predefined-csv','--dry-run','./videos'],cwd=shadow,env=env,capture_output=True,text=True)
    self.assertEqual(completed.returncode,0,completed.stderr)
    plan=json.loads(completed.stdout)
    self.assertTrue(plan['output'].startswith(str(checkout)))
    self.assertTrue(all(stage['command'][1]=='-m' for stage in plan['stages']))
   self.assertFalse((checkout/'outputs').exists())

class SparseLandmarkTests(unittest.TestCase):
 def test_sparse_inference_without_external_checkout_or_bfm_decoder(self):
  import pickle
  import types
  from aimara.optimized_evaluation.landmarks import SparseLandmarkModel
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   bfm=dict(keypoints=np.arange(204),u=np.arange(204,dtype=np.float32).reshape(-1,1),w_shp=np.zeros((204,40),np.float32),w_exp=np.zeros((204,10),np.float32))
   normalization=dict(mean=np.zeros(62,np.float32),std=np.ones(62,np.float32))
   for name,value in [('bfm.pkl',bfm),('normalization.pkl',normalization)]:
    with (root/name).open('wb') as handle:pickle.dump(value,handle)
   params=np.zeros(62,np.float32);params[:12]=np.concatenate((np.eye(3),np.zeros((3,1))),axis=1).flatten()
   class Session:
    def __init__(self,*a,**kw):pass
    def get_inputs(self):return [types.SimpleNamespace(name='input')]
    def run(self,_,inputs):
     if inputs['input'].shape!=(1,3,120,120):raise AssertionError('Bad preprocessing')
     return [params[None,:]]
   with patch.dict(sys.modules,onnxruntime=types.SimpleNamespace(InferenceSession=Session)):
    model=SparseLandmarkModel(root/'weights.onnx',root/'bfm.pkl',root/'normalization.pkl')
    landmarks=model.extract(np.zeros((100,120,3),np.uint8))
   self.assertEqual(landmarks.shape,(68,3));self.assertTrue(np.isfinite(landmarks).all())
   np.testing.assert_allclose(landmarks[30],0,atol=1e-6)
   self.assertAlmostEqual(float(np.linalg.norm(landmarks[36]-landmarks[45])),1,places=5)
