import argparse
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from behavior_interface_eval_test import prepare_submission_artifacts as bundle
from behavior_interface_eval_test import watch_submission_artifacts as watcher

class SubmissionArtifactTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.output=self.root/'output'
        for name in ('json','videos'):(self.output/name).mkdir(parents=True)
        self.metric=self.output/'json/task_301_0.json';self.metric.write_text('{"q_score": {"final": 0.5}}')
        self.video=self.output/'videos/task_301_0.mp4';self.video.write_bytes(b'finalized-video')
        self.dest=self.output/'submission/videos'

    def test_incomplete_json_is_not_mirrored(self):
        self.metric.write_text('{"q_score":')
        names=bundle._copy_files(self.metric.parent,self.output/'submission/metrics','.json')
        self.assertEqual(names,[])

    def test_open_video_is_not_published(self):
        with patch.object(bundle,'probe_video',side_effect=ValueError('moov atom not found')):
            self.assertEqual(bundle._copy_files(self.video.parent,self.dest,'.mp4'),[])
        self.assertFalse((self.dest/self.video.name).exists())

    def test_final_video_is_copied_verbatim_and_old_mirror_is_repaired(self):
        self.dest.mkdir(parents=True);target=self.dest/self.video.name;target.write_bytes(b'partial')
        with patch.object(bundle,'probe_video',return_value={}):
            self.assertEqual(bundle._copy_files(self.video.parent,self.dest,'.mp4'),[self.video.name])
        self.assertEqual(target.read_bytes(),self.video.read_bytes())
        self.assertEqual(list(self.dest.glob('*.tmp')),[])

    def test_source_changing_during_copy_is_not_published(self):
        original=shutil.copy2
        def changing(source,target):
            result=original(source,target);source.write_bytes(source.read_bytes()+b'new-frame');return result
        with patch.object(bundle,'probe_video',return_value={}),patch.object(bundle.shutil,'copy2',side_effect=changing):
            self.assertEqual(bundle._copy_files(self.video.parent,self.dest,'.mp4'),[])
        self.assertFalse((self.dest/self.video.name).exists())
        self.assertEqual(list(self.dest.glob('*.tmp')),[])

    def test_launch_inputs_are_preserved_and_invalid_legacy_video_is_not_zipped(self):
        wrapper=self.root/'wrapper.py';wrapper.write_text('original wrapper')
        robot=self.root/'robot.yaml';robot.write_text('original robot')
        args=argparse.Namespace(output_dir=self.output,submission_dir=None,task='task',task_index=10,
            scene='scene',port=15010,policy_port=28010,gpu=0,timeout_version='2026-1.5x',
            timeout_steps=100,prompt_path=str(self.root/'absent.txt'),finalize=False)
        with patch.object(bundle,'WRAPPER',wrapper),patch.object(bundle,'ROBOT_CONFIG',robot),patch.object(bundle,'probe_video',return_value={}):
            bundle.prepare(args)
        wrapper.write_text('changed after launch');robot.write_text('changed after launch');args.finalize=True
        with patch.object(bundle,'WRAPPER',wrapper),patch.object(bundle,'ROBOT_CONFIG',robot),patch.object(bundle,'probe_video',side_effect=ValueError('writer open')):
            manifest=bundle.prepare(args)
        self.assertEqual(manifest['pending_artifacts']['videos'],[self.video.name])
        submission=self.output/'submission'
        self.assertEqual((submission/'wrapper/policy_wrapper.py').read_text(),'original wrapper')
        self.assertEqual((submission/'robot/robot_config.yaml').read_text(),'original robot')
        with zipfile.ZipFile(self.output/'submission.zip')as archive:
            self.assertNotIn('videos/'+self.video.name,archive.namelist())
            self.assertIn('metrics/'+self.metric.name,archive.namelist())

    def test_watcher_requests_retry_for_unvalidated_artifacts(self):
        args=argparse.Namespace(output_dir=self.output,submission_dir=self.output/'submission',task='task',
            task_index=10,scene='scene',port=15010,policy_port=28010,gpu=0,timeout_version='2026-1.5x',
            timeout_steps=100,prompt_path=None,status_path=None)
        result=argparse.Namespace(returncode=0,stdout=json.dumps({'pending_artifacts':{'videos':['open.mp4']}}),stderr='')
        with patch.object(watcher.subprocess,'run',return_value=result):
            self.assertFalse(watcher._sync(args))

if __name__=='__main__':unittest.main()
