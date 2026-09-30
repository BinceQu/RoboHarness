import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.report_validation import collect, main
from roboharness.runner import process_start


class ValidationReport(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = {'task': 'task01', 'task_name': 'picking_up_trash'}
        plan = {'task_config': self.task, 'model': 'test', 'harness': 'claude_code',
                'cases': [{'instance_id': iid, 'reference_q': 1.0} for iid in (301, 304)]}
        (self.root / 'plan.json').write_text(json.dumps(plan))
        self.summary = {'status': 'running', 'cases': []}
        (self.root / 'output/json').mkdir(parents=True)

    def add_case(self, iid, q):
        path = self.root / f'output/json/picking_up_trash_{iid}_0.json'
        path.write_text(json.dumps({'task': self.task['task_name'], 'instance_id': iid,
            'rollout_id': 0, 'q_score': {'final': q}, 'steps': 20, 'success': q == 1}))
        self.summary['cases'].append({'instance_id': iid, 'result': str(path.relative_to(self.root)),
                                     'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return path

    def save(self):
        (self.root / 'summary.json').write_text(json.dumps(self.summary))

    def record_local_processes(self, owner_pid=None):
        owner_pid = os.getpid() if owner_pid is None else owner_pid
        identity = {'pid': os.getpid(), 'start': process_start(os.getpid())}
        (self.root / 'processes.json').write_text(json.dumps({
            role: identity for role in ('guardian', 'interface', 'gate', 'evaluator')}))
        (self.root / 'active_session.json').write_text(json.dumps({
            'owner_pid': owner_pid, 'owner_start': process_start(owner_pid)}))

    def require_match_exit_code(self):
        with mock.patch('sys.argv', ['report_validation', str(self.root),
                        '--output', str(self.root / 'report'), '--require-match']):
            return main()

    def test_partial_score_is_not_a_reproduction_claim(self):
        plan_path = self.root / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['cases'][1]['reference_q'] = 0.0
        plan_path.write_text(json.dumps(plan))
        self.add_case(301, 5 / 9)
        self.save()
        row = collect(self.root)
        self.assertEqual(row['n_finished'], 1)
        self.assertEqual(row['mean_q'], 5 / 9)
        self.assertEqual(row['archive_mean_q'], 0.5)
        self.assertEqual(row['completed_archive_mean_q'], 1.0)
        self.assertIsNone(row['matches_archive_mean'])
        self.assertIsNone(row['delta_archive_mean_q'])
        self.assertEqual(self.require_match_exit_code(), 2)

    def test_complete_requires_every_selected_case(self):
        self.add_case(301, 1.0)
        self.summary['status'] = 'complete'
        self.save()
        with self.assertRaises(ValueError):
            collect(self.root)
        self.add_case(304, 0.0)
        self.save()
        row = collect(self.root)
        self.assertEqual(row['mean_q'], 0.5)
        self.assertEqual(row['delta_archive_mean_q'], -0.5)
        self.assertFalse(row['matches_archive_mean'])

    def test_modified_official_result_is_rejected(self):
        path = self.add_case(301, 1.0)
        self.save()
        body = json.loads(path.read_text())
        body['q_score']['final'] = 0.0
        path.write_text(json.dumps(body))
        with self.assertRaises(ValueError):
            collect(self.root)

    def test_local_check_tracks_controller_death_without_altering_saved_results(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            self.record_local_processes(child.pid)
            self.add_case(301, 1.0)
            self.save()
            original = (self.root / 'summary.json').read_bytes()
            row = collect(self.root, check_live=True)
            self.assertEqual(row['local_liveness']['state'], 'alive')
            self.assertEqual(row['status'], 'running')
            child.kill()
            child.wait(timeout=5)
            row = collect(self.root, check_live=True)
            self.assertEqual(row['status'], 'interrupted')
            self.assertEqual(row['reported_status'], 'running')
            self.assertEqual(row['local_liveness']['failed_roles'], ['controller'])
            self.assertEqual(row['n_finished'], 1)
            self.assertEqual(row['mean_q'], 1.0)
            self.assertFalse(row['reproduction_verified'])
            self.assertEqual((self.root / 'summary.json').read_bytes(), original)
            with mock.patch('sys.argv', ['report_validation', str(self.root),
                            '--output', str(self.root / 'report'), '--watch', '--check-live', '--require-match']):
                self.assertEqual(main(), 1)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_local_check_rejects_reused_pid_for_a_required_service(self):
        self.record_local_processes()
        path = self.root / 'processes.json'
        data = json.loads(path.read_text())
        data['evaluator'] = {'pid': os.getpid(), 'start': '0'}
        path.write_text(json.dumps(data))
        self.save()
        row = collect(self.root, check_live=True)
        self.assertEqual(row['status'], 'interrupted')
        self.assertEqual(row['local_liveness']['processes']['evaluator']['state'], 'pid_reused')
        self.assertEqual(row['local_liveness']['failed_roles'], ['evaluator'])

    def test_liveness_is_optional_for_copied_runs_and_unverified_without_metadata(self):
        self.save()
        self.assertIsNone(collect(self.root)['local_liveness'])
        row = collect(self.root, check_live=True)
        self.assertEqual(row['local_liveness']['state'], 'unverified')
        self.assertEqual(row['status'], 'running')

    def test_local_check_does_not_require_agent_after_normal_exit(self):
        self.record_local_processes()
        path = self.root / 'processes.json'
        data = json.loads(path.read_text())
        data['agent'] = {'pid': os.getpid(), 'start': '0'}
        path.write_text(json.dumps(data))
        self.save()
        row = collect(self.root, check_live=True)
        self.assertEqual(row['status'], 'running')
        self.assertEqual(row['local_liveness']['state'], 'alive')

    def test_finished_runs_do_not_require_live_processes(self):
        self.add_case(301, 1.0)
        self.add_case(304, 1.0)
        self.summary['status'] = 'complete'
        self.save()
        with mock.patch('scripts.report_validation.local_liveness', side_effect=AssertionError):
            self.assertTrue(collect(self.root, check_live=True)['reproduction_verified'])

    def test_completion_race_is_not_reported_as_process_interruption(self):
        self.add_case(301, 1.0)
        self.add_case(304, 1.0)
        self.save()
        def completed_during_check(path):
            self.summary['status'] = 'complete'
            self.save()
            return {'state': 'dead', 'failed_roles': ['evaluator'], 'processes': {}}
        with mock.patch('scripts.report_validation.local_liveness', side_effect=completed_during_check):
            row = collect(self.root, check_live=True)
        self.assertEqual(row['status'], 'complete')
        self.assertTrue(row['reproduction_verified'])

    def test_equal_means_do_not_hide_swapped_case_scores(self):
        plan_path = self.root / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['cases'][1]['reference_q'] = 0.0
        plan_path.write_text(json.dumps(plan))
        self.add_case(301, 0.0)
        self.add_case(304, 1.0)
        self.summary['status'] = 'complete'
        self.save()
        row = collect(self.root)
        self.assertTrue(row['matches_archive_mean'])
        self.assertFalse(row['matches_archive_cases'])
        self.assertFalse(row['reproduction_verified'])
        self.assertEqual(self.require_match_exit_code(), 2)

    def test_exact_directory_scores_override_conflicting_raw_references(self):
        plan_path = self.root / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['cases'][0]['archive_reported_q'] = 0.0
        plan_path.write_text(json.dumps(plan))
        self.add_case(301, 0.0)
        self.add_case(304, 1.0)
        self.summary['status'] = 'complete'
        self.save()
        row = collect(self.root)
        self.assertTrue(row['matches_archive_cases'])
        self.assertTrue(row['reproduction_verified'])
        self.assertEqual(row['cases'][0]['raw_reference_q'], 1.0)
        self.assertEqual(self.require_match_exit_code(), 0)

    def test_diagnostic_plan_never_becomes_verified_reproduction(self):
        plan_path = self.root / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['diagnostic_only'] = True
        plan_path.write_text(json.dumps(plan))
        self.add_case(301, 1.0)
        self.add_case(304, 1.0)
        self.summary['status'] = 'complete'
        self.save()
        row = collect(self.root)
        self.assertIsNone(row['matches_archive_cases'])
        self.assertIsNone(row['matches_archive_mean'])
        self.assertFalse(row['reproduction_verified'])

    def test_wall_timeout_does_not_pass_even_when_official_scores_match(self):
        self.add_case(301, 1.0)
        self.add_case(304, 1.0)
        self.summary['cases'][0]['finish_reason'] = 'wall_timeout'
        self.summary['status'] = 'complete'
        self.save()
        row = collect(self.root)
        self.assertEqual(row['mean_q'], 1.0)
        self.assertIsNone(row['matches_archive_cases'])
        self.assertIsNone(row['matches_archive_mean'])
        self.assertFalse(row['reproduction_verified'])
        self.assertEqual(row['reproduction_caveats'][0]['instance_id'], 301)
        self.assertIn('wall-clock', row['reproduction_caveats'][0]['reason'])
        self.assertEqual(self.require_match_exit_code(), 2)

    def test_superseded_scores_remain_visible_without_reproduction_claim(self):
        self.add_case(301, 1.0)
        self.add_case(304, 1.0)
        self.summary['status'] = 'complete'
        self.save()
        (self.root / 'superseded_context.json').write_text(json.dumps({
            'reason': 'Different model-visible context', 'replacement_run': 'corrected'}))
        row = collect(self.root)
        self.assertEqual(row['n_finished'], 2)
        self.assertEqual(row['mean_q'], 1.0)
        self.assertEqual(row['superseded']['replacement_run'], 'corrected')
        self.assertIsNone(row['matches_archive_mean'])
        self.assertIsNone(row['delta_archive_mean_q'])

    def test_matching_scores_do_not_hide_different_archived_starting_state(self):
        plan_path = self.root / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['cases'][0]['reproduction_caveats'] = ['Archived scene snapshot unavailable']
        plan_path.write_text(json.dumps(plan))
        self.add_case(301, 1.0)
        self.add_case(304, 1.0)
        self.summary['status'] = 'complete'
        self.save()
        row = collect(self.root)
        self.assertEqual(row['mean_q'], 1.0)
        self.assertEqual(row['reproduction_caveats'][0]['instance_id'], 301)
        self.assertIsNone(row['matches_archive_mean'])
        self.assertIsNone(row['delta_archive_mean_q'])
