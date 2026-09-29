import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_validation import collect


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

    def test_partial_score_is_not_a_reproduction_claim(self):
        self.add_case(301, 1.0)
        self.save()
        row = collect(self.root)
        self.assertEqual(row['n_finished'], 1)
        self.assertEqual(row['mean_q'], 1.0)
        self.assertIsNone(row['matches_archive_mean'])
        self.assertIsNone(row['delta_archive_mean_q'])

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
