"""Small synthetic tests of recovery reporting, not scientific validation."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from audit_mcbm_recovery import summarize


class RecoveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'replay_diagnostic.csv'
        self.data=pd.DataFrame(dict(render_id=['a','b'],max_score_error=[.021,.001],
            probability_error=[.003,.0],outcome_changed=[False,True],
            accepted_boundary_distance=[1.3,0.],accepted_margin=[-1.3,-11.],
            replayed_margin=[-1.34,-10.999]))

    def tearDown(self):
        self.temp.cleanup()

    def test_exceeding_coordinate_is_not_outcome_change(self):
        self.data.to_csv(self.path,index=False)
        result=summarize(self.path)
        self.assertEqual(result['above_002'],1)
        self.assertEqual(result['outcome_changes'],1)
        self.assertEqual(result['changed_boundary_max'],0.)
        self.assertFalse(result['full_replay'])

    def test_rejects_invalid_boolean(self):
        self.data['outcome_changed']=['unknown','false']
        self.data.to_csv(self.path,index=False)
        with self.assertRaises(ValueError): summarize(self.path)

    def test_rejects_duplicate_identity(self):
        self.data['render_id']=['a','a']
        self.data.to_csv(self.path,index=False)
        with self.assertRaises(ValueError): summarize(self.path)


if __name__=='__main__': unittest.main()
