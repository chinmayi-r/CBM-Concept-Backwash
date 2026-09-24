from pathlib import Path
import sys
import unittest
import tempfile
import hashlib
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from cub_intervention_analysis import score_interventions, compare_relabeling, joint_logit_gradient, load_reviewed_results, REVIEWS


def fixture():
    return pd.DataFrame([dict(trial_id="one", seed=1, source_image="a", same_value_donor_image="b",
        different_value_donor_image="c", attribute_type="has_tail_color", source_concept="red",
        donor_concept="blue", mask_group="tail", source_pixels_sha256="a"*64,
        same_value_pixels_sha256="b"*64, different_value_pixels_sha256="c"*64,
        z_source_original=5., z_donor_original=-5., z_source_same_value=4., z_donor_same_value=-4.,
        z_source_different_value=2., z_donor_different_value=0., **{k: True for k in REVIEWS})])


class InterventionTests(unittest.TestCase):
    def test_blinded_review_join_and_prediction_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=fixture()
            data.to_parquet(root/'predictions_unreviewed.parquet')
            data[['trial_id','seed']+REVIEWS].to_csv(root/'blinded_review.csv',index=False)
            digest=hashlib.sha256((root/'predictions_unreviewed.parquet').read_bytes()).hexdigest()
            (root/'INFERENCE.json').write_text(json.dumps(dict(framework='koh_joint',training=False,predictions_sha256=digest)))
            scored,_=load_reviewed_results(root)
            self.assertTrue(scored.iloc[0].review_eligible)
            data['z_donor_original']=1.
            data.to_parquet(root/'predictions_unreviewed.parquet')
            with self.assertRaisesRegex(ValueError,'changed after inference'):
                load_reviewed_results(root)

    def test_response_and_control_do_not_equal_donor_victory(self):
        result = score_interventions(fixture()).iloc[0]
        self.assertEqual(result.response_delta, 8)
        self.assertEqual(result.control_adjusted_response, 6)
        self.assertTrue(result.positive_response_source_retained)
        self.assertFalse(result.donor_wins)

    def test_losing_response_is_not_improved_grounding(self):
        b = fixture()
        b['z_donor_different_value'] = -10.
        result = compare_relabeling(fixture(), b).iloc[0]
        self.assertTrue(result.retention_resolved)
        self.assertTrue(result.resolved_by_losing_response)
        self.assertFalse(result.resolved_with_donor_win)

    def test_mismatched_pixels_and_string_booleans_rejected(self):
        b = fixture()
        b['different_value_pixels_sha256'] = 'd'*64
        with self.assertRaises(ValueError): compare_relabeling(fixture(), b)
        b = fixture()
        b['target_value_reviewed'] = 'False'
        with self.assertRaises(ValueError): score_interventions(b)

    def test_joint_gradient_matches_loss_finite_difference(self):
        z = np.array([.4, -.6]); w = np.array([[.2, .8], [-.3, .1]])
        b = np.array([.1, -.2]); c = np.array([1., 0.]); q = np.array([2., 1.])
        def loss(v):
            scores = w @ v + b
            return (np.logaddexp.reduce(scores) - scores[1] + .01 * np.sum(q * (np.logaddexp(0, v) - c*v))) / 1.02
        numeric = np.array([(loss(z + np.eye(2)[j]*1e-6)-loss(z - np.eye(2)[j]*1e-6))/2e-6 for j in range(2)])
        actual = joint_logit_gradient(z,w,b,1,c,q)['total_gradient']
        np.testing.assert_allclose(actual,numeric,rtol=1e-7,atol=1e-8)


if __name__ == '__main__':
    print('PREFLIGHT ONLY - NO SCIENTIFIC TRAINING')
    unittest.main()
