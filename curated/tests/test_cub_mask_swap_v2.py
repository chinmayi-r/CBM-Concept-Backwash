import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from cub_mask_swap_v2 import replace_part


class SwapTest(unittest.TestCase):
    def setUp(self):
        y,x=np.indices((32,32))
        self.source=np.stack([x*7,y*7,(x+y)*3],-1).astype(np.uint8)
        self.mask=np.zeros((32,32),bool);self.mask[10:20,10:20]=True

    def test_identity_reconstruction(self):
        out,_,_,m=replace_part(self.source,self.mask,self.source,self.mask)
        np.testing.assert_array_equal(out,self.source)
        self.assertEqual(m['changed_pixels'],0)

    def test_donor_shape_survives_and_context_is_unchanged(self):
        dm=np.zeros((32,32),bool);dm[6:26,13:18]=True
        donor=np.full_like(self.source,240)
        out,placed,allowed,m=replace_part(self.source,self.mask,donor,dm)
        self.assertTrue(np.any(placed & ~self.mask))
        self.assertTrue(np.any(self.mask & ~placed))
        np.testing.assert_array_equal(out[~allowed],self.source[~allowed])
        self.assertTrue(np.all(out[placed]==240))
        self.assertGreater(m['source_only_filled_pixels'],0)

    def test_anchors_rotate_without_anisotropic_shape_warp(self):
        _,_,_,m=replace_part(self.source,self.mask,self.source,self.mask,[[16,16],[16,22]],[[16,16],[22,16]])
        a=np.array(m['transform'])[:,:2]
        np.testing.assert_allclose(a.T@a,np.eye(2),atol=1e-10)

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):replace_part(self.source,self.mask,self.source,np.zeros_like(self.mask))


if __name__=='__main__':unittest.main()
