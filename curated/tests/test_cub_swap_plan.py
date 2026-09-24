import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image


class PlanTests(unittest.TestCase):
    def test_end_to_end_plan_and_four_image_edits_without_logits(self):
        analysis=Path(__file__).resolve().parents[1]/"analysis"
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"images").mkdir(); (root/"masks"/"class").mkdir(parents=True)
            rows=[]
            for i in range(4):
                stem=f"bird_{i}"; image=np.zeros((50,60,3),np.uint8)+90
                mask=np.zeros((50,60),np.uint8); mask[20:30,20:35]=255
                image[mask>0]=[200,20,20] if i<2 else [20,20,200]
                Image.fromarray(image).save(root/"images"/(stem+".png"))
                Image.fromarray(mask).save(root/"masks"/"class"/(stem+"_tail.png"))
                for j,value in enumerate(("red","blue")):
                    rows.append(dict(image=stem+".png",concept_name="has_upper_tail_color::"+value,
                                     gt_label=int(j==(i//2))))
            pd.DataFrame(rows).to_parquet(root/"labels.parquet")
            subprocess.run([sys.executable,str(analysis/"cub_swap_plan.py"),"--evaluation",str(root/"labels.parquet"),
                "--images",str(root/"images"),"--masks",str(root/"masks"),"--out",str(root/"selection"),
                "--per-group","1"],check=True,capture_output=True,text=True)
            plan=json.loads((root/"selection"/"plan.json").read_text())
            self.assertEqual(len(plan),1)
            subprocess.run([sys.executable,str(analysis/"cub_mask_swap_v2.py"),"--plan",str(root/"selection"/"plan.json"),
                "--out",str(root/"edits")],check=True,capture_output=True,text=True)
            edit=json.loads((root/"edits"/"tail_01"/"EDIT.json").read_text())
            self.assertEqual(edit["edits"]["self"]["changed_pixels"],0)
            self.assertGreater(edit["edits"]["different_value"]["changed_pixels"],0)
            self.assertFalse(edit["model_scores_used_for_selection"])
            from export_cub_swap_review import export
            export(root, root/'review.html')
            rendered=(root/'review.html').read_text()
            self.assertIn('data:image/png;base64,',rendered)
            self.assertIn('Example 1: tail_01',rendered)
            self.assertNotIn('z_source',rendered)


if __name__=="__main__": unittest.main()
