import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from cub_mask_inventory import inventory


def archive(path, records):
    with tarfile.open(path, "w:xz") as target:
        for name, values in records:
            buf = io.BytesIO()
            Image.fromarray(np.asarray(values, dtype=np.uint8)).save(buf, format="PNG")
            data = buf.getvalue()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))


class InventoryTests(unittest.TestCase):
    def test_missing_is_distinct_from_empty_and_overlap_counted_once(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "masks.tar.xz"
            archive(path, [("1/bird_head.png", [[255, 255], [0, 0]]),
                           ("1/bird_beak.png", [[255, 0], [0, 0]]),
                           ("1/bird_tail.png", [[0, 0], [0, 0]])])
            _, states, summary, manifest = inventory(path)
            state = {r["part"]: r["annotation_state"] for r in states}
            self.assertEqual(state["tail"], "file_empty")
            self.assertEqual(state["left_eye"], "file_missing")
            self.assertEqual(state["head"], "file_nonempty")
            self.assertEqual(manifest["overlapping_pixels"], 1)
            self.assertFalse(manifest["physical_visibility_verified"])
            self.assertEqual(sum(r["file_nonempty"] for r in summary), 2)

    def test_duplicate_and_mismatched_geometry_rejected(self):
        for records in [ [("1/b_head.png", [[255]]), ("1/b_head.png", [[0]])],
                         [("1/b_head.png", [[255]]), ("1/b_tail.png", [[0, 255]])] ]:
            with self.subTest(records=records), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "masks.tar.xz"
                archive(path, records)
                with self.assertRaises(ValueError):
                    inventory(path)


if __name__ == "__main__":
    print("PREFLIGHT ONLY - NO SCIENTIFIC TRAINING")
    unittest.main()
