"""Add the chronological handoff from discovery notebook 02 to the RLv2 follow-up."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


CURATED = Path(__file__).resolve().parent.parent
NOTEBOOK = CURATED / "notebooks" / "02_funnybirds_cbm.ipynb"
MARKER = "### Follow-up: visibility-aware relabeling"

addition = dedent(
    r"""

    ### Follow-up: visibility-aware relabeling

    The training-label alternative is tested in **03rl · FunnyBirds visibility-aware labels**.
    That notebook must be read after this discovery notebook, not as a replacement for it.

    It first freezes the predictions made here:

    - the removed source concept should turn off more strongly;
    - donor-over-source ordering should improve most for tail, moderately for beak/eye, and
      little for foot/wing;
    - deletion should leave less retained concept probability;
    - variant and source-species structure may remain because relabeling does not change the
      number of variants or the body/species correlations.

    It then evaluates standard and RLv2 checkpoints on identical validated renders, decomposes
    donor and source score changes, repeats visibility and all-part variant controls, tests
    source-species residuals after matching variant/direction/visibility, and selects actual
    unexplained examples for visual inspection. The current seed-1 result is provisional until
    seeds 2–3 and the independent RLv2 deletion run are complete.
    """
).strip("\n")


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
target = None
for cell in notebook["cells"]:
    source = "".join(cell.get("source", []))
    if source.startswith("## 11") and "What is established" in source:
        target = cell
        break
if target is None:
    raise RuntimeError("Could not find notebook 02 conclusion cell")

source = "".join(target["source"])
if MARKER not in source:
    source = source.rstrip() + "\n\n" + addition + "\n"
else:
    source = source.replace("\n### Follow-up: visibility-aware relabeling",
                            "\n\n### Follow-up: visibility-aware relabeling", 1)
target["source"] = source.splitlines(keepends=True)
NOTEBOOK.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"updated {NOTEBOOK}")
