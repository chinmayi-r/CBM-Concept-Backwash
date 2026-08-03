# Standard-CBM cross-dataset proof map

This map keeps the primary comparison at the correct stage:

`FunnyBird standard CBM discovery (notebook 02) -> CUB70 standard CBM test (notebook 05)`

MCBM and RLv2 come later. They must not replace this comparison.

## Required reading order and closest comparisons

| Question | FunnyBird standard CBM | Closest CUB70 standard-CBM evidence | What a matching result would mean | Present limitation |
|---|---|---|---|---|
| What labels, variants, species, and visible parts exist? | Notebook 01 plus notebook 02 setup | Notebook 05 Figures 1–5 | Establishes whether species-linked labels and hidden positive labels are available as risk factors | Data structure is not model backwash |
| Is the trained model usable? | Notebook 02 Figure 1 | Notebook 05 Figure 6 | Later concept behavior is worth studying | Necessary guard, never causal proof |
| Does the model behave normally on ordinary images? | Notebook 02 Figure 2 | Notebook 05 Figures 7 and 12, restricted to naturally visible positive concepts | Present evidence should usually support the named concept | CUB photographs are not controlled pairs |
| Does removing the named part reduce its concept? | Notebook 02 Figure 3 and clean renderer deletion | Notebook 05 failed whole-part and patch sections | Direct local-pixel reliance if the edit is calibrated | Current shared CUB deletion/patch calibration failed |
| Can we change only one part? | Notebook 02 Figures 4–5 | Notebook 05 Figure C1 insertion sheets | Required before donor/source reasoning | CUB pastes were often tiny/artificial |
| Did the new pixels move the donor-source margin? | Notebook 02 response-delta figure | Notebook 05 Figure C2 | Positive movement is required before retained-source backwash can be named | CUB pilot failed: only 40% positive and median response about zero |
| Does the old source still win after a positive response? | Notebook 02 ordering and margin figures | CUB Figure C2 only if donor response passes | This is the candidate backwash event | Not currently licensed in CUB |
| Is low visibility sufficient to explain the failures? | Notebook 02 visibility-bin and visible-only figures | Notebook 05 Figures 7, 9, 10, and 11 | Similar concentration would support visibility as a contributor | Filtering is selection; CUB groups are different photographs |
| Do exact visual values differ in difficulty? | Notebook 02 variant bars/matrix | Notebook 05 exact-concept dots, contrast, and collapsed-concept inventory | Organizes remaining failures by exact value | Observational in both unless variants are independently manipulated |
| Does source species/body organize the remainder? | Notebook 02 source-species and within-variant residual figures | Notebook 05 species-matched visibility, species probe, and controlled species residual | Identifies where a body/species intervention should be aimed | Association is not proof that species/body caused the remainder |
| Does concept behavior change the species prediction? | Notebook 02 downstream species-probability figures | No equally clean CUB counterpart yet | Measures explanatory cost at the classifier output | Must not be inferred from species decodability alone |
| What is left after the measured influences? | New notebook-02 standard-CBM residual section | New notebook-05 observational residual section | Reports the unexplained remainder without pretending factors add to 100% | The two residuals have different causal strength and cannot be numerically pooled |

## Final-claim rule

FunnyBird may support a causal final because its one-part edit and positive donor
response predicates pass. CUB70 currently cannot. CUB may be compared at every
earlier question, but its final statement remains: related observational risk
factors are present; causal CUB backwash is neither proved nor disproved.

