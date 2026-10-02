# Concept backwash in concept bottleneck models

I study whether a concept bottleneck model predicts a named attribute from the corresponding part of an image. The model first predicts concept scores, then uses them to classify species. In FunnyBird, replacing a part can increase the inserted variant’s score while leaving the removed variant’s score higher. I refer to this response pattern as **concept backwash**.

The investigation asks whether this behavior is associated with species information in concept scores, whether restricting the representation reduces it, and whether visibility-aware training labels change it. The strongest current finding is that relabeling substantially improves tail replacement outcomes. Stronger minimality alone worsens those outcomes in the evaluated MCBM runs. Extending the experiment to natural CUB70 photographs remains limited by mask quality and intervention validity.

## FunnyBird: why compare tail and wing?

All five parts—tail, wing, beak, foot, and eye—are evaluated. **Tail has the highest backwash frequency in the Standard CBM; wing has the lowest.** Their contrast helps distinguish explanations for the part-specific failure. Tail also has substantially more positive training labels associated with invisible parts. Wing provides a comparison with strong replacement responses and very little of this label conflict.

The Standard model is a jointly trained Koh CBM with a ResNet-50 encoder, 26 raw concept logits, and a linear 50-species head. Ordinary test accuracy is 99.2% for species and 99.68% for concepts. The renderer then replaces one part while retaining the remaining scene parameters, giving 1,000 swaps per part.

Let the margin be the inserted donor logit minus the removed source logit. Positive values favor the donor. Its decomposition is:

**Final margin = original margin + donor-score increase + source-score decrease.**

| Standard CBM | Original margin | Donor increase | Source decrease | Final margin |
|---|---:|---:|---:|---:|
| Tail | −10.756 | 4.599 | 5.201 | **−0.956** |
| Wing | −9.914 | 8.029 | 8.361 | **+6.476** |

These are means in raw-logit units. Tail and wing begin with similar source preferences. Wing’s larger donor increase and source decrease reverse that preference. Tail’s combined change is insufficient, leaving a negative mean final margin. Individual swaps count as backwash when the margin moves donorward and remains negative.

[Dataset analysis](01_funnybirds_analysis/review.md) · [Standard CBM experiments](02_funnybirds_cbm/review.md)

## Minimality: the tail representation becomes more label-aligned and less responsive to replacement

The analysis first compares the Standard CBM with MCBM at γ=0, then increases γ within MCBM. **At γ=0, exact donor recognition is already lower:** tail falls from 39.5% to 27.8%, and wing from 97.7% to 81.1%. This first comparison changes architecture and training components together; attributing the difference to a specific component requires further ablation.

MCBM penalizes departures of internal concept coordinates from label-dependent targets. Increasing the penalty weight γ from 0 to 5 reduces tail target error from 24.40 to 0.54. The table shows both comparison stages:

| Tail model | Original margin | Donor increase | Source decrease | Final margin |
|---|---:|---:|---:|---:|
| Standard CBM | −10.756 | 4.599 | 5.201 | **−0.956** |
| MCBM γ = 0 | −21.291 | 6.558 | 11.645 | **−3.088** |
| MCBM γ = 5 | −11.732 | 1.450 | 4.822 | **−5.460** |

The initial deficit becomes smaller, while the replacement response falls even more. **The final margin becomes more negative, and tail backwash increases from 52.2% to 73.9%.** The weaker response is already evident before the private concept readers. In these runs, stronger minimality improves ordinary target agreement while worsening the tail’s response to changed pixels. Other parts differ: foot’s exact donor recognition improves across the same endpoints. For tail, the Standard-to-γ=0 transition produces a larger starting deficit and a more negative final margin. Increasing γ then weakens the replacement response further, despite improving ordinary target agreement.

[MCBM experiments](03_funnybirds_mcbm/review.md)

## Relabeling: the removed tail score decreases more after replacement

Visibility-aware relabeling, called RLv2, changes positive targets when the renderer identifies the corresponding part as invisible. It changes approximately 19.9% of positive tail labels and 0.032% of positive wing labels. Both trained conditions are evaluated on identical replacement images.

| Tail comparison | Starting-margin improvement | Additional donor increase | Additional source decrease | Final margin before → after |
|---|---:|---:|---:|---:|
| Standard CBM → RLv2 | +0.990 | +1.000 | +0.869 | **−0.956 → +1.904** |
| MCBM γ=5 → RLv2 γ=5 | +0.653 | +1.423 | +4.415 | **−5.460 → +1.032** |

**Relabeling reverses the mean tail margin in both comparisons.** At MCBM γ=5, the largest component is the greater decrease in the removed source score; backwash falls from 73.9% to 34.7%. This supports a contribution from visibility-related supervision to source retention. The improvement remains incomplete and part-specific: wing exact donor recognition decreases from 83.2% to 77.6% in the same MCBM comparison.

Within MCBM, the γ=0→5 comparison changes direction after relabeling: tail backwash rises from 52.2% to 73.9% under Standard labels and falls from 44.8% to 34.7% under RLv2. This endpoint interaction supports further evaluation of visibility-aware labels with minimality. Exact donor recognition under RLv2 changes only from 42.5% to 42.8%, and intermediate γ values are nonmonotonic, so the broader improvement remains limited.

[Standard CBM relabeling](02rl_funnybirds_cbm_relabeled/review.md) · [MCBM relabeling](03rl_funnybirds_mcbm_relabeled/review.md)

## Species information: does it account for tail–wing differences?

One proposed explanation is that concept scores encode species identity, allowing the original bird’s identity to influence the predicted part. A diagnostic classifier can recover additional species information from both tail and wing scores beyond their binary labels. **Wing nevertheless has strong replacement responses.** Species decodability alone therefore gives an incomplete account of the tail–wing difference.

I also replace within-label score variation by the corresponding mean and rerun the saved species head. Wing replacement leaves all top-one predictions unchanged in the evaluated Standard model. After relabeling, tail replacement outcomes improve while its additional species-predictive information remains. Together, these observations limit the hypothesis that removing residual species information is necessary for the observed improvement. The specific visual cues responsible for retained source scores remain unresolved.

[Species diagnostics and classifier interventions](02_funnybirds_cbm/review.md)

## CUB70: mask limitations and the current intervention work

CUB70 provides photographs and released anatomical masks, with incomplete correspondence between attributes, visible regions, and annotations. Neck masks, for example, can be unavailable even when the neck is visible. This complicates both visibility comparisons and the construction of controlled part replacements. Label preprocessing can also replace image-level variation with species-level targets.

The existing analysis finds label-associated score differences among images whose mapped masks are absent. These observations are compatible with contextual prediction, although visible regions omitted from the masks and differences in species or pose remain plausible explanations.

I investigated matched species recall gaps as a possible proxy for replacement behavior. **Calibration against FunnyBird’s controlled swaps was insufficient:** concepts with larger recall gaps did not consistently have higher backwash rates, particularly within parts. CUB currently lacks the validated swap reference needed for an equivalent calibration.

I am currently adapting the segmentation approach in [From Segments to Concepts: Interpretable Image Classification via Concept-Guided Segmentation](https://arxiv.org/abs/2510.04180v1) to identify regions for mask-based donor replacements. The aim is to construct a crude proxy for the FunnyBird swap. The paper supplies the segmentation approach; the replacement procedure is being developed and requires validation of donor evidence and changes to surrounding pixels.

[CUB labels and preprocessing](04_cub_analysis/review.md) · [CUB70 analysis and methods](05_cub_cbm/review.md)

## Remaining questions

The FunnyBird results connect visibility-aware supervision to improved tail responses and show that a stronger minimality penalty can worsen those responses. Further controlled comparisons of target pixels and surrounding parts are needed to identify the visual source of the retained preference. Independent training replication remains necessary: the principal Standard comparison uses seed 1, and historical MCBM initialization was uncontrolled.
