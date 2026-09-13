# FunnyBird four-condition pilot

This is frozen inference on the accepted Koh Joint Standard CBM. It performs no training.

Here `context` has one narrow meaning: the four other named FunnyBird part meshes. Removing them does not remove the base body, pose, camera, lighting, or background. This pilot therefore tests dependence on the other named parts conditional on all of that retained information; it does not claim to erase every species/body cue.

The two target-absent native renders are `01` (other parts present) and `00` (other parts absent). We take the visible target pixels and mask from the native `11` render, then apply the same paste operation twice: onto `01` to make scientific `11`, and onto `00` to make scientific `10`. Thus `11` and `10` contain byte-identical target pixels. The native `11` and native `10` renders are saved only for calibration; neither enters the four-cell formulas.

The five paired quantities are:

- part response with context: `z11 - z01`;
- part response without context: `z10 - z00`;
- context evidence with the part: `z11 - z10`;
- context evidence without the part: `z01 - z00`;
- interaction: `(z11 - z01) - (z10 - z00)`.

Example: if the four target scores are `z11=6`, `z01=4`, `z10=1`, and `z00=-3`, the target-pixel response is `6-4=2` with the other parts and `1-(-3)=4` without them. The other-part contribution is `6-1=5` when the target pixels are present and `4-(-3)=7` when they are absent. The interaction is `2-4=-2`: in this image, adding the other named parts makes the model two raw-logit units less responsive to the unchanged target pixels.

A row is mechanically valid only if the native `11` target mask contains at least 8 pixels, the target is absent from both base renders, both pastes visibly change their base, and removing the four other named parts changes at least 8 pixels. These checks prevent empty or unchanged renders from entering the summaries. They do not prove that the composites look natural.

`audit_A1_native_vs_composite.png` and the native/composite difference columns in `rows.csv` are the calibration evidence. The run is only a pilot awaiting visual review. It is not accepted scientific evidence merely because the mechanical checks pass.

## Counts

```text
part  selected_rows  eligible_rows  excluded_rows  median_z11  mean_z11  median_z01  mean_z01  median_z10  mean_z10  median_z00  mean_z00  median_part_response_with_context  mean_part_response_with_context  median_part_response_without_context  mean_part_response_without_context  median_context_evidence_with_part  mean_context_evidence_with_part  median_context_evidence_without_part  mean_context_evidence_without_part  median_interaction  mean_interaction  fraction_context_evidence_without_part_positive  fraction_part_response_with_context_positive
tail             25             20              5    5.040865  4.792956    0.266872  0.169602    4.679763  3.551278   -5.014076 -5.165440                           4.889818                         4.623354                              9.706779                            8.716719                           0.297999                         1.241678                              5.238580                            5.335042           -3.955029         -4.093365                                             1.00                                           1.0
wing             25             25              0    4.706493  4.801859   -1.365874 -1.734059    6.458704  6.225889   -4.727709 -4.675347                           6.273793                         6.535918                             10.765344                           10.901236                          -1.676879                        -1.424030                              3.285804                            2.941288           -4.167651         -4.365318                                             0.96                                           1.0
```

No result interpretation is prewritten here. Review the current figures and tables first.
