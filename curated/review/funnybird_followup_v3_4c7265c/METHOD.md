# FunnyBird Standard-CBM focused follow-ups

These are read-only, post-hoc diagnostics on the accepted seed-1 Koh Joint
ResNet-50 model and its accepted 5,000 fixed-render swaps. They do not train or
alter a CBM and do not assign causal percentages.

## Follow-up 1: information available versus information used

Panel A compares a species probe given only the official 0/1 labels with the
same probe also given each raw score after subtracting its training-fold mean
for that label. Positive held-out log-loss gain means raw magnitudes reveal
species information beyond the 0/1 concepts. Panel B gives every part exactly
three coordinates; vertical lines are the range across coordinate subsets, not
uncertainty bars. Panel C/D replace raw magnitudes by label-conditioned means
and pass them through the unchanged saved `Wz+b` head. This separates what a
new probe can recover from what the CBM's own class head uses.

## Follow-up 2: off-target source evidence during swaps

For each swap, remove the old-value and inserted-value coordinates from the
replaced part block. Center every remaining logit by its ordinary absent-label
mean, then multiply by the saved source-class minus donor-class weights. A
positive number is direct source-over-donor class-logit evidence used by the
saved head. Both evidence and final concept margin are centered within the same
exact old-to-new value pair before association is measured. This is a weak
mechanism test, not a causal intervention on the fingerprint itself.

## Follow-up 3: label–visibility conflict and matched response components

Conflict is the fraction of Standard positive training/validation labels that
RLv2 changes to zero because the named part is not visible. Each exact value is
one plotted point. The inserted-value conflict is compared with how much that
inserted logit rises; removed-value conflict is compared with how much the old
logit falls. Point size is species support and the printed numeral is the exact
value. The causal test is the matched Standard-versus-RLv2 replay in notebook
02rl, not this association.

## Follow-up 4: held-out predictability, not causal contribution

Five-fold evaluation keeps every swap from an original image together. The full
diagnostic uses part, starting margin, visible area, source/donor support,
source/donor label conflict, ordinary exact-value recognition, and source
species. Each bar is how much held-out error changes when one entire family is
removed. Correlated families can substitute for each other, so bars are not
causal percentages and do not transfer automatically to CUB. The separate
value-holdout table is only a stress test for a new exact value inside this
FunnyBird system.
