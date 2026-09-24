"""Select a small deterministic label/mask pilot without reading model scores.

Geometry matching is a candidate filter, not proof of matched pose or exact
value. All selected edits require the blinded image review before interpretation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def geometry(mask):
    y,x = np.where(mask)
    if not len(y): return None
    # Match normalized area, aspect and position; not a learned pose estimator.
    return np.array([np.log(mask.mean()), np.log((x.max()-x.min()+1)/(y.max()-y.min()+1)),
                     x.mean()/mask.shape[1], y.mean()/mask.shape[0]])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation","images","masks","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--per-group",type=int,default=3)
    args=ap.parse_args()
    if args.per_group < 1: raise ValueError("per-group must be positive")
    from cub_koh_spatial_audit import released_mask_index, koh_center_crop_mask
    from cub70_parts import ATTRIBUTE_TYPE_TO_MASK, COARSE_TO_CUB70
    # Explicit column selection ensures prediction values never enter selection.
    labels=pd.read_parquet(args.evaluation,columns=["image","concept_name","gt_label"])
    labels["stem"]=labels.image.map(lambda p:Path(str(p)).stem)
    labels["family"]=labels.concept_name.str.split("::").str[0]
    if labels.duplicated(["stem","concept_name"]).any(): raise ValueError("Ambiguous image/concept identity")
    images={}
    for p in sorted(args.images.rglob("*")):
        if p.suffix.lower() in (".jpg",".jpeg",".png"):
            if p.stem in images: raise ValueError("Ambiguous photo stem: "+p.stem)
            images[p.stem]=p.resolve()
    masks=released_mask_index(args.masks)
    geometry_cache={}
    def eligible(stem,part):
        key=(stem,part)
        if key in geometry_cache: return geometry_cache[key]
        if key not in masks or stem not in images: return None
        mask=np.asarray(Image.open(masks[key]).convert("L"))>0
        if Image.open(images[stem]).size != mask.shape[::-1]: raise ValueError("Image/mask grid mismatch")
        if not mask.any() or mask.all(): return None
        # Candidate selection keeps at least half the annotated part in model view.
        # This is disclosed pilot engineering, not a scientific acceptance gate.
        if koh_center_crop_mask(mask).sum() < .5*mask.sum(): return None
        geometry_cache[key]=geometry(mask)
        return geometry_cache[key]
    plan=[]; coverage=[]
    for group,parts in COARSE_TO_CUB70.items():
        count=0
        for family in sorted(k for k,v in ATTRIBUTE_TYPE_TO_MASK.items() if v==group):
            subset=labels[labels.family==family]
            positives=subset[subset.gt_label==1]
            n=positives.groupby("stem").size()
            positives=positives[positives.stem.isin(n[n==1].index)].sort_values("stem")
            for source in positives.itertuples():
                if count>=args.per_group: break
                for part in parts:
                    sg=eligible(source.stem,part)
                    if sg is None: continue
                    candidates=[]
                    for donor in positives.itertuples():
                        if donor.stem==source.stem: continue
                        dg=eligible(donor.stem,part)
                        if dg is not None:
                            candidates.append((float(np.linalg.norm(sg-dg)),donor.stem,donor.concept_name))
                    same=sorted(c for c in candidates if c[2]==source.concept_name)
                    different=sorted(c for c in candidates if c[2]!=source.concept_name)
                    if not same or not different: continue
                    _,s,_=same[0]; _,d,dc=different[0]
                    trial=dict(trial_id=f"{group}_{count+1:02d}",mask_group=group,fine_part=part,
                        attribute_type=family,source_concept=source.concept_name,donor_concept=dc,
                        selection="label-unique within family; geometry matched; no model scores",
                        source_image=str(images[source.stem]),source_mask=str(masks[(source.stem,part)].resolve()),
                        same_value_image=str(images[s]),same_value_mask=str(masks[(s,part)].resolve()),
                        different_value_image=str(images[d]),different_value_mask=str(masks[(d,part)].resolve()),
                        same_geometry_distance=same[0][0],different_geometry_distance=different[0][0],
                        status="METHOD NOT CALIBRATED")
                    plan.append(trial); count+=1; break
            if count>=args.per_group: break
        coverage.append(dict(group=group,selected=count,requested=args.per_group))
    if not plan: raise ValueError("No candidate triplets; inspect image/mask/label identity coverage")
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    (args.out/"SELECTION.json").write_text(json.dumps(dict(training=False,
        scores_used=False,coverage=coverage,selection_bias="small deterministic convenience pilot; not population prevalence",
        limitations="geometry is not pose; coarse masks may change other attributes; review required"),indent=2)+"\n")
    print(f"Selected {len(plan)} candidate triplets across {sum(c['selected']>0 for c in coverage)} groups; no training")


if __name__=="__main__": main()
