"""Inventory existing MCBM replay evidence without inference, fitting or mutation.

No checkpoint is loaded as executable pickle; no cache is promoted or deleted.
Pass rejected cache directories explicitly: their old diagnostics lack gamma metadata.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize(path):
    d = pd.read_csv(path)
    required = {'render_id', 'max_score_error', 'probability_error',
                'outcome_changed', 'accepted_boundary_distance',
                'accepted_margin', 'replayed_margin'}
    if required - set(d):
        raise ValueError(f'missing columns: {sorted(required-set(d))}')
    if d.render_id.duplicated().any():
        raise ValueError('duplicate render IDs')
    numeric = list(required - {'render_id', 'outcome_changed'})
    if not np.isfinite(d[numeric].to_numpy(float)).all():
        raise ValueError('non-finite diagnostic values')
    flags = d.outcome_changed.astype(str).str.lower()
    if not flags.isin(['true','false']).all():
        raise ValueError('outcome_changed is not boolean')
    changed = d.loc[flags.eq('true')]
    exceeded = d.loc[d.max_score_error > .02]
    return dict(rows=len(d), full_replay=len(d)==5000,
        max_score_error=float(d.max_score_error.max()),
        above_002=len(exceeded), probability_mean=float(d.probability_error.mean()),
        probability_max=float(d.probability_error.max()),
        outcome_changes=len(changed),
        changed_boundary_max=float(changed.accepted_boundary_distance.max()) if len(changed) else 0.,
        changed_zero_boundary=int(changed.accepted_boundary_distance.eq(0).sum()),
        largest_discrepancies=d.nlargest(5,'max_score_error').to_dict('records'),
        interpretation='Historical comparison only; original-image margin was not replayed. No automatic acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rejected',action='append',default=[],metavar='GAMMA=CACHE_NAME')
    args = parser.parse_args()
    curated = Path(os.environ['CURATED_DATA'])
    repo = Path(__file__).resolve().parents[1]
    root = curated/'mcbm_notebook03_replay'
    print('READ-ONLY RECOVERY AUDIT: no GPU, training, diagnostic fits, cache changes or acceptance changes.',flush=True)
    records=[]
    for marker in sorted(root.glob('*/SUCCESS.json')):
        try:
            meta=json.loads(marker.read_text())
            checkpoint=Path(meta['checkpoint'])
            # Only this chapter's exact named model family belongs in this inventory.
            name=checkpoint.parent.parent.parent.name
            if not name.startswith('funnybirds-mcbm-g'):
                continue
            values=marker.parent/'h_cf.npy'
            valid=digest(values)==meta['sha256']
            h=np.load(values,allow_pickle=False,mmap_mode='r')
            valid=valid and h.shape==(5000,26) and bool(np.isfinite(h).all())
            record=dict(model=name,status='EXISTING VERIFIED CACHE' if valid else 'INVALID OUTPUT',
                        cache=str(marker.parent),checkpoint_exists=checkpoint.is_file(),
                        limitation='Saved h hash/shape verified; old manifest does not store checkpoint hash separately.')
            diagnostic=marker.parent/'replay_diagnostic.csv'
            if diagnostic.is_file(): record['diagnostic']=summarize(diagnostic)
            records.append(record)
        except Exception as exc:
            records.append(dict(status='ERROR',cache=str(marker.parent),reason=str(exc)))
    for item in args.rejected:
        gamma,name=item.split('=',1)
        if Path(name).name!=name or name in ('.','..'):
            raise ValueError('Rejected cache must be a directory name, not a path')
        path=root/name/'replay_diagnostic.csv'
        try:
            records.append(dict(gamma=gamma,status='INCOMPLETE: REPLAY NOT ACCEPTED',
                                cache=str(path.parent),diagnostic=summarize(path)))
        except Exception as exc:
            records.append(dict(gamma=gamma,status='ERROR',reason=str(exc)))
    print(json.dumps(records,indent=2,allow_nan=False),flush=True)
    print('REQUIRED INPUT INVENTORY (no restoration or patching):',flush=True)
    for tag in ('0','0p1','0p3','1','3','5'):
        name=f'funnybirds-mcbm-g{tag}'
        paths=[repo/'external/minimal_cbm/configs/funnybirds'/f'{name}.yaml',
               repo/'external/minimal_cbm/results'/name/'1/models/epoch_100.pt',
               repo/'external/minimal_cbm/results'/name/'1/predictions/epoch_100.pth',
               curated/'swap_fixed_v2_attempt2'/f'{name}-s1.csv']
        print(name, json.dumps({str(p):p.is_file() for p in paths}),flush=True)
    print('Audit complete. Gamma 3/5 are NOT approved by this inventory. Existing accepted artifacts remain unchanged.')


if __name__=='__main__':
    main()
