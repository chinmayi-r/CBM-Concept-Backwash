"""Export every pilot edit into a portable, model-score-blind HTML gallery."""
import argparse
import base64
import html
import json
from pathlib import Path


def export(pilot, out):
    edits = sorted((pilot / 'edits').glob('*/EDIT.json'))
    if not edits:
        raise ValueError('No edit manifests available for review')
    sections = []
    for i, path in enumerate(edits, 1):
        record = json.loads(path.read_text())
        trial = record['trial']
        png = path.parent / 'review.png'
        encoded = base64.b64encode(png.read_bytes()).decode('ascii')
        title = f"Example {i}: {trial['trial_id']}"
        identity = {k: trial[k] for k in ('source_image', 'same_value_image', 'different_value_image',
                    'source_concept', 'donor_concept', 'mask_group')}
        sections.append(f'<section><h2>{html.escape(title)}</h2>'
            '<p>Panels, left to right: original photograph, self-reinsertion, same-value donor, '
            'different-value donor. Inspect inserted value, pose, old-part removal, and changes '
            'to other attributes. No model predictions are displayed.</p>'
            f'<img alt="{html.escape(title)}: four edit conditions" src="data:image/png;base64,{encoded}">'
            f'<details><summary>Image and concept identities</summary><pre>{html.escape(json.dumps(identity,indent=2))}</pre></details>'
            '<p>Review status: unreviewed. Record checks separately; this gallery does not certify calibration.</p></section>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Notebook 05 — blinded edit review</title>'
        '<style>body{font:17px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;line-height:1.5}'
        'img{width:100%;height:auto}section{margin:40px 0;border-top:1px solid #aaa}pre{white-space:pre-wrap}</style>'
        '<h1>Notebook 05: blinded CUB edit review</h1><p><strong>METHOD NOT CALIBRATED.</strong> '
        'This is a small label/mask-selected pilot, not a population backwash estimate. '
        'Review every example before opening prediction tables. All images are embedded in this file.</p>'
        + ''.join(sections) + '</html>')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:
        stream.write(document)
    print(f'Blinded HTML: {out.resolve()} ({len(edits)} examples)')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    export(args.pilot,args.out)
