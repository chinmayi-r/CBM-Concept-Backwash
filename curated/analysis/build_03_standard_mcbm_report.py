"""Build the FunnyBird MCBM loss-engineering chapter.

This builder deliberately does not adapt every Notebook 02 cell once per gamma.
It displays the executed Koh figure when its construction is the useful control,
then puts one matched all-gamma MCBM analysis underneath.  Scientific results
are computed in the notebook; the prose fixes questions and interpretation
boundaries, not outcomes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from pathlib import Path

from build_standard_cbm_reports import build_funnybird
from build_mcbm_parity import TAGS, find


HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
OUTPUT = CURATED / "notebooks/03_funnybirds_mcbm.ipynb"


def _source(value: str) -> list[str]:
    value = textwrap.dedent(value).strip("\n") + "\n"
    return value.splitlines(keepends=True)


def _id(prefix: str, source: str) -> str:
    return f"m3-{prefix}-{hashlib.sha256(source.encode()).hexdigest()[:10]}"


def md(prefix: str, source: str) -> dict:
    return {"cell_type": "markdown", "id": _id(prefix, source),
            "metadata": {}, "source": _source(source)}


def code(prefix: str, source: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "id": _id(prefix, source), "metadata": {}, "outputs": [],
            "source": _source(source)}


EXPECTED = build_funnybird()["cells"]
STANDARD_HASHES = {
    tag: hashlib.sha256("".join(find(EXPECTED, tag)["source"]).encode()).hexdigest()
    for tag in TAGS
}


def build() -> dict:
    cells: list[dict] = []
    cells += [md("title", r"""
    # Chapter 03 — What MCBM changes, and why its parts behave differently

    This chapter is not a gallery of six gamma sweeps.  It asks two different
    questions that must not be mixed:

    1. **Koh Standard → MCBM gamma 0:** what changes when we replace the whole
       model and training recipe, even before the minimality penalty is active?
    2. **MCBM gamma 0 → positive gamma:** what changes as the *same MCBM
       architecture* is increasingly pushed toward its label-defined target?

    The scientific endpoint is: **what compression changed, whether grounding
    improved, and which measured failure gives us a reason to try a particular
    better loss.**  A description such as “the model did not generalize” is not
    treated as an explanation.  We locate the change in the computation and
    test candidate mechanisms wherever the existing artifacts allow it.

    All results are seed 1.  They establish model-specific mechanisms, not a
    population estimate over random initializations.
    """), md("model", r"""
    ## Models, symbols, and losses

    ### Koh Joint Standard CBM

    The ResNet-50 image encoder emits features.  Twenty-six scalar concept heads
    produce raw concept logits `z`.  A single saved linear layer reads all 26
    logits and predicts one of 50 species.  Training minimizes the normalized
    species loss plus `0.01 ×` concept loss.

    ### MCBM

    The ResNet-50 encoder and a shared projector produce a 26-number internal
    vector called `h` in this report.  The official MCBM source calls this vector
    `z`; we rename it only to avoid confusing it with Koh's raw concept logits.
    Each coordinate has a small learned reader `q_j`, giving the concept logit

    `z_j = q_j(h_j)`.

    The saved species classifier reads the **entire h vector**, not the
    thresholded concepts and not the post-reader `z` vector.  Its loss is

    `L = L_species + beta L_concept + gamma L_min`,

    with `beta=1`.  For binary label `c_j`, the MCBM target is

    `t_j = 6 c_j - 3`, so a negative label targets `-3` and a positive label
    targets `+3`.  The implemented minimality term is

    `L_min = 0.2 Σ_j mean[(h_j - t_j)^2]`.

    It is a soft training pressure, not a hard clamp: `h_j` may still differ
    from `-3/+3`.  Gamma 0 removes this term but keeps the MCBM architecture,
    noise path, preprocessing, optimizer, and nonlinear species head.  Therefore
    gamma 0 is the correct internal baseline for gamma, but it is not Koh.

    ### Controlled-swap quantities

    For a source bird whose old value is replaced with a donor value:

    - `m_orig = z_donor,orig - z_source,orig`;
    - `donor_gain = z_donor,cf - z_donor,orig`;
    - `source_decrease = z_source,orig - z_source,cf`;
    - `response_delta = donor_gain + source_decrease`;
    - `m_cf = m_orig + response_delta`.

    Example: `m_orig=-10`, donor rises by 6, and source falls by 3.  Then
    `response_delta=9` but `m_cf=-1`: the pixels helped, yet the old value is
    still above the inserted value.  That is the controlled backwash candidate.
    Exact-value recognition is stricter: among *all values for that part*, did
    the inserted value have the largest final `z`?
    """), md("plan", r"""
    ## Evidence plan and Notebook 02 parity

    Each listed Standard analysis is retained, combined, or replaced for a
    stated reason.  “Combined” means the metric and denominator are unchanged;
    only model/gamma is added as a color, row, or column.  Complex Standard
    figures are shown exactly as rendered in Notebook 02 and followed by a
    separately constructed MCBM figure.

    | Notebook 02 evidence | Treatment here | Reason |
    |---|---|---|
    | exact concept health (1) | exact Standard, then MCBM heatmaps | different internal architecture; do not fake one scale |
    | semantic and one-swap checks (2a/2b) | exact Standard once | identical accepted rendered inputs |
    | response/decomposition/event/outcomes (3/3b/4/4b) | exact Standard, then combined all-gamma MCBM tables | same swap definitions; MCBM also needs `h→q(h)` localization |
    | direction, visibility, conflict (5/6/6b/6c) | Standard reference plus all-gamma summaries | candidate contributors, not automatic explanations |
    | exact values/gallery/support (7/7a/7b/7c) | Standard reference plus per-value MCBM table | part averages can hide opposite value behavior |
    | source residuals (8) | Standard reference plus gamma/value residual audit | asks what remains after exact transition is controlled |
    | species information/head use (8b/8c/8d) | Standard reference plus MCBM information and direct frozen-head intervention | decoding, use, and causal head sensitivity are different questions |
    | predictive accounting (9) | Standard reference plus MCBM measured-contributor table | prediction is not causal decomposition |
    | descriptive synthesis (9b) | retired | selected four fractions arbitrarily; the final all-fronts table is complete |
    | downstream species association (10) | Standard reference plus model/gamma table | separates concept grounding from species-head consequences |
    | within-part evidence correlation | appendix only | weak secondary question; it does not rank parts |
    """), code("setup", rf"""
    import os, json, re, hashlib, sys
    from pathlib import Path
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
    from IPython.display import display, Markdown

    CURATED=Path(os.environ['CURATED_DATA'])
    REPO=Path.cwd() if (Path.cwd()/'analysis').is_dir() else Path.cwd().parent
    sys.path.insert(0,str(REPO/'analysis'))
    sys.path.insert(0,str(REPO/'data/funnybirds'))
    from minimal_cbm_scores import concept_logits_from_saved_latent, validate_saved_probabilities
    from funnybirds_concepts import load_parts, concept_names, group_slices

    ORDER=['tail','wing','beak','foot','eye']
    COLORS=dict(tail='#7115B5',wing='#087EB8',beak='#E9A000',foot='#00A478',eye='#C878A5')
    GAMMAS=[0.,.1,.3,1.,3.,5.]
    LABELS={{0.:'MCBM γ0',.1:'MCBM γ0.1',.3:'MCBM γ0.3',1.:'MCBM γ1',3.:'MCBM γ3',5.:'MCBM γ5'}}
    TAG={{0.:'g0',.1:'g0p1',.3:'g0p3',1.:'g1',3.:'g3',5.:'g5'}}
    STANDARD_TAGS={json.dumps(TAGS)}
    STANDARD_NOTEBOOK=REPO/'notebooks/02_funnybirds_cbm.ipynb'
    STANDARD=json.loads(STANDARD_NOTEBOOK.read_text(encoding='utf-8'))
    STANDARD_HASHES={json.dumps(STANDARD_HASHES, sort_keys=True)}
    SHOWN_STANDARD=set()
    plt.rcParams.update({{'figure.dpi':120,'axes.grid':False}})

    def emit_standard_cell(cell):
        if cell.get('cell_type')=='markdown':
            display(Markdown(''.join(cell.get('source',[]))))
            return
        for out in cell.get('outputs',[]):
            if 'data' in out:
                display({{k:''.join(v) if isinstance(v,list) else v for k,v in out['data'].items()}},raw=True)
            elif out.get('output_type')=='stream': print(''.join(out.get('text',[])),end='')

    def show_standard(*tags):
        for tag in tags:
            if tag in SHOWN_STANDARD: continue
            positions=[i for i,c in enumerate(STANDARD['cells']) if c.get('cell_type')=='code' and
                       re.fullmatch('fb-'+re.escape(tag)+'-[0-9a-f]+',c.get('id',''))]
            if len(positions)!=1: raise RuntimeError(f'Standard figure {{tag}} missing')
            position=positions[0]; cell=STANDARD['cells'][position]
            actual=hashlib.sha256(''.join(cell['source']).encode()).hexdigest()
            if actual!=STANDARD_HASHES[tag]:
                raise RuntimeError(f'Standard figure {{tag}} is stale; render Notebook 02 first')
            if not cell.get('outputs') or any(o.get('output_type')=='error' for o in cell['outputs']):
                raise RuntimeError(f'Standard figure {{tag}} lacks successful output')
            # Copy the question and any immediately following explanation before
            # the figure, not the result paragraph from the preceding figure.
            question_prefix='fb-q'+tag+'-'
            question=[i for i in range(position) if STANDARD['cells'][i].get('id','').startswith(question_prefix)]
            if question:
                for item in STANDARD['cells'][max(question):position]: emit_standard_cell(item)
            display(Markdown(f'**Exact executed Koh Standard block — Notebook 02 `{{tag}}`**'))
            emit_standard_cell(cell); SHOWN_STANDARD.add(tag)
            # Preserve its method, literal explanation, and executed review cells
            # until the next numbered question begins.
            for item in STANDARD['cells'][position+1:]:
                identity=item.get('id','')
                if item.get('cell_type')=='markdown' and identity.startswith('fb-q'): break
                if item.get('cell_type')=='code':
                    match=re.fullmatch(r'fb-([^\s]+)-[0-9a-f]+',identity)
                    if match: SHOWN_STANDARD.add(match.group(1))
                emit_standard_cell(item)

    def heat(ax,table,title,label,vmin=None,vmax=None,cmap='viridis',fmt='.2f'):
        values=table.astype(float).to_numpy()
        im=ax.imshow(values,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_xticks(range(len(table.columns))); ax.set_xticklabels(table.columns)
        ax.set_yticks(range(len(table.index))); ax.set_yticklabels(table.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i,j]):
                    ax.text(j,i,format(values[i,j],fmt),ha='center',va='center',fontsize=8)
        ax.set_title(title); plt.colorbar(im,ax=ax,label=label,fraction=.046)

    def read_swap(path,model,gamma):
        d=pd.read_csv(path)
        required={{'part','z_new','z_old','z_new_orig','z_old_orig','margin','var_src','var_donor','sid_src'}}
        if required-set(d): raise RuntimeError(f'{{path}} lacks {{sorted(required-set(d))}}')
        d=d.copy(); d['model']=model; d['gamma']=gamma
        d['m_orig']=d.z_new_orig-d.z_old_orig
        d['m_cf']=d.margin
        d['donor_gain']=d.z_new-d.z_new_orig
        d['source_decrease']=d.z_old_orig-d.z_old
        d['response_delta']=d.m_cf-d.m_orig
        if not np.allclose(d.response_delta,d.donor_gain+d.source_decrease,atol=2e-4):
            raise RuntimeError(f'margin decomposition failed: {{path}}')
        d['backwash']=(d.response_delta>0)&(d.m_cf<0)
        d['no_donorward_move']=d.response_delta<=0
        exact=[]
        for row in d.itertuples():
            cols=sorted([c for c in d if c.startswith(f'z_cf_{{row.part}}_')],key=lambda c:int(c.rsplit('_',1)[1]))
            if not cols: exact.append((np.nan,np.nan,np.nan)); continue
            winner=int(np.argmax([getattr(row,c) for c in cols]))
            exact.append((winner==int(row.var_donor),winner==int(row.var_src),winner not in {{int(row.var_donor),int(row.var_src)}}))
        d[['exact_donor','exact_source','exact_third']]=pd.DataFrame(exact,index=d.index)
        return d

    KOH=read_swap(CURATED/'swap_koh_joint_resnet_accelerated_converged_v1_seed1'/'funnybirds-cbm-s1.csv','Koh Standard',np.nan)
    MCBM={{g:read_swap(CURATED/'swap_fixed_v2_attempt2'/f'funnybirds-mcbm-{{TAG[g]}}-s1.csv',LABELS[g],g) for g in GAMMAS}}
    ALL=pd.concat([KOH]+[MCBM[g] for g in GAMMAS],ignore_index=True)
    ids=KOH.render_id.astype(str).to_numpy()
    for g,d in MCBM.items():
        if not np.array_equal(ids,d.render_id.astype(str).to_numpy()):
            raise RuntimeError(f'gamma {{g}} swap row identity/order differs from Koh')
    PATHWAY=CURATED/'mcbm_swap_pathway_v3'
    required_pathway=['SUCCESS.json','pathway_rows.csv','pathway_summary.csv','per_value_pathway_summary.csv',
      'original_restored_offtarget_summary.csv','matched_original_species_health.csv']
    missing=[name for name in required_pathway if not (PATHWAY/name).is_file()]
    if missing: raise RuntimeError(f'Run run_mcbm_swap_pathway_report.sh first; missing {{missing}}')
    P_ROWS=pd.read_csv(PATHWAY/'pathway_rows.csv')
    P_SUM=pd.read_csv(PATHWAY/'pathway_summary.csv')
    P_VALUE=pd.read_csv(PATHWAY/'per_value_pathway_summary.csv')
    HYBRID=pd.read_csv(PATHWAY/'original_restored_offtarget_summary.csv')
    MATCHED_HEALTH=pd.read_csv(PATHWAY/'matched_original_species_health.csv')
    TABLES=CURATED/'mcbm_notebook03_tables'
    print(f'Loaded {{len(KOH):,}} Koh rows and {{sum(len(v) for v in MCBM.values()):,}} MCBM rows; no training.')
    """)]

    cells += [md("shared-before", r"""
    ## 1. Same physical intervention, checked once

    **Question and prediction.** Are Koh and all six MCBMs evaluated on the same
    one-part replacements? They must be; otherwise a model comparison could be
    an image comparison.

    **Inputs and method.** The accepted fixed-render CSVs contain the identical
    5,000 rows: 1,000 per part.  The following are the exact Notebook 02 semantic
    preflight and representative-swap outputs.  No diagnostic is trained and no
    model result is inferred here.

    **Reading rule.** These checks establish intervention identity only.  They do
    not establish that any model recognized the inserted part.
    """), code("shared", "show_standard('f2a','f2b')"), md("shared-after", r"""
    **Literal result.** The same render IDs were also asserted when the CSVs were
    loaded above.  Thus later differences come from model outputs on matched
    pixels.  **Next question:** were the ordinary concept outputs healthy before
    examining swaps?
    """)]

    cells += [md("health-before", r"""
    ## 2. Ordinary-image health: first rule out collapse

    **Question and prediction.** A swap failure is uninterpretable if the concept
    coordinate is numerically collapsed or cannot separate its ordinary 0/1
    labels.  Healthy models should have nonzero raw-score spread and above-chance
    balanced accuracy.

    **Standard baseline.** Figure 1 below is copied exactly from executed
    Notebook 02.  Its definitions and per-concept denominators remain unchanged.
    The MCBM panel then evaluates every gamma on its saved ordinary export.

    **MCBM quantities.** For each concept `j`, raw-logit spread is
    `Q95(z_j)-Q05(z_j)`. Balanced accuracy is one half of positive recall plus
    negative recall after thresholding `z_j` at zero. `h` target RMSE is
    `sqrt(mean((h_j-(6c_j-3))^2))`.  Lower target RMSE means compression toward
    the label target; it is not grounding evidence.
    """), code("health-standard", "show_standard('f1')"), code("health-mcbm", r"""
    from minimal_cbm_scores import concept_logits_from_saved_latent, validate_saved_probabilities
    health=[]; per_part=[]
    for g in GAMMAS:
        base=REPO/'external/minimal_cbm/results'/f'funnybirds-mcbm-{TAG[g]}'/'1'
        d=torch.load(base/'predictions/epoch_100.pth',map_location='cpu',weights_only=False)
        h=d['z'].float().reshape(len(d['z']),-1); c=d['c'].float().reshape(len(h),-1)
        z=concept_logits_from_saved_latent(h,base/'models/epoch_100.pt',c.shape[1]).float()
        validate_saved_probabilities(z,d['c_preds'])
        parts=load_parts(Path(os.environ.get('FUNNYBIRDS_ROOT',CURATED/'FunnyBirds')))
        spans=group_slices(parts)
        for part in ORDER:
            lo,hi=spans[part]; zz=z[:,lo:hi]; cc=c[:,lo:hi]
            pred=zz>0
            tpr=((pred)&(cc==1)).sum()/max(1,int((cc==1).sum()))
            tnr=((~pred)&(cc==0)).sum()/max(1,int((cc==0).sum()))
            spreads=[]
            for j in range(lo,hi): spreads.append(float(torch.quantile(z[:,j],.95)-torch.quantile(z[:,j],.05)))
            per_part.append(dict(gamma=g,part=part,balanced_accuracy=float((tpr+tnr)/2),
                median_z_spread=float(np.median(spreads)),h_target_RMSE=float(torch.sqrt(((h[:,lo:hi]-(6*cc-3))**2).mean()))))
    HEALTH=pd.DataFrame(per_part)
    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    for ax,(col,title,cmap,lo,hi) in zip(axes,[
      ('balanced_accuracy','A · Ordinary exact-concept balanced accuracy','viridis',0,1),
      ('median_z_spread','B · Median raw-z spread within each part','viridis',0,None),
      ('h_target_RMSE','C · Distance of h from the label target ±3','magma',0,None)]):
        table=HEALTH.pivot(index='gamma',columns='part',values=col).reindex(index=GAMMAS,columns=ORDER)
        heat(ax,table,title,'fraction' if col=='balanced_accuracy' else ('raw z units' if col=='median_z_spread' else 'h units'),lo,hi,cmap)
    plt.tight_layout(); plt.show(); display(HEALTH.round(4))
    """), md("health-after", r"""
    **How to interpret this figure.** Panel A says whether ordinary labels remain
    classifiable. Panel B tests literal raw-score collapse. Panel C says whether
    gamma did what its squared-error term requests. A lower C together with a
    worse swap result would mean “more label-compressed” did not mean “more
    pixel-grounded.”

    **Alternative.** High ordinary accuracy can still come from body/species
    context. **Distinguishing test:** the controlled swap below. **Limited
    conclusion:** use this only as a health gate. **Next:** locate the first
    Koh→gamma-0 behavior change.
    """)]

    cells += [md("stage-a-before", r"""
    ## 3. Transition A — Koh Standard to MCBM gamma 0

    This transition changes architecture and recipe together: shared projector,
    private `q_j` readers, injected training noise, nonlinear species head,
    preprocessing, optimizer details, and uncontrolled initialization.  Gamma is
    zero, so the minimality penalty cannot explain this transition.

    **Question.** Which part of the swap computation changes first?

    **Prediction.** If MCBM gamma 0 damage is mainly inside `q_j`, label-calibrated
    `h` will recognize the donor but final `z=q(h)` will not.  If `h` already
    fails, the representation/encoder stage is the earlier location.

    **Standard reference.** The next outputs are Notebook 02's exact donorward
    response, five-term decomposition, backwash predicate, and exhaustive
    three-way pairwise outcome. They are intentionally not redrawn.
    """), code("standard-core", "show_standard('f3','f3b','f4','f4b')"), md("outcomes-before", r"""
    ### Matched MCBM outcome accounting

    Rows are gamma and columns are parts. Every cell uses all 1,000 swaps for
    that gamma/part.  Panel A asks whether the donor is largest among every exact
    value. B asks whether the old value is largest. C records a third value.
    D is `response_delta<=0`. E is the controlled backwash predicate
    `response_delta>0 and m_cf<0`. Exact-winner panels partition all rows; D/E
    are pairwise diagnostics and overlap with the exact categories, so their
    percentages must not be added to A/B/C.
    """), code("outcomes", r"""
    out=[]
    for g,d in MCBM.items():
        for part,q in d.groupby('part'):
            out.append(dict(gamma=g,part=part,n=len(q),exact_donor=q.exact_donor.mean(),
              exact_source=q.exact_source.mean(),exact_third=q.exact_third.mean(),
              no_donorward_move=q.no_donorward_move.mean(),backwash=q.backwash.mean()))
    OUT=pd.DataFrame(out)
    fig,axes=plt.subplots(1,5,figsize=(22,4.5))
    for ax,(col,title) in zip(axes,[('exact_donor','A · donor is exact winner'),('exact_source','B · old value is exact winner'),
      ('exact_third','C · third value wins'),('no_donorward_move','D · no donorward movement'),('backwash','E · donorward, old still above donor')]):
        table=OUT.pivot(index='gamma',columns='part',values=col).reindex(index=GAMMAS,columns=ORDER)
        heat(ax,table,title,'fraction',0,1,'viridis')
    plt.tight_layout(); plt.show(); display(OUT.round(4))
    """), md("outcomes-after", r"""
    **Literal result:** read the printed cells, not just color. The gamma-0 row is
    the direct comparison with Koh; later rows are the gamma experiment.

    **What it supports:** it identifies which part/outcome changed. **What it
    does not explain:** why. **Alternative:** a poor final margin could begin as
    a deeper starting deficit, a smaller donor rise, a smaller source fall, or
    distortion by `q`. **Discriminating test:** decompose those terms and compare
    calibrated `h` with `z`. **Next:** that localization.
    """), md("decomp-before", r"""
    ### Where the response is lost

    The first five panels average the exact equations defined at the chapter
    start.  The last three use the v3 replay audit. “Calibrated h response” first
    orients each internal coordinate using ordinary absent/present examples, so
    positive always means movement toward the inserted label even if a learned
    reader reverses sign. “q breaks h” means `h` selected the donor exact value
    but `z=q(h)` did not; “q repairs h” is the reverse.

    No new diagnostic classifier is trained. The accepted frozen MCBM checkpoint
    is replayed, and the existing `q_j` readers are reused.
    """), code("decomp", r"""
    strict=P_SUM[P_SUM.population.eq('strict matched replay')].copy()
    metrics=[('mean_z_donor_gain','donor gain'),('mean_z_source_decrease','source decrease'),
      ('mean_z_response','total z response'),('z_response_positive_rate','positive z-response rate'),
      ('z_exact_donor_recognition','exact donor recognition'),('mean_calibrated_h_response','calibrated h response'),
      ('q_breaks_h_success_rate','q breaks h success'),('q_repairs_h_failure_rate','q repairs h failure')]
    fig,axes=plt.subplots(2,4,figsize=(20,9))
    for ax,(col,title) in zip(axes.flat,metrics):
        table=strict.pivot(index='gamma',columns='part',values=col).reindex(index=GAMMAS,columns=ORDER)
        rate=('rate' in col or 'recognition' in col)
        heat(ax,table,title,'fraction' if rate else ('calibrated h units' if 'h_response' in col else 'raw z units'),0 if rate else None,1 if rate else None,'viridis' if rate else 'coolwarm')
    plt.tight_layout(); plt.show()
    display(strict[['gamma','part']+[x[0] for x in metrics]].round(4))
    display(MATCHED_HEALTH.round(4))
    """), md("decomp-after", r"""
    **Reading the mechanism.** If calibrated `h` and final `z` both fail for a
    part, blaming `q` is wrong: the useful donor response was already missing in
    `h`. Large “q breaks” would instead locate the damage after `h`. The matched
    ordinary-health table prevents a different 5,000-image population from being
    mistaken for model damage.

    **Alternative.** This localizes the failure within the stored computation but
    does not separate the shared projector, noise, optimizer, preprocessing, or
    initialization inside the Koh→gamma-0 bundle. **Distinguishing experiment:**
    matched controlled ablations changing one of those at a time. **Limited
    conclusion:** name the earliest observed stage; do not call the bundled cause
    identified. **Next:** within MCBM, ask what gamma itself changes.
    """)]

    cells += [md("stage-b-before", r"""
    ## 4. Transition B — MCBM gamma 0 to positive gamma

    Now architecture and stored recipe are nominally shared, and gamma controls
    the added squared-error pressure toward `h=-3/+3`. Initialization was not
    seeded in these historical runs, so non-monotonic differences can still be
    run-to-run variation. A smooth dose trend is more persuasive than one jump.

    **Question.** Does stronger compression improve inserted-pixel response, or
    merely make ordinary `h` values more label-like?

    **Prediction.** If the loss removes harmful within-label context while
    preserving pixel response, target RMSE and within-label spread should fall
    while donor gain and exact donor recognition rise. If response falls while
    target RMSE improves, the loss is compressing information useful for the
    controlled intervention, or consolidating a contextual shortcut.
    """), code("loss-gradients", r"""
    gradient_path=TABLES/'loss_gradients.csv'
    if gradient_path.is_file():
        GRAD=pd.read_csv(gradient_path)
        display(Markdown('**Stored frozen-gradient diagnostic** — magnitudes compare terms within this implementation; they are not a Koh-versus-MCBM loss-ratio claim.'))
        display(GRAD.round(5))
    else:
        print('No loss_gradients.csv: gradient-magnitude appendix unavailable; outcome/pathway analyses remain valid.')
    """), md("direction-before", r"""
    ### Direction, visibility, label conflict, and exact values

    These are candidate contributors, not interchangeable explanations.
    Direction asks whether forward/backward swaps differ. Visibility is the
    corrected total mask area of the inserted bilateral part. Conflict rate is a
    data rate: among positive training labels for an exact value, the fraction
    whose named region is hidden. Support is the number of 50 species naturally
    carrying that exact value.

    The exact Standard figures follow first. Shared label/mask and value-gallery
    figures appear once because their inputs do not depend on the model.
    """), code("standard-contributors", "show_standard('f5','f6','f6b','f6c','f7','f7a','f7b','f7c')"), md("value-before", r"""
    ### MCBM per-value audit

    Every row below is one donor value at one gamma. It reports its row count,
    original-image count, support, label/mask conflict, corrected visible pixels,
    donor gain, source decrease, total response, final margin, and exact outcome.
    This is the correct level for questions such as “why did foot improve while
    wing worsened?”: a part average cannot reveal whether one value dominates.

    The heatmaps summarize values within each part only after the full table is
    printed. No causal model is fitted.
    """), code("values", r"""
    display(P_VALUE.round(4))
    value_metrics=[('mean_corrected_visible_pixels','visibility (pixels)'),('donor_conflict_rate','label/mask conflict'),
      ('donor_species_support','species support'),('mean_z_donor_gain','donor gain'),('mean_z_source_decrease','source decrease'),
      ('mean_z_response','response'),('exact_donor_recognition','exact donor wins')]
    VALUE_PART=P_VALUE.groupby(['gamma','part'])[ [m[0] for m in value_metrics] ].mean().reset_index()
    fig,axes=plt.subplots(2,4,figsize=(20,9))
    for ax,(col,title) in zip(axes.flat,value_metrics):
        table=VALUE_PART.pivot(index='gamma',columns='part',values=col).reindex(index=GAMMAS,columns=ORDER)
        heat(ax,table,title,title,0 if col in {'donor_conflict_rate','exact_donor_recognition'} else None,
             1 if col in {'donor_conflict_rate','exact_donor_recognition'} else None,
             'viridis' if col not in {'mean_z_donor_gain','mean_z_source_decrease','mean_z_response'} else 'coolwarm')
    axes.flat[-1].axis('off'); plt.tight_layout(); plt.show()
    """), md("values-after", r"""
    **Interpretation rule.** A contributor is not established merely because a
    bad part has an extreme mean. Look for matched-support or matched-visibility
    values with different outcomes, dose trends across gamma, and whether the
    candidate changes the matching response component. Label conflict cannot
    explain Koh→gamma-0 damage if low-conflict parts degrade while the
    high-conflict tail does not.

    **Alternative.** Visual compatibility with an unusual source body, species
    diversity inside a value, or initialization may differ even at equal support
    and area. **Distinguishing tests:** grouped row-level models and replicated
    one-factor training ablations. **Next:** test the species/context mechanism
    directly rather than naming every unexplained residual “context.”
    """)]

    cells += [md("context-before", r"""
    ## 5. Species information: available, used, and causally relevant are different

    **Question.** Does MCBM reduce species information in concept magnitudes, and
    does its saved species head actually depend on that information?

    **Three separate tests.** (1) A new held-out logistic diagnostic measures how
    much species prediction improves from raw magnitudes after the 0/1 labels are
    known. (2) The unchanged saved species head is rerun after within-label
    magnitudes are replaced by training-fold label means. (3) On swaps, off-target
    same-part `h` coordinates are restored to their exact original-image values,
    while the source and donor coordinates remain fixed; the unchanged saved
    species head is rerun. Test 3 is the direct intervention.

    The Standard decoding, saved-head reliance, and off-target intervention are
    displayed exactly first. Their linear `Wz+b` head is not substituted for
    MCBM's nonlinear head.
    """), code("standard-context", "show_standard('f8b','r8b-compare','f8c-source','f8d-source')"), code("mcbm-info", r"""
    info=[]; equal=[]; head=[]
    for g in GAMMAS:
        paths={'info':TABLES/f'g{TAG[g][1:]}_FULL_WIDTH_INFORMATION.csv',
                'equal':TABLES/f'g{TAG[g][1:]}_EQUAL_WIDTH_INFORMATION.csv',
                'head':TABLES/f'g{TAG[g][1:]}_HEAD_USE.csv'}
        # Older table names use the literal tag, including g0p1.
        paths={k:(TABLES/f'{TAG[g]}_{suffix}.csv') for k,suffix in
               [('info','FULL_WIDTH_INFORMATION'),('equal','EQUAL_WIDTH_INFORMATION'),('head','HEAD_USE')]}
        for path in paths.values():
            if not path.is_file(): raise FileNotFoundError(path)
        a=pd.read_csv(paths['info']); a['gamma']=g; info.append(a)
        b=pd.read_csv(paths['equal']); b['gamma']=g; equal.append(b)
        c=pd.read_csv(paths['head']); c['gamma']=g; head.append(c)
    INFO=pd.concat(info,ignore_index=True); EQUAL=pd.concat(equal,ignore_index=True); HEAD=pd.concat(head,ignore_index=True)
    fig,axes=plt.subplots(1,3,figsize=(18,4.8))
    for ax,data,col,title in [
      (axes[0],INFO,'conditional_logloss_gain','A · species information beyond 0/1 labels'),
      (axes[1],EQUAL,'mean_conditional_gain','B · same three-coordinate budget'),
      (axes[2],HEAD[HEAD.replaced_block.isin(ORDER)],'mean_probability_mass_moved','C · saved-head sensitivity to magnitudes')]:
        idx='part' if 'part' in data else 'replaced_block'
        table=data.pivot(index='gamma',columns=idx,values=col).reindex(index=GAMMAS,columns=ORDER)
        heat(ax,table,title,'held-out log-loss gain' if 'gain' in col else 'mean probability mass moved',0,None,'viridis')
    plt.tight_layout(); plt.show()
    display(INFO.round(4)); display(EQUAL.round(4)); display(HEAD.round(4))
    """), md("hybrid-before", r"""
    ### Direct frozen-head intervention on swapped images

    Example for `tail_2 → tail_7`: MCBM always has nine tail coordinates. Keep
    the counterfactual `tail_2` and `tail_7` values exactly unchanged. Replace
    the other seven tail coordinates with their values from the matching
    original image. Rerun the same saved species head.

    `source evidence = (source logit - donor logit)_before -
    (source logit - donor logit)_after`.

    Positive means the swap-induced off-target changes had been helping the old
    source species; negative means they had been helping the donor. Panel C's
    pairwise flip rate asks the narrow decision-relevant question. Panel D's
    probability-mass movement can include unrelated species and must not be
    called a source-to-donor flip.
    """), code("hybrid", r"""
    metrics=[('mean_swap_induced_offtarget_source_evidence','A · mean source evidence','species-logit units',None,None,'coolwarm'),
      ('fraction_source_evidence_positive','B · fraction favouring source','fraction',0,1,'viridis'),
      ('pairwise_source_to_donor_flip_rate','C · source-to-donor pair flips','fraction',0,1,'magma'),
      ('mean_probability_mass_moved','D · total class probability moved','probability mass',0,None,'viridis')]
    fig,axes=plt.subplots(1,4,figsize=(21,4.7))
    for ax,(col,title,label,lo,hi,cmap) in zip(axes,metrics):
        table=HYBRID.pivot(index='gamma',columns='part',values=col).reindex(index=GAMMAS,columns=ORDER)
        heat(ax,table,title,label,lo,hi,cmap,fmt='.3f')
    plt.tight_layout(); plt.show(); display(HYBRID.round(5))
    """), md("context-after", r"""
    **Logic chain.** Decodability says information exists. Label-mean replacement
    says the saved classifier is sensitive to some within-label magnitudes.
    Only the original-restored intervention asks whether the *swap-induced
    off-target changes* causally sustain the source-over-donor species gap.

    **Critical boundary.** Even a large species-head effect cannot cause the
    upstream concept margin: the species head is downstream. A near-zero
    intervention cannot prove that no context exists inside the protected source
    or donor coordinate; it rejects only the measured off-target pathway.

    **Alternative.** Context may be compressed into the source/donor coordinate
    itself, or the exact-value failure may be unrelated to the class head.
    **Distinguishing test:** intervene on matched source/donor `h` components only,
    or train a spatially grounded loss. **Next:** inspect residual organization
    and predictive accounting without pretending they are causal percentages.
    """)]

    cells += [md("residual-before", r"""
    ## 6. What remains unexplained

    The exact Standard source-species residual and grouped predictive audit are
    shown first. The residual subtracts the training-fold mean for the identical
    `(part, old value, donor value)` transition; it does not subtract a species
    mean. Therefore transition residuals average to zero overall, while one
    source species can remain systematically positive or negative.

    Example: if the pooled margins for one exact transition are
    `[-5,-3,+1,+3]`, their mean is `-1` and residuals are `[-4,-2,+2,+4]`.
    They sum to zero, but two species can occupy opposite sides.

    A predictive model may show that measured pre-outcome variables are useful;
    it cannot divide backwash into additive causal percentages.
    """), code("standard-residual", "show_standard('f8','f9-new')"), code("mcbm-residual", r"""
    residual_rows=[]
    for g,d in MCBM.items():
        q=d.copy()
        keys=['part','var_src','var_donor']
        q['pair_mean']=q.groupby(keys).m_cf.transform('mean')
        q['pair_residual']=q.m_cf-q.pair_mean
        for part,p in q.groupby('part'):
            species=p.groupby('sid_src').pair_residual.mean()
            residual_rows.append(dict(gamma=g,part=part,n_rows=len(p),n_species=len(species),
              species_residual_SD=species.std(ddof=0),species_residual_range=species.max()-species.min()))
    RESIDUAL=pd.DataFrame(residual_rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,col,title in [(axes[0],'species_residual_SD','A · spread of source-species residual means'),
                         (axes[1],'species_residual_range','B · range of source-species residual means')]:
        table=RESIDUAL.pivot(index='gamma',columns='part',values=col).reindex(index=GAMMAS,columns=ORDER)
        heat(ax,table,title,'raw z margin units',0,None,'viridis')
    plt.tight_layout(); plt.show(); display(RESIDUAL.round(4))

    predictive=[]
    holdout=[]
    for g in GAMMAS:
        path=TABLES/f'{TAG[g]}_PREDICTIVE.csv'
        value_path=TABLES/f'{TAG[g]}_VALUE_HOLDOUT.csv'
        q=pd.read_csv(path); q['gamma']=g; predictive.append(q)
        q=pd.read_csv(value_path); q['gamma']=g; holdout.append(q)
    PREDICTIVE=pd.concat(predictive,ignore_index=True)
    VALUE_HOLDOUT=pd.concat(holdout,ignore_index=True)
    display(Markdown('**Grouped held-out measured-contributor models**'))
    display(PREDICTIVE.round(4))
    display(Markdown('**Leave-one-donor-value-out stress test**'))
    display(VALUE_HOLDOUT.round(4))
    """), md("residual-after", r"""
    **What this supports.** Larger residual spread means exact transition alone
    does not account for all source-species organization. **Alternative:** body,
    pose, value prevalence, or a few extreme images can produce the same pattern.
    **Discriminating test:** repeated seeds and image-grouped held-out prediction.
    **Limited conclusion:** this quantifies remaining structure, not its cause.
    **Next:** ask whether concept-margin changes have a downstream association.
    """)]

    cells += [md("downstream-before", r"""
    ## 7. Downstream species consequence

    **Question.** When the inserted concept value becomes more favored, does the
    frozen classifier assign more probability to the donor species?

    **Quantity.** Swaps are divided into ten approximately equal-count bins by
    final raw concept margin `m_cf`. For each bin, x is mean `m_cf`; y is mean
    saved donor-species probability. This reuses each model's original saved
    classifier. No new classifier is trained.

    The Standard Figure 10 appears first. MCBM is then plotted with the same
    construction. This is an association downstream of concept scores; it cannot
    establish that the species head caused backwash.
    """), code("standard-downstream", "show_standard('f10')"), code("mcbm-downstream", r"""
    fig,axes=plt.subplots(2,3,figsize=(16,9)); downstream=[]
    for ax,g in zip(axes.flat,GAMMAS):
        d=MCBM[g]; prob=next((c for c in ['p_cf_donor','p_donor_cf','donor_species_prob'] if c in d),None)
        if prob is None:
            ax.text(.5,.5,'donor-species probability absent',ha='center'); ax.axis('off'); continue
        bins=pd.qcut(d.m_cf,10,duplicates='drop')
        q=d.groupby(bins,observed=True).agg(n=('m_cf','size'),mean_margin=('m_cf','mean'),mean_donor_probability=(prob,'mean')).reset_index(drop=True)
        q['gamma']=g; downstream.append(q)
        ax.plot(q.mean_margin,q.mean_donor_probability,'o-',color='#333333'); ax.axvline(0,color='black',ls='--')
        ax.set_title(LABELS[g]); ax.set_xlabel('mean final donor-minus-source concept margin'); ax.set_ylabel('mean saved donor-species probability')
        for row in q.itertuples(): ax.annotate(f'n={row.n}',(row.mean_margin,row.mean_donor_probability),fontsize=7)
    plt.tight_layout(); plt.show()
    if downstream: display(pd.concat(downstream,ignore_index=True).round(5))
    """), md("downstream-after", r"""
    **Interpretation.** An upward curve means donor-favoring concept scores are
    associated with more donor-species probability. The absolute y values say
    how large that consequence is. A one-part swap usually leaves the body and
    four parts belonging to the source, so a low donor-species probability is
    not paradoxical.

    **Alternative:** both quantities can respond to the same image features.
    **Distinguishing test:** a frozen-head intervention on selected concept
    coordinates. **Limited conclusion:** report downstream association, not
    wholesale species replacement. **Next:** synthesize every part and gamma.
    """)]

    cells += [md("final-before", r"""
    ## 8. All-fronts synthesis and loss recommendation

    This table is deliberately complete rather than tail-centered. For every
    model and part it prints the starting margin, donor rise, source fall, total
    response, final margin, exact donor/source/third winner rates, no-movement
    rate, and controlled-backwash rate. MCBM rows then join calibrated `h`
    response, `q` break/repair, ordinary health, and the direct off-target
    intervention.

    Read it in this order for each part:

    1. Did Koh→gamma 0 change the behavior before any minimality pressure?
    2. Is gamma's dose trend smooth, non-monotonic, or flat?
    3. Is the changed outcome explained by starting deficit, donor rise, source
       fall, or a third exact value?
    4. Is the loss already present in calibrated `h`, or introduced by `q`?
    5. Do visibility/conflict/support track the matching component?
    6. Does the saved-head intervention causally move the source/donor species
       decision enough to explain it?
    """), code("final-table", r"""
    summary=[]
    for (model,part),q in ALL.groupby(['model','part'],sort=False):
        summary.append(dict(model=model,part=part,n=len(q),m_orig=q.m_orig.mean(),donor_gain=q.donor_gain.mean(),
          source_decrease=q.source_decrease.mean(),response=q.response_delta.mean(),m_cf=q.m_cf.mean(),
          exact_donor=q.exact_donor.mean(),exact_source=q.exact_source.mean(),exact_third=q.exact_third.mean(),
          no_move=q.no_donorward_move.mean(),backwash=q.backwash.mean()))
    FINAL=pd.DataFrame(summary)
    mcbm_extra=strict[['gamma','part','mean_calibrated_h_response','q_breaks_h_success_rate','q_repairs_h_failure_rate']].merge(
      HEALTH[['gamma','part','balanced_accuracy','h_target_RMSE']],on=['gamma','part'],how='left').merge(
      HYBRID[['gamma','part','mean_swap_induced_offtarget_source_evidence','pairwise_source_to_donor_flip_rate']],on=['gamma','part'],how='left')
    FINAL['gamma']=FINAL.model.map({v:k for k,v in LABELS.items()})
    FINAL=FINAL.merge(mcbm_extra,on=['gamma','part'],how='left')
    display(FINAL.round(4))
    FINAL.to_csv(CURATED/'mcbm_notebook03_final_all_fronts.csv',index=False)
    """), md("final-answer", r"""
    ### Claim boundaries and the next loss

    The rendered numbers decide the detailed part-by-part conclusions. The
    admissible mechanism claims are:

    - **Koh→gamma 0:** localize the first changed stage, but do not attribute the
      bundled difference to gamma or to one architecture component.
    - **Gamma dose:** compression toward `-3/+3` is demonstrated only by lower
      `h` target error/spread. Grounding improves only where donor response and
      exact donor recognition also improve.
    - **`q` readers:** call them causal only when `q` break/repair rates account
      for the behavior. Otherwise the difference is already in `h`.
    - **Species/context:** decoding proves availability; saved-head replacement
      proves use on ordinary images; the original-restored swap intervention
      tests the specific off-target causal route. These are not synonyms.
    - **Measured contributors:** visibility, conflict, support, and source species
      may predict rows without summing to a causal decomposition.

    A defensible next loss follows from the location of the measured failure. If
    gamma makes `h` more label-like but removes donor response, the next candidate
    is **not merely larger gamma**. It is a spatial/interventional objective that
    rewards the correct coordinate for changing when its named pixels change,
    while penalizing changes in other coordinates. One concrete form for a
    matched original/counterfactual pair is

    `L_ground = max(0, margin_required - calibrated_h_response_changed_part)
              + lambda_off ||h_cf,offtarget - h_orig,offtarget||²`.

    This directly targets the two behaviors the renderer can verify. It is a
    proposal, not a result. If only final `q(h)` is damaged, regularizing the
    reader's monotonic orientation is the narrower repair. If neither measured
    failure is present, more loss engineering is not yet justified; first isolate
    the Koh→gamma-0 architecture/recipe component with matched seeded ablations.

    ### What RLv2 must test next

    Repeat this exact chapter with matched labels and renders. RLv2 is informative
    only if it changes the component predicted by label/mask conflict: the
    calibrated `h` response or exact donor recognition for values whose hidden
    positive labels were removed. A generic accuracy change is insufficient.
    """), md("appendix", r"""
    ## Appendix — secondary Standard evidence and completion ledger

    The within-part evidence-fifths/Spearman analysis is retained here because it
    asks a real but secondary question: among swaps of the same part, does more
    off-target source evidence co-occur with worse direct concept grounding? A
    weak correlation does not rank how much each part uses species information.
    """), code("appendix-corr", "show_standard('app-evidence-correlation-code')"), code("ledger", r"""
    accounted={
      'f1':'exact Standard + MCBM health','f2a':'shared exact','f2b':'shared exact','f3':'exact Standard + all-gamma response',
      'f3b':'exact Standard + all-gamma decomposition','f4':'exact Standard + all-gamma event','f4b':'exact Standard + exact winners',
      'f5':'exact Standard + per-value MCBM','f6':'exact Standard + corrected visibility','f6b':'shared exact','f6c':'exact Standard + per-value conflict',
      'f7':'exact Standard + per-value MCBM','f7a':'shared exact','f7b':'exact Standard + per-value support','f7c':'exact Standard + per-value support',
      'f8':'exact Standard + MCBM residual spread','f8b':'exact Standard + MCBM information','r8b-compare':'exact Standard comparison',
      'f8c-source':'exact Standard + MCBM saved-head use','f8d-source':'exact Standard + MCBM direct intervention',
      'f9-new':'exact Standard + stored MCBM predictive tables','f9b':'retired; replaced by complete all-fronts table',
      'f10':'exact Standard + same MCBM binned association','app-evidence-correlation-code':'appendix exact Standard'}
    LEDGER=pd.DataFrame([{'standard_tag':tag,'treatment':accounted.get(tag,'MISSING')} for tag in STANDARD_TAGS])
    if LEDGER.treatment.eq('MISSING').any(): raise RuntimeError('Incomplete Standard parity ledger')
    display(LEDGER)
    print('NOTEBOOK 03 COMPLETION PASS: every Standard tag accounted; no training; no Slurm.')
    """)]

    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}


def check_cached_inputs() -> None:
    root = Path(os.environ["CURATED_DATA"]) / "mcbm_notebook03_tables"
    tags = ("g0", "g0p1", "g0p3", "g1", "g3", "g5")
    suffixes = (
        "FULL_WIDTH_INFORMATION", "EQUAL_WIDTH_INFORMATION", "HEAD_USE",
        "PREDICTIVE", "VALUE_HOLDOUT",
    )
    required = [root / f"{tag}_{suffix}.csv" for tag in tags for suffix in suffixes]
    required.append(root / "loss_gradients.csv")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Notebook 03 requires the already-computed source tables from the accepted "
            "previous execution; missing:\n" + "\n".join(missing)
        )
    print(f"NOTEBOOK 03 CACHED-TABLE PREFLIGHT PASS: {len(required)} files under {root}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if args.check_inputs:
        check_cached_inputs()
        return
    notebook = build()
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
