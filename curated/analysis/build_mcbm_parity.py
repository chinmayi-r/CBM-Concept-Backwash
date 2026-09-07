"""Assemble exact Standard figure constructions into the loss-engineering chapter.

Generation copies visible Python cells, not opaque runtime exec strings. Only the
model-specific replay and input-population adapters differ. Standard is always
displayed from its executed notebook; its historical observations are not copied
onto MCBM results.
"""
from __future__ import annotations

import copy
import hashlib
import re

from build_standard_cbm_reports import build_funnybird

GAMMAS = [0., .1, .3, 1., 3., 5.]
TAGS = ["f1", "f2a", "f2b", "f3", "f3b", "f4", "f4b", "f5", "f6", "f6b",
        "f6c", "f7", "f7a", "f7b", "f7c", "f8", "f8b", "r8b-compare",
        "f8c-source", "f8d-source", "f9-new", "f9b", "f10",
        "app-evidence-correlation-code"]
SHARED = {"f2a", "f2b", "f6b", "f7a"}
DEPENDENCIES = {"f7c": ["VS"], "r8b-compare": ["PROBE", "diag"],
                "f9b": ["diag"], "app-evidence-correlation-code": ["EVIDENCE_ROWS"]}
SAVE = {"f7": ["diag"], "f7b": ["VS"], "f8b": ["PROBE"],
        "f8c-source": ["HEAD_USE", "FULL_WIDTH_INFORMATION", "EQUAL_WIDTH_INFORMATION"],
        "f8d-source": ["EVIDENCE_ROWS"], "f9-new": ["PREDICTIVE", "VALUE_HOLDOUT"]}


def find(cells, tag):
    selected = [c for c in cells if c["cell_type"] == "code"
                and re.fullmatch(r"fb-" + re.escape(tag) + r"-[0-9a-f]+", c["id"])]
    if len(selected) != 1:
        raise ValueError(f"Standard figure {tag}: expected one source cell, found {len(selected)}")
    return selected[0]


SETUP = r'''
from IPython.display import Markdown, Image as DisplayImage
from mcbm_loss_report import (task_head, checkpoint_tag, replacement_use, score_reference,
                             replay_counterfactual_h, off_target_erasure)
from funnybird_followup_diagnostics import (conditional_information, load_label_conflict,
    conflict_response_table, ordinary_value_recognition, add_descriptors, prediction_audit,
    value_holdout_audit)
# The standalone helper selects Agg for CLI figures; restore notebook display.
get_ipython().run_line_magic('matplotlib','inline')
parts=FB_PARTS
STANDARD_NOTEBOOK=REPO/'notebooks/02_funnybirds_cbm.ipynb'
STANDARD_RENDER=json.loads(STANDARD_NOTEBOOK.read_text(encoding='utf-8'))
PARITY_CACHE={g:{} for g in GAMMAS}
PARITY_RUN=[]
CONFLICT=load_label_conflict(CURATED,CONCEPT_NAMES,SPANS)
PART_CONFLICT=CONFLICT.groupby('part').agg(n_positive=('positive_images','sum'),
    n_changed=('hidden_positive_images','sum')).reindex(ORDER)
PART_CONFLICT['conflict_rate']=PART_CONFLICT.n_changed/PART_CONFLICT.n_positive
for g in GAMMAS:
    if (g,1) not in HEALTH_DATA:
        raise RuntimeError(f'Required gamma={g} seed-1 ordinary predictions unavailable')
    if HEALTH_DATA[(g,1)]['z'].shape[1]!=26:
        raise RuntimeError('Expected 26 post-head logits')
    if not np.array_equal(HEALTH_DATA[(g,1)]['c'],HEALTH_DATA[(0.,1)]['c']):
        raise RuntimeError('MCBM ordinary label populations/order differ across gamma')
    if not np.array_equal(HEALTH_DATA[(g,1)]['y'],HEALTH_DATA[(0.,1)]['y']):
        raise RuntimeError('MCBM ordinary species populations/order differ across gamma')

def show_standard(tag, expected_source_hash):
    candidates=[c for c in STANDARD_RENDER['cells'] if c['cell_type']=='code'
                and re.fullmatch('fb-'+re.escape(tag)+'-[0-9a-f]+',c.get('id',''))]
    if len(candidates)!=1: raise RuntimeError(f'Standard baseline figure missing: {tag}')
    cell=candidates[0]
    if hashlib.sha256(''.join(cell['source']).encode()).hexdigest()!=expected_source_hash:
        raise RuntimeError(f'Standard {tag} source changed: render current Notebook02 before Notebook03')
    if not cell.get('outputs'):
        raise RuntimeError(f'Standard {tag} has no executed output; run Notebook02 first')
    if any(o['output_type']=='error' for o in cell['outputs']):
        raise RuntimeError(f'Standard {tag} contains an execution error')
    print('STANDARD KOH BASELINE — saved current Notebook02 output:',tag)
    for output in cell['outputs']:
        if 'data' in output:
            display({key:''.join(value) if isinstance(value,list) else value
                     for key,value in output['data'].items()},raw=True)
        elif output.get('output_type')=='stream':
            print(''.join(output.get('text',[])),end='')
    PARITY_RUN.append(dict(figure=tag,model='Koh Standard',status='executed baseline displayed'))
'''

INFORMATION = r'''
# A/B: same conditional-information diagnostic on raw concept z as Standard.
FULL_WIDTH_INFORMATION,EQUAL_WIDTH_INFORMATION=conditional_information(z_saved,c_saved,y_saved,SPANS,progress=lambda s:print('z:',s,flush=True))
# Additional h table: the representation the MCBM species head actually consumes.
H_INFORMATION,H_EQUAL_WIDTH=conditional_information(h_saved,c_saved,y_saved,SPANS,progress=lambda s:print('h:',s,flush=True))
HEAD_USE=replacement_use(h_saved,c_saved,y_saved,ordinary['y_probability'],saved_head,SPANS)
display(score_reference(h_saved,z_saved,c_saved,CONCEPT_NAMES).round(4))
display(FULL_WIDTH_INFORMATION.round(4)); display(EQUAL_WIDTH_INFORMATION.round(4))
print('Additional internal-slot h information (not substituted for the raw-z panels):')
display(H_INFORMATION.round(4)); display(H_EQUAL_WIDTH.round(4)); display(HEAD_USE.round(4))
full=FULL_WIDTH_INFORMATION.set_index('part').reindex(ORDER)
equal=EQUAL_WIDTH_INFORMATION.set_index('part').reindex(ORDER)
head=HEAD_USE.set_index('replaced_block')
fig,axes=plt.subplots(2,2,figsize=(16,10))
axes[0,0].bar(ORDER,full.conditional_logloss_gain,color=[COLORS[p] for p in ORDER])
axes[0,0].set_ylabel('held-out log-loss improvement'); axes[0,0].set_title('A · Extra species information after 0/1 labels are known')
mean=equal.mean_conditional_gain.to_numpy()
axes[0,1].bar(ORDER,mean,color=[COLORS[p] for p in ORDER])
axes[0,1].errorbar(np.arange(5),mean,
 yerr=np.stack([mean-equal.min_conditional_gain.to_numpy(),equal.max_conditional_gain.to_numpy()-mean]),fmt='none',color='black')
axes[0,1].set_ylabel('mean gain; line = subset range'); axes[0,1].set_title('B · Same three-coordinate budget for every part')
axes[1,0].bar(['raw internal h','label-conditioned\nmeans'],
 [head.loc['all 26','raw_accuracy'],head.loc['all 26','accuracy_after_replacement']],color=['#333333','#BBBBBB'])
axes[1,0].set_ylim(0,1); axes[1,0].set_ylabel('saved MCBM head accuracy')
axes[1,0].set_title('C · What the unchanged nonlinear species head uses')
blocks=['all 26']+ORDER
axes[1,1].bar(blocks,head.loc[blocks,'mean_probability_mass_moved'],color=['#333333']+[COLORS[p] for p in ORDER])
axes[1,1].set_ylabel('mean probability mass moved'); axes[1,1].set_title('D · Sensitivity to replacing within-label magnitudes')
for ax in axes.flat: ax.axhline(0,color='black',lw=.5)
fig.suptitle(f'MCBM gamma={GAMMA:g} · Information available versus saved-head sensitivity')
plt.tight_layout(); plt.show()
'''

ERASURE = r'''
h_cf=replay_counterfactual_h(S,GAMMA,CURATED,REPO)
EVIDENCE_ROWS=off_target_erasure(S,h_cf,h_saved,c_saved,saved_head,SPANS)
EVIDENCE_SUMMARY=EVIDENCE_ROWS.groupby('part').agg(
 n=('m_cf','size'), originals=('original_image','nunique'),
 mean_e=('off_target_source_evidence','mean'), median_e=('off_target_source_evidence','median'),
 fraction_e_positive=('off_target_source_evidence',lambda a:float((a>0).mean())),
 mean_pairwise_share_reduction=('pairwise_source_share_reduction','mean'),
 top1_change_rate=('top1_changed','mean'), source_to_donor_pair_flip_rate=('source_to_donor_pair_flip','mean')
).reindex(ORDER)
display(EVIDENCE_SUMMARY.round(4))
fig,axes=plt.subplots(1,3,figsize=(17,5))
for ax,column,scale,title in [
 (axes[0],'off_target_source_evidence',1,'A · Source-over-donor gap reduction after erasure'),
 (axes[1],'pairwise_source_share_reduction',100,'B · Pairwise source-share reduction (percentage points)'),
 (axes[2],'source_minus_donor_logit_before',1,'C · Source-over-donor species gap before erasure')]:
    vals=[EVIDENCE_ROWS.loc[EVIDENCE_ROWS.part==p,column].to_numpy()*scale for p in ORDER]
    boxes=ax.boxplot(vals,tick_labels=ORDER,patch_artist=True,showfliers=False,whis=(5,95))
    for patch,p in zip(boxes['boxes'],ORDER): patch.set_facecolor(COLORS[p]); patch.set_alpha(.65)
    ax.axhline(0,color='black',lw=.8); ax.set_title(title,fontsize=9)
    ax.set_ylabel('percentage points' if scale==100 else 'species-logit units')
fig.suptitle(f'MCBM gamma={GAMMA:g} · Frozen-head off-target erasure, old/new slots untouched')
plt.tight_layout(); plt.show()
'''


def adapt(source, tag):
    if tag == "f8c-source":
        return INFORMATION
    if tag == "f8d-source":
        return ERASURE
    if tag == "f6c":
        source = source[source.index('required={'):]
        source = 'CONFLICT_RESPONSE=conflict_response_table(S,CONFLICT)\n' + source
    if tag == "f9-new":
        source = source[source.index('full=PREDICTIVE'):]
        source = ("recognition=ordinary_value_recognition(z_saved,c_saved,SPANS)\n"
                  "descriptors=add_descriptors(S,CONFLICT,recognition)\n"
                  "PREDICTIVE,ROW_PREDICTIONS=prediction_audit(descriptors)\n"
                  "VALUE_HOLDOUT=value_holdout_audit(descriptors)\n" + source)
    if tag == "r8b-compare":
        source = source[:source.index('display(Markdown(')] + 'display(GROUNDING_COMPARISON.round(4))\n'
    if tag == 'f8':
        start=source.index('species_extremes=')
        end=source.index('extreme_keys=',start)
        source=source[:start]+'''species_extremes=pd.concat([
    pd.concat([group.nsmallest(2,'exact_pair_centered_residual'),
               group.nlargest(2,'exact_pair_centered_residual')])
    for _,group in species_transition.groupby('part')
]).sort_values(['part','exact_pair_centered_residual'])
'''+source[end:]
    # Plot titles must ask a question, not carry the old model's answer.
    source = source.replace('Rarer values tend to have deeper ordinary absent baselines',
                            'Do rarer values have deeper ordinary absent baselines?')
    source = source.replace('VISUAL DIFFICULTY: value misidentified', 'MODEL RECOGNITION: value misidentified')
    source = source.replace('150 held-out images', 'held-out ordinary images (N printed above)')
    source = source.replace('Standard FunnyBird CBM', 'FunnyBird MCBM')
    source = source.replace('standard-CBM', 'MCBM')
    # Keep exact ties visible instead of silently including them in source wins.
    if tag in {'f4b','f7b'}:
        source += "\nprint('Exact final-margin ties (not strict source wins):',int((S.m_cf==0).sum()))\n"
    return source


def assemble(core, md, code):
    standard = build_funnybird(preserve_outputs=False)['cells']
    accounted={find(standard,tag)['id'] for tag in TAGS}
    excluded_prefixes=('fb-setup-','fb-prov-code-','fb-r8c-source-',
                       'fb-r8d-source-','fb-app-evidence-correlation-result-')
    unaccounted=[c['id'] for c in standard if c['cell_type']=='code'
                 and c['id'] not in accounted and not c['id'].startswith(excluded_prefixes)]
    if unaccounted:raise RuntimeError(f'New Standard outputs need explicit MCBM parity: {unaccounted}')
    # Keep model/loss introduction, manifest audit, and MCBM-specific compression,
    # local q_j slope and collapsed-coordinate sensitivity. Drop duplicate plots.
    cut = next(i for i,c in enumerate(core) if c['id'].startswith('03-f11-new-') and c['cell_type']=='markdown')
    result = copy.deepcopy(core[:cut])
    result = [c for c in result if not c['id'].startswith('03-roadmap-')]
    for c in result:
        text=''.join(c['source'])
        text=text.replace('Only gamma-zero-to-positive-gamma changes within MCBM isolate the\nadded regularizer.',
            'Gamma-zero-to-positive-gamma changes within MCBM are the closer regularizer comparison,\nbut independent optimization and limited seed coverage remain competing explanations.')
        text=text.replace('does the one collapsed tail output create', 'does the historically flagged tail output create')
        text=text.replace('Does the one collapsed tail output create', 'Does the historically flagged tail output create')
        text=text.replace('Figure 2b found one exactly constant\noutput:', 'The previous report flagged one exactly constant\noutput (verify against the current Figure 2b):')
        if c['id'].startswith('03-f2d-') and c['cell_type']=='code':
            text=text.replace('tail=SW[SW.part.eq("tail")]', 'tail=SW[SW.part.eq("tail") & SW.seed.eq(1)]')
        c['source']=text.splitlines(keepends=True)
    result += [md('parity-route', r'''
## The loss-engineering story: same questions, new model

1. Did the regularizer compress **h**, and did q_j turn that compression into usable **z**?
2. On the **same changed pixels**, did the named concept become more correct? Keep
   donor wins, positive-but-insufficient response, no response, and ties separate.
3. Which alternatives survive: visibility, conflicting labels, exact-value recognition,
   support, or source-species organization? Test all parts, not only the expected tail.
4. What information remains available, what does the original species reader use,
   and which loss incentive could address the measured failure?

Below, each Standard construction is displayed first, followed by the same
construction separately for each gamma. Shared renderer/data figures are shown
once because the pixels and labels do not change with gamma. This is explicit
parity, not a claim that different architectures have identical raw-score units.

**Population warning:** Standard ordinary diagnostics use its 500 saved images;
MCBM ordinary exports use their own saved population (printed before each graph,
usually 5,000). They are not a matched-image Koh-versus-MCBM accuracy experiment.
Within MCBM, labels/order and split rules are checked across gamma. Swaps are the
same 5,000 directed rows from 250 originals. Fivefold diagnostics hold out ordinary
images; they do not hold out species or provide independent trained-model seeds.

**Scale warning:** raw h/z and class logits are model-specific units. Use within-model
changes and dimensionless outcome fractions for cross-model claims. Three-slot
sampling controls coordinate count, not decoder capacity or all information.

**Checklist scope:** every Standard figure/example/table is enumerated. Its three
result-writing cells are replaced by current MCBM tables and review rules, not copied
as observations. Its Koh setup and provenance cell are replaced by the MCBM input
audit and final execution ledger. Shared label-conflict counts describe the declared
Standard training-view population; they are candidate descriptors, not proof of a
loss mechanism or training-time identity of a historical MCBM artifact.
'''), code('parity-setup', SETUP, 'Input adapters and explicit Standard baseline display; no scientific figure.')]
    for tag in TAGS:
        cell = find(standard, tag)
        source = ''.join(cell['source'])
        index = standard.index(cell)
        preceding = []
        k = index - 1
        while k >= 0 and standard[k]['cell_type'] == 'markdown':
            candidate = standard[k]
            if re.match(r'fb-(?:q|f8b-explain)', candidate['id']):
                preceding.insert(0, ''.join(candidate['source']))
            k -= 1
        # The detailed method was historically after the image; move it before.
        following = standard[index+1:index+2]
        method = ''.join(following[0]['source']) if following and following[0]['id'].startswith('fb-m') else ''
        if tag in {'f8c-source','f8d-source'}:
            preceding, method = [], SPECIAL_METHODS[tag]
        elif tag == 'r8b-compare':
            method = 'Compare the already defined decoding accuracy, mean donorward movement, exact inserted-value recognition, and strict controlled-backwash fraction for each part. No additional model is fitted.'
        text = '\n\n'.join(preceding + [method])
        text = text.replace('500 ordinary', 'the printed number of ordinary').replace('150 held-out', 'the printed number of held-out')
        text = text.replace('visual difficulty', 'model value-recognition difficulty')
        result.append(md('parity-'+tag, f'## Standard construction {tag}\n\n' + text + '\n\n'
            '**Inputs:** frozen Standard baseline, then official MCBM seed 1 at each declared gamma. '
            'The adapters below change the model/data arrays, not the pixel intervention. '
            'Any raw-score axis remains in that model’s own units. Read the printed denominators; '
            'a correlation is not a causal attribution.'))
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        result.append(code('standard-'+tag, f"show_standard({tag!r},{source_hash!r})", cell['metadata'].get('alt',tag)))
        if tag in SHARED:
            result.append(md('shared-'+tag, 'Shared across gamma: this is the identical renderer/data evidence, not six new model results. The MCBM manifest audit above verifies matching replacement bytes.'))
            continue
        adapted = adapt(source, tag)
        for g in GAMMAS:
            preamble = f'''GAMMA={g!r}
ordinary=HEALTH_DATA[(GAMMA,1)]
h_saved,z_saved,c_saved,y_saved=ordinary['h'],ordinary['z'],ordinary['c'],ordinary['y']
y_pred_saved=ordinary['y_probability'].argmax(1)
S=SW[(SW.gamma==GAMMA)&(SW.seed==1)].copy().reset_index(drop=True)
S['donor_gain']=S.z_new-S.z_new_orig
S['source_decrease']=S.z_old_orig-S.z_old
S['responded_but_source_wins']=(S.response_delta>0)&(S.m_cf<0)
S['controlled_event']=S.responded_but_source_wins.astype(int)
checkpoint=REPO/'external/minimal_cbm/results'/f'funnybirds-mcbm-g{{checkpoint_tag(GAMMA)}}'/'1/models/epoch_100.pt'
saved_head=task_head(checkpoint)
print('MCBM gamma=',GAMMA,'ordinary images=',len(y_saved),'swaps=',len(S),'originals=',S.orig_render_id.nunique(),flush=True)
'''
            for name in DEPENDENCIES.get(tag, []):
                preamble += f"{name}=PARITY_CACHE[GAMMA][{name!r}]\n"
            suffix = '\n'
            for name in SAVE.get(tag, []):
                suffix += f"PARITY_CACHE[GAMMA][{name!r}]={name}\n"
            suffix += f"PARITY_RUN.append(dict(figure={tag!r},model=f'MCBM gamma={{GAMMA:g}}',status='analysis executed; interpretation pending visual review'))\n"
            result.append(code(f'parity-{tag}-g{g:g}', preamble+adapted+suffix, f'MCBM gamma={g:g}: '+cell['metadata'].get('alt',tag)))
        result.append(md('reading-'+tag, '**Reading rule after this figure.** The displayed tables and labels are the current literal results; no Standard conclusion is assumed. '
            'Compare every part against gamma zero. Record improvement, worsening, ties, and unavailable strata separately. '
            'A model difference supports only this measured association/intervention; altered score scale, optimization, '
            'and a single causal seed remain alternatives. Independent frozen-render seed replay distinguishes reproducibility. '
            'The next figure tests the next step, not an explanation already established here.'))
    # Preserve the developed matched-recall analysis as a supporting appendix.
    result += [md('recall-placement', '## Supporting appendix: species-matched recognition\n\nThis is a health/species-dependence diagnostic, not a substitute for the controlled swaps.'),
               code('recall-data', 'MODEL_DATA={f"g={g:g}":HEALTH_DATA[(g,1)] for g in GAMMAS}',
                    'Ordinary MCBM populations for the matched-recall appendix.')]
    start=next(i for i,c in enumerate(core) if c['id'].startswith('03-f11-new-') and c['cell_type']=='markdown')
    end=len(core)
    result.extend(copy.deepcopy(core[start:end]))
    result += [md('gradient-method', GRADIENT_METHOD),code('gradient-audit',GRADIENT_CODE,
        'Frozen-checkpoint official loss gradients by part and gamma; not a training trajectory.'),
        md('gradient-boundary','**What this can support:** the displayed gradient magnitudes and directions describe incentives at these frozen checkpoints. '
           '**Alternative:** another noise draw or earlier training epoch may differ. '
           '**Discriminator:** repeat noise draws and inspect saved training checkpoints under a matched recipe. '
           '**Next:** select a candidate loss only when the physical-swap and health results identify the failure it should address.'),
        md('loss-next', LOSS_NEXT), code('loss-ledger', LEDGER,
        'Loss-engineering decision table from current gamma outputs and an explicit execution parity checklist.')]
    return result


SPECIAL_METHODS = {
'f8c-source': r'''
**Question/prediction:** does compression reduce extra species information and saved-head sensitivity together?
Panel A/B fit **new** fivefold species logistic regressions. For each coordinate,
r_ij=z_ij−mean_train(z_j|c_j=c_ij). Give one probe c, the other [c,r].
Log-loss L=−mean_i log P_probe(y_i); gain=L(c)−L(c,r), in natural-log units.
Positive gain means improved held-out prediction, not a percentage of species information.
A uses all coordinates in that part; B uses exactly three (all subsets up to 40,
otherwise 40 deterministic sampled subsets); bars average gains, lines show subset
min/max, **not confidence intervals**. The h versions are printed separately.

C/D reuse the **unchanged original MCBM species MLP F(h)**, not a probe and not Wz+b.
Replace a part's h_j by training-fold mean(h_j|c_j). Example: positive tail_4 values
[2,3,4] give mean3; replace a held-out4.2 with3. True labels determine the bucket;
this is not guaranteed to preserve the model's predicted sign or be deployable without labels.
C compares correct-species fractions before and after replacing all26. D shows
mean_i[0.5 sum_species |P_before−P_after|], replacing each block separately.
For [0.8,0.2]→[0.6,0.4], mass moved=.2. This is replacement sensitivity, not pure
species-information usage. Colors preserve part identity; black is all26.
Denominator: all ordinary exported images, split five ways stratified by species,
seed20260903; all centering/probe fitting uses only the other folds. N/mean/SD
for absent/present h AND z are printed for every coordinate before the plots.
''',
'f8d-source': r'''
**Question/prediction:** after swapping pixels, do other same-part slots still favor
the old species through the saved classifier? Compare distributions **between parts**;
within-part correlations are retained later in the appendix, not the main test.

For tail_2→tail_7 there are always nine fixed tail outputs. Keep slots2 and7 unchanged.
For this diagnostic only, replace the other seven internal h slots by their ordinary
mean when the corresponding c_j=0. Example: absent tail_4 averages−2.8; replace its
counterfactual value−1.2 by−2.8. No image or trained weight is changed.

Let G(h)=F_source(h)−F_donor(h), using the saved **nonlinear** MCBM species head.
e=G(h_cf)−G(h_erased). Positive e means the removed variation favored the source
in this replacement context. It is a joint finite intervention, **not** the sum of
fixed W_j contributions and not an additive decomposition of nonlinear interactions.
No new classifier is trained. The raw old/new concept margin m_cf is unchanged:
each q_j reads only its own untouched h_j. The species head cannot cause an upstream
concept error during this forward pass; task gradients during training are a different question.

A shows e in species-logit units; B shows 100[sigmoid(G_before)−sigmoid(G_after)]
in percentage points; C shows G_before. Boxes show quartiles, middle line median,
whiskers5–95%; omitted outliers are still in all summaries. Fractions positive,
means/medians, original counts, top-one changes and source→donor pair flips are printed.
Example: G_before=5,G_after=4 gives e=1 but both still choose the source pairwise.
Neither pairwise share nor an unrelated top-one change is a whole donor-species takeover.
All5000 swaps are included,1000 perpart; repeated originals are not independent seeds.
Part blocks contain different numbers of erased slots and head scales can differ;
absolute cross-gamma e is not a calibrated information measure. Erasure may remove
visibility or other useful variation, not only a pure species fingerprint.
'''
}

LOSS_NEXT = r'''
## What would justify a better loss—not merely another graph?

The original objective is a penalty on **what h resembles**, not on **where its
evidence came from**. Its coordinate gradient is 0.4γ(h_j−(6c_j−3)) per image
before batch averaging. A label-positive hidden tail and a visible tail receive
the same target. Making h perfectly±3 therefore does not guarantee named pixels
were used. Conversely, suppressing all variation can discard useful visibility
information. These are possible incentives, not established causes of our results.

Koh's concept BCE gradient (unweighted example) is sigmoid(z)−c: for c=1,
z=0 gives−.5 but z=5 gives about−.0067. It rewards correct confident labels without
choosing one common positive magnitude. MCBM's squared penalty adds that pressure
on h, but a learned q_j can amplify or flatten small changes. A nonlinear species
reader can also use interactions. Thus inspect h compression, q_j sensitivity,
z recognition, physical response, and the original reader separately.

| Measured failure, if present | Candidate change to test next | Exact additional term / needed data | What could still go wrong? |
|---|---|---|---|
| h compresses but inserted concepts still lose | add correct physical-swap ranking | λ mean max(0,κ−[z_d,cf−z_s,cf]); known valid donor/source swaps, κ>0 declared before training | ranking may overfit rendered edits; check untouched parts and ordinary health |
| score changes on a part whose pixels were untouched | enforce off-target invariance | λ mean sum_(j outside replaced part)(z_j,cf−z_j,orig)^2 | legitimate interactions/occlusion may change those pixels; validate masks first |
| positive hidden labels associate with poor response | visibility-aware targets (RLv2 chapter) | replace the declared hidden-positive targets, train under the matched protocol | other contextual shortcuts survive; do not declare all residuals explained |
| variation in h vanishes but z becomes flat/miscalibrated | test a simpler or calibrated concept reader | compare fixed monotone q_j with learned q_j under a matched recipe | changes architecture too; not a loss-only causal comparison |
| ordinary z magnitudes vary within the same answer | direct z prototype penalty as a separate ablation | λ mean (z_j−a(2c_j−1))², e.g. a=5 | a neat ±5 answer can still come entirely from species context |

These are **proposed training experiments**, not implemented or validated repairs.
This notebook trains diagnostics only and reuses frozen CBMs/MCBMs. No new scientific
model is submitted. A useful conclusion can be “compression changed X but did not
fix Y; measured failure Y motivates test Z,” without pretending every cause is known.

**Paper boundary:** the MCBM conditional-information motivation and its decoding,
disentanglement and internal-correction tests concern representation properties.
They do not by themselves establish physical part grounding. A weak diagnostic
does not prove every decoder fails; lower task accuracy can reflect optimization.
Gamma0 remains the closest MCBM baseline, but independent initialization and training
outcomes prevent a single-seed sweep from being a pure, replicated causal loss estimate.

**Next chapters:** RLv2 tests the label-conflict intervention, then CUB70 tests how
far these questions survive with released masks but no accepted native donor swap.
Neither observational mask associations nor uncalibrated pasted images inherit
FunnyBird's renderer-quality causal claim.
'''

LEDGER = r'''
decision_rows=[]
for g in GAMMAS:
    s=SW[(SW.gamma==g)&(SW.seed==1)]
    ordinary_health=H[(H.gamma==g)&(H.seed==1)].iloc[0]
    for part in ORDER:
        q=s[s.part==part]
        decision_rows.append(dict(gamma=g,part=part,n=len(q),
            internal_target_RMSE=ordinary_health.target_rmse,
            ordinary_species_accuracy=ordinary_health.species_accuracy,
            donor_finishes_higher=float((q.m_cf>0).mean()),
            donorward_but_source_wins=float(((q.response_delta>0)&(q.m_cf<0)).mean()),
            no_donorward_move=float((q.response_delta<=0).mean()),
            final_ties=float((q.m_cf==0).mean()),
            inserted_value_accuracy=PARITY_CACHE[g]['diag'][part]))
LOSS_DECISIONS=pd.DataFrame(decision_rows)
display(LOSS_DECISIONS.round(4))
display(pd.DataFrame(PARITY_RUN))
print('Computation finished. Scientific interpretation remains pending review of EVERY current image; this is not a trained-model replication claim.')
report_dir=CURATED/'mcbm_notebook03_tables'
report_dir.mkdir(parents=True,exist_ok=True)
LOSS_DECISIONS.to_csv(report_dir/'loss_decisions.csv',index=False)
LOSS_GRADIENTS.to_csv(report_dir/'loss_gradients.csv',index=False)
pd.DataFrame(PARITY_RUN).to_csv(report_dir/'figure_execution_checklist.csv',index=False)
for gamma,results in PARITY_CACHE.items():
    for name,value in results.items():
        if isinstance(value,pd.DataFrame):
            value.to_csv(report_dir/f'g{checkpoint_tag(gamma)}_{name}.csv',index=False)
print('Current source tables:',report_dir)
'''

GRADIENT_METHOD = r'''
## Loss-engineering diagnostic · Which objectives currently push h in competing directions?

**Question/prediction:** if task and concept supervision compete at the internal
slot, their gradients can point in opposite directions. Compression may dominate
or be small at the final checkpoint. Neither pattern is assumed beforehand.

**Inputs/model:** each gamma's frozen official MCBM, saved ordinary h,c,y, its
actual concept-loss positive weights and β/γ. No diagnostic classifier is trained.
Autograd differentiates only copies of h; trained parameters are frozen and no
optimizer or parameter update exists. We use the implementation's training-noise
rule, one deterministic Normal draw per image (seed20260903), not an observed
historical training batch. Batch64 means are multiplied by their actual batch size
to express per-image derivatives, including the short final batch.

Let a=∂L_task/∂h, b=∂(βL_concept)/∂h, r=∂(γL_rep)/∂h. For every part, A/B/C show
sqrt(mean_(image,coordinate) gradient²): gradient RMS per coordinate, not summed
over a wider part. D shows mean cosine(a_part,b_part)=a·b/(||a||||b||) on images
where both norms exceed the declared denominator tolerance1e−12; counts and the
fraction with cosine<0 are printed. Undefined zero-gradient cosines remain missing.
Rows are gamma; columns are the same five parts. Larger RMS means a stronger
local push, not a larger historical causal contribution. D ranges−1 to+1.

Example: a=[1,0],b=[−2,0] gives cosine−1: descending one objective locally raises
the other to first order. a=[1,0],b=[0,1] gives0: orthogonal pushes. A tiny norm
can mean a satisfied objective, saturation or a flat reader, so consult health
and q_j slopes before calling it successful compression or broken learning.
'''

GRADIENT_CODE = r'''
from mcbm_loss_report import loss_gradient_audit
LOSS_GRADIENTS=pd.concat([loss_gradient_audit(
    HEALTH_DATA[(g,1)]['h'],HEALTH_DATA[(g,1)]['c'],HEALTH_DATA[(g,1)]['y'],g,SPANS
) for g in GAMMAS],ignore_index=True)
fig,axes=plt.subplots(1,4,figsize=(19,4.8))
for ax,column,title in zip(axes,
 ['task_gradient_RMS','concept_gradient_RMS','weighted_compression_gradient_RMS','task_concept_cosine_mean'],
 ['A · Task gradient','B · Weighted concept gradient','C · Weighted compression gradient','D · Task/concept alignment']):
    values=LOSS_GRADIENTS.pivot(index='gamma',columns='part',values=column).reindex(index=GAMMAS,columns=ORDER)
    heat(ax,values,title,'cosine' if column.endswith('mean') else 'gradient RMS per coordinate',
         -1 if column.endswith('mean') else 0,1 if column.endswith('mean') else None,
         'coolwarm' if column.endswith('mean') else 'viridis',fmt='.3f')
plt.tight_layout();plt.show();display(LOSS_GRADIENTS.round(5))
'''
