# -*- coding: utf-8 -*-
import base64, html
from pathlib import Path

FIGS = Path("/home/user/CBM-Concept-Backwash/curated/figs/02")
OUT  = Path("/home/user/CBM-Concept-Backwash/curated/figs/cbm_review.html")

def uri(fn):
    b = base64.b64encode((FIGS/fn).read_bytes()).decode()
    return f"data:image/png;base64,{b}"

# num, file, family, title, print, criticise, consider, verdict(keep|caveat|fix), story
F = [
(1,"04_Training-curve-best-task-epoch-best.png","fit","Training curve",
 "Task val 0.64→0.75 (plateau by epoch 50); concept 0.96→0.99. Best task epoch 150.",
 "Only 4 epochs sampled (5/50/100/150), and it flags best=150 while the deletion analysis uses epoch 100.",
 "100 vs 150 differ by &lt;0.01 — immaterial. The model clearly fits.",
 "keep",
 "Necessary control: backwash can't be blamed on underfitting."),
(2,"07_FunnyBirds-CBM-removed-part-concept-rete.png","deletion","Deletion retention per part",
 "Tail visible-only <b>0.16 ±0.04</b>; eye/beak/foot/wing 0.01–0.04. All-removals inflated (tail 0.35).",
 "0.16 is <em>modest</em> — a skeptic notes the tail concept does vanish 84% of the time.",
 "But it is 4–16× every other part, with tight seed bars — the <em>contrast</em> is the result, not the magnitude.",
 "caveat",
 "Supports 'tail most backwashed' — but call it 'highest, though modest', not 'large'."),
(3,"10_species-recoverable-from-bottleneck.png","mechanism","Species recoverable from the bottleneck",
 "Full z / c_preds ≈ <b>0.99</b> (chance 0.02). Per single part: tail 0.19, wing 0.12, beak/foot 0.08, eye 0.06.",
 "The per-part numbers are <em>low</em> — tail's 0.19 means one part's concepts barely predict species.",
 "So the species code lives in the <em>assembled</em> vector, not any one part; tail merely leaks the most.",
 "caveat",
 "Supports the mechanism — reword to 'the concept vector is a species code; tail leaks most of any single part.'"),
(4,"13_fig.png","mechanism","Line them up — retention vs species-code / #variants",
 "Tail sits top-right on both axes (species-code 0.19, 9 variants).",
 "Five points, with tail an extreme outlier — one point drives any apparent correlation.",
 "It is a visual consistency check, not a fit.",
 "caveat",
 "Fine as illustration; label it 'illustrative, n=5' and never quote a correlation."),
(5,"16_Per-species-tail-retained-frac-NOT-unifo.png","mechanism","Per-species tail retention",
 "Sorted 0.66→0; histogram right-skewed — most species &lt;0.15, a tail out to 0.66.",
 "Is the spread just noise (few swaps per species)?",
 "The shape is smooth and one-sided, not symmetric noise → real between-species variation.",
 "keep",
 "Supports 'species-specific, not uniform off-distribution confusion.'"),
(6,"19_On-distribution-donor-concept-correctly-.png","control","On-distribution grounding",
 "Source-present concept +15–20; donor-absent concept −18 to −36 (tail most negative).",
 "This looks like the model is <em>perfectly grounded</em> — seemingly against the whole thesis.",
 "That is exactly the point: on-distribution it IS grounded. Backwash is <b>intervention-only</b>.",
 "keep",
 "The strongest honesty control — 'looks perfect until you swap.'"),
(7,"22_Renderer-swap-z-ordering-per-part-fwd-bw.png","causal","z-ordering, fwd vs bwd",
 "Tail 0.35/0.37 (lowest); wing 0.87/0.82, beak 0.43/0.54, foot 0.87/0.85, eye 0.57/0.63.",
 "Beak and eye straddle the 0.5 line too — it isn't tail-only.",
 "Tail is clearly worst and symmetric, but the failure is <b>graded</b> across parts.",
 "caveat",
 "Supports the causal test; add beak/eye to the narrative."),
(8,"25_Per-swap-margin-dots-below-0-violations-.png","causal","Per-swap margin dots (by direction)",
 "Tail cloud straddles 0 with much mass below; wing/foot ride above 0.",
 "The points are <b>not independent</b> — each original image's z is reused across its swaps, inflating density.",
 "The <em>position</em> (tail centered on 0) is valid; the density is not.",
 "keep",
 "Supports it — caption must say 'read positions, not density.'"),
(9,"28_part-t.png","control","Inspection grid — only the target part changes",
 "Across orig / swap / delete, only the named part changes; camera, light, background fixed.",
 "None — it is a sanity check.",
 "Confirms failures are not a rendering artifact.",
 "keep",
 "Supports the whole swap methodology."),
(10,"32_Margin-per-part-whiskers-5-95-box-below-.png","causal","Margin box per part",
 "Violation rate: tail <b>63%</b>, wing 16%, beak <b>51%</b>, foot 11%, eye <b>40%</b>. Tail box sits below 0.",
 "Beak and eye are heavily violated too — 'tail is the problem' is too narrow.",
 "Correct: it is a gradient (tail worst → foot/wing clean), not a binary.",
 "keep",
 "The figure that forces 'not tail-only' into the story — keep and lead with it."),
(11,"35_Grounding-vs-visibility-ALL-parts-does-t.png","control","Grounding vs visibility, all parts",
 "Tail flat ~0.3–0.6 across pixel bins, peaks 0.6 at 100–200px, never reaches foot ~0.9. Beak/eye climb with pixels.",
 "For beak/eye visibility <em>does</em> explain much — the 'not occlusion' claim isn't universal.",
 "Clean for tail (doesn't climb); mixed for beak/eye.",
 "caveat",
 "Say 'occlusion ruled out for tail; partial for beak/eye.'"),
(12,"39_Filter-low-visibility-swaps.png","control","Filter low-visibility swaps",
 "Tail 0.37→0.48 after the visibility filter; right scatter shows no visibility→margin trend.",
 "The 0.37→0.48 jump could be read as 'occlusion mattered.'",
 "It is only the &lt;50px no-ops being dropped; above that it is flat and the scatter has no slope.",
 "keep",
 "Supports — occlusion does not rescue the tail."),
(13,"42_Tail-concept-confusion-after-swap-diagon.png","causal","Tail concept confusion",
 "Diagonal is dim; column 2 is bright — many donor variants fire as tail_2 (a 'default tail concept').",
 "Argmax ≠ the ordering metric; single seed; the diagonal isn't fully dark; and it is <b>tail-only</b> — no grounded part to contrast.",
 "A hypothesis (default-concept collapse), not proof.",
 "fix",
 "Overclaims as-is. Re-run for the all-part matrix so a grounded part's clean diagonal contrasts. Soften until then."),
(14,"45_Grounding-before-swap-high-y-backwash.png","control","Grounding before swap (scatter)",
 "Donor-absent z well below 0 and below the diagonal; tail (blue) down to −100.",
 "The vertical streaks are the reused-activation artifact again.",
 "Positions still show strong on-distribution grounding, tail most-off when absent.",
 "keep",
 "Supports (intervention-only) — caption the streaks."),
(15,"48_Top-20-concepts-by-z-ordering-violation.png","causal","Top-20 concept slots by violation",
 "The worst-grounded slots are tail variants — they dominate the top of the ranking.",
 "None major — it is a ranking, not a claim of magnitude.",
 "Consistent with every other figure: tail slots are the least grounded.",
 "keep",
 "Supports the tail conclusion."),
(16,"51_Per-source-species-tail-violations.png","mechanism","Per-source-species violations",
 "Sorted violation rate is non-uniform — some source species' tails almost never survive a swap.",
 "Could sparse per-species counts fake the spread?",
 "The smooth one-sided shape argues for real per-species structure.",
 "keep",
 "Supports 'species-specific backwash.'"),
(17,"54_Swapped-in-part-visibility.png","control","Swapped-in part visibility per part",
 "Tail is the smallest / most-often-occluded part.",
 "This is exactly <em>why</em> backwash can't be cleanly separated from training-time occlusion.",
 "It motivates the confound caveat (the RL notebook is the disentangler).",
 "keep",
 "Supports the mechanism AND the confound — keep both readings."),
(18,"57_Downstream-does-a-larger-margin-move-spe.png","limit","Downstream: margin → P(species)",
 "Binned mean tops out at only <b>~0.05</b> — a larger margin barely moves the species probability.",
 "The class-level effect is <b>tiny</b> — tail backwash hardly changes the prediction.",
 "Backwash is a concept-layer phenomenon; it rarely flips the label.",
 "caveat",
 "The paper must claim 'un-grounded concepts', NOT 'wrong predictions.' Do not overclaim downstream harm."),
]

VLABEL = {"keep":"Keep", "caveat":"Keep · reword", "fix":"Fix"}

cards = []
for n,fn,fam,title,pr,cr,co,vd,st in F:
    cards.append(f"""
    <article class="fig" id="f{n}">
      <div class="fig-head">
        <span class="eyebrow">Fig {n:02d} · {fam}</span>
        <span class="chip {vd}">{VLABEL[vd]}</span>
      </div>
      <h2>{html.escape(title)}</h2>
      <div class="frame"><img loading="lazy" alt="{html.escape(title)}" src="{uri(fn)}"></div>
      <dl>
        <div class="row"><dt>Print</dt><dd>{pr}</dd></div>
        <div class="row"><dt>Objection</dt><dd>{cr}</dd></div>
        <div class="row"><dt>Consider</dt><dd>{co}</dd></div>
        <div class="row"><dt>Story fit</dt><dd>{st}</dd></div>
      </dl>
    </article>""")

BODY = f"""
<header class="masthead">
  <p class="kicker">Figure review · referee pass</p>
  <h1>Notebook 02 — FunnyBirds · CBM</h1>
  <p class="dek">Every plot the CBM notebook prints, read cold: what it shows, the strongest
  objection first, then a verdict and how it fits the paper's story. Blue accents match the
  notebook's own palette. Standard model — CBM data is complete, so these are final.</p>
  <div class="summary">
    <h3>Verdict against the story</h3>
    <p>The figures <b>support the core claim</b> — concepts are species-anchored, exposed only
    under intervention, worst for the tail — but three of them force honesty in:</p>
    <ul>
      <li><b>#10 + #7 — graded, not tail-only.</b> beak 51%, eye 40% violations.</li>
      <li><b>#18 — concept-layer only.</b> Near-zero downstream on the class; claim "un-grounded
      concepts," not "wrong predictions."</li>
      <li><b>#13 — overclaims</b> until the all-part confusion re-run.</li>
    </ul>
    <p class="actions">Action items: fix <b>#13</b> (all-part matrix); add the "read positions,
    not density" caption to every swap scatter (#8, #14). Everything else is a keeper.</p>
  </div>
</header>
<main>{''.join(cards)}</main>
<footer><p>Generated from <code>curated/figs/02/</code> · 18 figures · CBM (standard) baseline</p></footer>
"""

CSS = """
:root{
  --paper:#f6f5f2; --surface:#ffffff; --ink:#1b1e24; --muted:#5c626b;
  --accent:#0f6fb3; --rule:#e4e2dc; --frame:#eceae4;
  --keep:#2e7d52; --keep-bg:#e7f2ea; --caveat:#9a6a12; --caveat-bg:#f6efdf;
  --fix:#b23a2b; --fix-bg:#f7e6e2;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#15171b; --surface:#1c1f25; --ink:#e7e8ea; --muted:#9aa0a8;
    --accent:#57a7d8; --rule:#2a2e35; --frame:#12141a;
    --keep:#7cc79b; --keep-bg:#18271e; --caveat:#e0ac52; --caveat-bg:#2a2313;
    --fix:#e88b7d; --fix-bg:#2c1a17;
  }
}
:root[data-theme="light"]{
  --paper:#f6f5f2; --surface:#ffffff; --ink:#1b1e24; --muted:#5c626b;
  --accent:#0f6fb3; --rule:#e4e2dc; --frame:#eceae4;
  --keep:#2e7d52; --keep-bg:#e7f2ea; --caveat:#9a6a12; --caveat-bg:#f6efdf;
  --fix:#b23a2b; --fix-bg:#f7e6e2;
}
:root[data-theme="dark"]{
  --paper:#15171b; --surface:#1c1f25; --ink:#e7e8ea; --muted:#9aa0a8;
  --accent:#57a7d8; --rule:#2a2e35; --frame:#12141a;
  --keep:#7cc79b; --keep-bg:#18271e; --caveat:#e0ac52; --caveat-bg:#2a2313;
  --fix:#e88b7d; --fix-bg:#2c1a17;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:Georgia,"Iowan Old Style",Cambria,"Times New Roman",serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.masthead,main,footer{max-width:60rem;margin-inline:auto;padding-inline:1.4rem}
.masthead{padding-top:3.2rem;padding-bottom:1.2rem}
.kicker,.eyebrow{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  text-transform:uppercase;letter-spacing:.16em;font-size:.7rem;color:var(--accent);font-weight:600}
h1{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-weight:680;
  font-size:clamp(1.8rem,4.5vw,2.7rem);line-height:1.08;letter-spacing:-.015em;
  text-wrap:balance;margin:.5rem 0 .4rem}
.dek{color:var(--muted);font-size:1.02rem;max-width:36rem;margin:0 0 1.6rem}
.summary{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);
  border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:1rem}
.summary h3{font-family:system-ui,sans-serif;font-size:.95rem;letter-spacing:.02em;margin:.1rem 0 .5rem}
.summary p{margin:.4rem 0;font-size:.96rem}
.summary ul{margin:.5rem 0;padding-left:1.1rem}
.summary li{margin:.28rem 0;font-size:.95rem}
.summary .actions{color:var(--muted);font-size:.9rem;border-top:1px solid var(--rule);padding-top:.6rem;margin-top:.7rem}
main{display:flex;flex-direction:column;gap:2.4rem;padding-top:1.6rem}
.fig{border-top:1px solid var(--rule);padding-top:1.3rem}
.fig-head{display:flex;justify-content:space-between;align-items:center;gap:1rem}
.fig h2{font-family:system-ui,sans-serif;font-weight:640;font-size:1.28rem;
  letter-spacing:-.01em;margin:.35rem 0 .9rem;text-wrap:balance}
.frame{background:var(--frame);border:1px solid var(--rule);border-radius:10px;
  padding:.7rem;overflow-x:auto}
.frame img{display:block;max-width:100%;height:auto;margin-inline:auto;border-radius:4px}
dl{margin:1.1rem 0 0;display:flex;flex-direction:column;gap:.05rem}
.row{display:grid;grid-template-columns:7.5rem 1fr;gap:.9rem;padding:.55rem 0;
  border-bottom:1px solid var(--rule)}
.row:last-child{border-bottom:0}
dt{font-family:ui-monospace,Menlo,Consolas,monospace;text-transform:uppercase;
  letter-spacing:.08em;font-size:.68rem;color:var(--muted);padding-top:.28rem}
dd{margin:0;font-size:.99rem}
.chip{font-family:system-ui,sans-serif;font-size:.72rem;font-weight:640;letter-spacing:.03em;
  padding:.2rem .6rem;border-radius:999px;white-space:nowrap}
.chip.keep{color:var(--keep);background:var(--keep-bg)}
.chip.caveat{color:var(--caveat);background:var(--caveat-bg)}
.chip.fix{color:var(--fix);background:var(--fix-bg)}
b{font-weight:680}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em;
  background:var(--frame);padding:.1rem .35rem;border-radius:4px}
footer{color:var(--muted);font-size:.85rem;padding-top:2.4rem;padding-bottom:3rem;
  border-top:1px solid var(--rule);margin-top:2rem}
@media(max-width:560px){.row{grid-template-columns:1fr;gap:.2rem}dt{padding-top:0}}
"""

doc = f"<style>{CSS}</style>\n<title>CBM figure review — notebook 02</title>\n{BODY}"
OUT.write_text(doc, encoding="utf-8")
print("wrote", OUT, f"{OUT.stat().st_size/1024:.0f} KB")
