"""Build Notebook 03rl as an exact analysis-parity RLv2 copy of Notebook 03.

The Standard builder remains the single visual/calculation template.  This
wrapper changes only explicit model/data/output roots, embeds each already
executed Standard-MCBM output immediately before its RLv2 counterpart, and
uses Notebook 02rl for the Koh RLv2 reference blocks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import textwrap
from pathlib import Path

from build_03_standard_mcbm_report import build as build_standard


HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
OUTPUT = CURATED / "notebooks/03rl_funnybirds_mcbm_relabeled.ipynb"

MODEL_PREFIX = "funnybirds-mcbm-rlv2matched"
SWAP_ROOT = "swap_fixed_v3_matched"
PATHWAY_ROOT = "mcbm_swap_pathway_rlv2_v3"
TABLE_ROOT = "mcbm_notebook03rl_tables"


def _source(text: str) -> list[str]:
    return (textwrap.dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def _prefix(cell: dict) -> str:
    identity = cell.get("id", "")
    if identity.startswith("m3rl-"):
        remainder = identity[5:]
    elif identity.startswith("m3-"):
        remainder = identity[3:]
    else:
        return ""
    return remainder.rsplit("-", 1)[0]


def _code(prefix: str, source: str) -> dict:
    digest = hashlib.sha256(source.encode()).hexdigest()[:10]
    return {
        "cell_type": "code", "execution_count": None,
        "id": f"m3rl-{prefix}-{digest}", "metadata": {}, "outputs": [],
        "source": _source(source),
    }


def _markdown(prefix: str, source: str) -> dict:
    digest = hashlib.sha256(source.encode()).hexdigest()[:10]
    return {
        "cell_type": "markdown", "id": f"m3rl-{prefix}-{digest}",
        "metadata": {}, "source": _source(source),
    }


def replace_explicit_roots(text: str) -> str:
    replacements = (
        ("funnybirds-mcbm-{TAG[g]}", "funnybirds-mcbm-rlv2matched-{TAG[g]}"),
        ("swap_fixed_v2_attempt2", SWAP_ROOT),
        ("mcbm_swap_pathway_v3", PATHWAY_ROOT),
        ("mcbm_notebook03_tables", TABLE_ROOT),
        ("mcbm_notebook03_final_all_fronts.csv", "mcbm_notebook03rl_final_all_fronts.csv"),
        ("'Koh Standard',np.nan", "'Koh RLv2',np.nan"),
        ("funnybirds/standard/seed1", "funnybirds/rlv2/seed1"),
        ("'funnybirds-cbm-s1.csv'", "'funnybirds-cbm-rlv2matched-s1.csv'"),
        ("MCBM γ0", "RLv2 MCBM γ0"),
        ("MCBM γ0.1", "RLv2 MCBM γ0.1"),
        ("MCBM γ0.3", "RLv2 MCBM γ0.3"),
        ("MCBM γ1", "RLv2 MCBM γ1"),
        ("MCBM γ3", "RLv2 MCBM γ3"),
        ("MCBM γ5", "RLv2 MCBM γ5"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


STANDARD_OUTPUT_PREFIXES = {
    "health-mcbm", "response-audit", "outcomes", "decomp", "loss-gradients",
    "values", "mcbm-info", "hybrid", "mcbm-residual", "mcbm-downstream",
    "final-table",
}

RESULT_MARKDOWN_PREFIXES = {
    "health-after", "outcomes-after", "decomp-after", "values-after",
    "info-after", "context-after", "residual-after", "downstream-after",
    "final-answer",
}


def build() -> dict:
    notebook = copy.deepcopy(build_standard())
    cells: list[dict] = []
    cells.append(_markdown("title", r"""
    # Chapter 03rl — The same MCBM loss experiment after visibility-aware relabeling

    This is an analysis-parity copy of Chapter 03. It asks whether RLv2 changes
    the MCBM gamma story. Every RLv2 calculation uses the same formula, panel
    layout, denominator, exclusion, and gamma order as Standard MCBM. Before
    each RLv2 result, the exact executed Standard-MCBM output is displayed.

    The only scientific substitutions are explicit: Standard MCBM run names
    become `funnybirds-mcbm-rlv2matched-*`; the fixed-swap CSVs come from the
    matched RLv2 root; and ordinary labels/checkpoints are the RLv2 versions.
    No model is trained and no Slurm job is submitted here.
    """))

    for original in notebook["cells"]:
        prefix = _prefix(original)
        if prefix == "title":
            continue
        cell = copy.deepcopy(original)
        text = replace_explicit_roots("".join(cell.get("source", [])))
        cell["source"] = _source(text)
        cell["id"] = f"m3rl-{prefix}-{hashlib.sha256(text.encode()).hexdigest()[:10]}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        if prefix == "setup":
            cell["source"] += _source(r"""
            STANDARD_MCBM_NOTEBOOK=REPO/'notebooks/03_funnybirds_mcbm.ipynb'
            STANDARD_MCBM=json.loads(STANDARD_MCBM_NOTEBOOK.read_text(encoding='utf-8'))
            SHOWN_STANDARD_MCBM=set()

            def show_standard_mcbm(prefix):
                if prefix in SHOWN_STANDARD_MCBM: return
                matches=[c for c in STANDARD_MCBM['cells'] if c.get('cell_type')=='code' and
                         c.get('id','').startswith('m3-'+prefix+'-')]
                if len(matches)!=1: raise RuntimeError(f'Executed Notebook 03 output missing for {prefix}')
                source_matches=[c for c in build_standard_mcbm_template['cells'] if
                                c.get('cell_type')=='code' and c.get('id','').startswith('m3-'+prefix+'-')]
                if len(source_matches)!=1 or ''.join(matches[0]['source'])!=''.join(source_matches[0]['source']):
                    raise RuntimeError(f'Notebook 03 {prefix} is stale; rerun run_03_standard_mcbm_report.sh')
                if not matches[0].get('outputs') or any(o.get('output_type')=='error' for o in matches[0]['outputs']):
                    raise RuntimeError(f'Notebook 03 {prefix} has no successful output')
                display(Markdown(f'### Executed Standard-MCBM result — {prefix}'))
                emit_standard_cell(matches[0]); SHOWN_STANDARD_MCBM.add(prefix)

            RL_CB_NOTEBOOK=json.loads((REPO/'notebooks/02rl_funnybirds_cbm_relabeled.ipynb').read_text(encoding='utf-8'))
            RL_CB_MAP={
              'f1':['02rl-f3-'], 'f2a':['02rl-f4-'], 'f2b':['02rl-f4-'],
              'f3':['02rl-f7b-'], 'f3b':['02rl-f7b-'], 'f4':['02rl-f5-'], 'f4b':['02rl-f5b-'],
              'f5':['02rl-f9-'], 'f6':['02rl-f10-'], 'f6b':['02rl-f13a-'], 'f6c':['02rl-f13a-'],
              'f7':['02rl-f11-'], 'f7a':['02rl-f11-'], 'f7b':['02rl-f13b-'], 'f7c':['02rl-f13c-'],
              'f8':['02rl-f12-'], 'f8b':['02rl-f13d-'], 'r8b-compare':['02rl-f13d-'],
              'f8c-source':['02rl-f13d-'], 'f8d-source':['02rl-f13e-'],
              'f9-new':['02rl-f14-'], 'f10':['02rl-f15-'],
            }
            SHOWN_RL_CB=set()
            def show_standard(*tags):
                for tag in tags:
                    for wanted in RL_CB_MAP.get(tag,[]):
                        if wanted in SHOWN_RL_CB: continue
                        matches=[c for c in RL_CB_NOTEBOOK['cells'] if c.get('cell_type')=='code' and c.get('id','').startswith(wanted)]
                        if len(matches)!=1 or not matches[0].get('outputs') or any(o.get('output_type')=='error' for o in matches[0].get('outputs',[])):
                            raise RuntimeError(f'Executed Notebook 02rl output missing for {wanted}')
                        display(Markdown(f'**Exact executed Koh Standard/RLv2 comparison — Notebook 02rl `{wanted}`**'))
                        emit_standard_cell(matches[0]); SHOWN_RL_CB.add(wanted)
            """)
            # The runtime needs the current Standard template only to verify
            # that the embedded executed Standard output is not stale.
            cell["source"] += _source(r"""
            from build_03_standard_mcbm_report import build as _build_standard_mcbm
            build_standard_mcbm_template=_build_standard_mcbm()
            """)
        if prefix in STANDARD_OUTPUT_PREFIXES:
            cells.append(_code(f"standard-{prefix}", f"show_standard_mcbm({prefix!r})"))
        if prefix in RESULT_MARKDOWN_PREFIXES:
            cell = _markdown(prefix, r"""
            **How to read the executed RLv2 result.** The figure and fully
            captioned source table immediately above are the result; no expected
            observation was written into the notebook before execution.

            **Question → test → boundary.** Compare the RLv2 output with the
            executed Standard-MCBM output directly above it. A change is evidence
            about visibility-aware labels only when model gamma, formula, physical
            swap rows, and denominator match. It does not turn an association into
            a cause, and seed 1 does not provide initialization uncertainty.

            **Next question.** Continue to the next identical analysis stage, then
            use the final matched-difference table to decide which part and pathway
            actually changed.
            """)
        cells.append(cell)

    # The base final table is now the RLv2 table. Add one direct, complete
    # Standard-minus-RLv2 comparison instead of relying on visual memory.
    insertion = next(i for i, c in enumerate(cells) if _prefix(c).startswith("final-answer"))
    cells.insert(insertion, _code("matched-final-differences", r"""
    standard_final=pd.read_csv(CURATED/'mcbm_notebook03_final_all_fronts.csv')
    rlv2_final=FINAL.copy()
    standard_final=standard_final[standard_final.model.str.startswith('MCBM')].copy()
    standard_final['gamma']=standard_final.model.str.extract(r'γ([0-9.]+)')[0].astype(float)
    compare_columns=['m_orig','donor_gain','source_decrease','response','m_cf','exact_donor',
      'exact_source','exact_third','no_move','backwash','mean_calibrated_h_response',
      'q_breaks_h_success_rate','q_repairs_h_failure_rate','balanced_accuracy','h_target_RMSE',
      'mean_swap_induced_offtarget_source_evidence','pairwise_source_to_donor_flip_rate']
    left=standard_final[['gamma','part']+compare_columns]
    right=rlv2_final[rlv2_final.gamma.notna()][['gamma','part']+compare_columns]
    MATCHED=left.merge(right,on=['gamma','part'],suffixes=('_standard','_rlv2'),validate='one_to_one')
    for column in compare_columns:
        MATCHED[column+'_change_rlv2_minus_standard']=MATCHED[column+'_rlv2']-MATCHED[column+'_standard']
    show_table(
      'complete Standard-versus-RLv2 MCBM matched differences',
      'For every gamma and part, `_standard` and `_rlv2` are the two actual values; '
      '`_change_rlv2_minus_standard` is RLv2 minus Standard. Positive is not always '
      'better: it is better for response/final margin/exact donor, but worse for '
      'backwash/no-move. Read each named quantity by its definition above.',
      MATCHED.round(4))
    for part in ORDER:
        q=MATCHED[MATCHED.part.eq(part)].set_index('gamma')
        display(Markdown(
          f'**{part}:** at gamma 0, RLv2 changes exact donor recognition by '
          f'`{q.loc[0,"exact_donor_change_rlv2_minus_standard"]:+.3f}` and backwash by '
          f'`{q.loc[0,"backwash_change_rlv2_minus_standard"]:+.3f}`; at gamma 5 the '
          f'changes are `{q.loc[5,"exact_donor_change_rlv2_minus_standard"]:+.3f}` and '
          f'`{q.loc[5,"backwash_change_rlv2_minus_standard"]:+.3f}`. The pathway columns '
          'state whether that change first appears in calibrated h or only after q(h).'))
    MATCHED.to_csv(CURATED/'mcbm_notebook03rl_standard_difference_all_fronts.csv',index=False)
    """))

    notebook["cells"] = cells
    return notebook


def main() -> None:
    notebook = build()
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
