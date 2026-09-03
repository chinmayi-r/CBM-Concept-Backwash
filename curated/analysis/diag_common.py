"""Shared loading, validation, and fold scheme for the predeclared notebook-02
follow-up diagnostics (DECISIONS.md section D.6).

Read-only: verifies and loads the accepted seed-1 artifacts, derives the same
columns the notebook-02 builder derives (with the same identity checks), and
reproduces the Figure 8c grouped fold scheme so every diagnostic shares it.

Run from the project root on Adroit with CURATED_DATA set, e.g.
    python curated/analysis/diag_dimension_adjusted_information.py
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # curated/analysis
REPO = HERE.parent                              # curated
sys.path.insert(0, str(REPO / "data" / "funnybirds"))

FOLD_SEED = 20260901        # identical to the Figure 8c diagnostic
SUBSET_SEED = 20260903      # D6.1 coordinate-subset sensitivity only
N_FOLDS = 5
MIN_ORIGINALS = 25          # 8c interpretability floor for outcome groups
ORDER = ["tail", "wing", "beak", "foot", "eye"]
COLORS = {"tail": "#6A0DAD", "wing": "#0072B2", "beak": "#E69F00",
          "foot": "#009E73", "eye": "#CC79A7"}
SOURCE_ID_CANDIDATES = ["orig_render_id", "source_render_id", "li",
                        "image_orig", "orig_image"]


def _require(path: Path, hint: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}\nProduce it with: {hint}")
    return path


def curated_root() -> Path:
    if "CURATED_DATA" not in os.environ:
        raise RuntimeError("Set CURATED_DATA (see curated/notebooks/run_02_standard_cbm_report.sh)")
    return Path(os.environ["CURATED_DATA"])


def out_dir() -> Path:
    out = Path(os.environ.get(
        "DIAGNOSTICS_OUTPUT_DIR",
        curated_root() / "diagnostics_predeclared_v2",
    ))
    out.mkdir(parents=True, exist_ok=True)
    return out


def controlled_event(frame: pd.DataFrame) -> pd.Series:
    """Canonical event: donorward movement while the final margin remains < 0."""
    return (frame["response_delta"] > 0) & (frame["m_cf"] < 0)


def load_concepts():
    """FunnyBird concept names and per-part index spans, as the builder loads them."""
    import funnybirds_concepts as fbc
    fb_root = Path(os.environ.get("FUNNYBIRDS_ROOT", curated_root() / "FunnyBirds"))
    parts = fbc.load_parts(fb_root)
    names = fbc.concept_names(parts)
    spans = fbc.group_slices(parts)
    if len(names) != 26:
        raise RuntimeError(f"expected 26 exact concepts, found {len(names)}")
    return names, spans


def load_swaps() -> tuple[pd.DataFrame, str, dict]:
    """Accepted seed-1 fixed-swap table with the builder's derived columns and checks.

    Returns (S, source_id_column, spans).
    """
    curated = curated_root()
    swap_root = curated / "swap_koh_joint_resnet_accelerated_converged_v1_seed1"
    _require(swap_root / "SUCCESS.json", "complete accepted converged FunnyBird fixed swaps")
    swap_csv = _require(swap_root / "funnybirds-cbm-s1.csv", "run accepted converged swaps")
    S = pd.read_csv(swap_csv)

    if "response_delta" not in S:
        S["response_delta"] = S.margin - (S.z_new_orig - S.z_old_orig)
    required = {"part", "z_new", "z_old", "z_new_orig", "z_old_orig", "margin",
                "response_delta", "var_src", "var_donor", "sid_src", "sid_donor",
                "pixel_count_cf"}
    missing = required - set(S.columns)
    if missing:
        raise RuntimeError(f"accepted swap CSV is missing {sorted(missing)}; "
                           f"available: {sorted(S.columns)}")

    S["m_orig"] = S.z_new_orig - S.z_old_orig
    S["m_cf"] = S.z_new - S.z_old
    S["donor_gain"] = S.z_new - S.z_new_orig
    S["source_decrease"] = S.z_old_orig - S.z_old
    if not np.allclose(S.m_cf, S.margin):
        raise RuntimeError("stored final margin disagrees with z_new-z_old")
    if not np.allclose(S.m_cf, S.m_orig + S.donor_gain + S.source_decrease):
        raise RuntimeError("starting-margin/response decomposition does not close")
    S["outcome"] = np.select(
        [S.m_cf > 0, controlled_event(S)],
        ["donor wins", "donorward, source wins"],
        default="no donorward move")

    source_id = next((c for c in SOURCE_ID_CANDIDATES if c in S), None)
    if source_id is None:
        raise RuntimeError("swap CSV lacks an original source-image identity column")
    if (S.groupby(source_id).sid_src.nunique().max()) != 1:
        raise RuntimeError("one original image maps to multiple source species")

    _, spans = load_concepts()
    for part, (lo, hi) in spans.items():
        block = [c for c in S.columns if c.startswith(f"z_cf_{part}_")]
        if len(block) != hi - lo:
            raise RuntimeError(f"post-swap block for {part}: found {len(block)} of {hi - lo} columns")
        bad_src = ~S.loc[S.part == part, "var_src"].between(0, hi - lo - 1)
        bad_don = ~S.loc[S.part == part, "var_donor"].between(0, hi - lo - 1)
        if bad_src.any() or bad_don.any():
            raise RuntimeError(f"{part}: var_src/var_donor outside block width {hi - lo}")

    print(f"[diag_common] swaps: {len(S)} rows, "
          f"{S[source_id].nunique()} original images, id column '{source_id}'")
    return S, source_id, spans


def block_columns(S: pd.DataFrame, part: str) -> list[str]:
    cols = [c for c in S.columns if c.startswith(f"z_cf_{part}_")]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def pre_swap_block_columns(S: pd.DataFrame, part: str) -> list[str] | None:
    """Pre-swap per-part blocks, if the CSV carries them (else None)."""
    for prefix in (f"z_orig_{part}_", f"z_pre_{part}_"):
        cols = [c for c in S.columns if c.startswith(prefix)]
        if cols:
            return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    return None


def grouped_folds(S: pd.DataFrame, source_id: str) -> pd.Series:
    """Figure 8c fold scheme: 5 folds over distinct original images, stratified by
    source species, seed 20260901. Returns a per-row fold Series aligned to S."""
    from sklearn.model_selection import StratifiedKFold
    units = (S[[source_id, "sid_src"]].drop_duplicates()
             .sort_values(source_id).reset_index(drop=True))
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    unit_fold = {}
    for fold, (_, test_idx) in enumerate(splitter.split(units[source_id], units.sid_src)):
        for unit in units.iloc[test_idx][source_id].astype(str):
            unit_fold[unit] = fold
    folds = S[source_id].astype(str).map(unit_fold)
    if folds.isna().any() or set(folds.unique()) != set(range(N_FOLDS)):
        raise RuntimeError("grouped fold assignment is incomplete")
    return folds.astype(int)


def load_eval():
    """Held-out ordinary predictions: (z [n,26], gt labels [n,26], species y [n],
    image ids [n], concept names)."""
    curated = curated_root()
    model_root = (curated / "koh_joint_resnet_accelerated_converged_v1"
                  / "funnybirds" / "standard" / "seed1")
    pred = _require(model_root / "final_test.parquet",
                    "complete accepted FunnyBird Standard evaluation")
    EV = pd.read_parquet(pred)
    required = {"image", "y_true", "concept_index", "concept_name", "z", "gt_label"}
    missing = required - set(EV.columns)
    if missing:
        raise RuntimeError(f"evaluation parquet missing {sorted(missing)}")
    if len(EV) != EV.image.nunique() * 26:
        raise RuntimeError("evaluation is not one row per image and exact concept")
    expected_names, _ = load_concepts()
    indices = sorted(EV.concept_index.unique().tolist())
    if indices != list(range(26)):
        raise RuntimeError(f"evaluation concept indices are {indices}, expected 0..25")
    order = EV.image.drop_duplicates().tolist()
    z = EV.pivot(index="image", columns="concept_index", values="z").reindex(order).to_numpy()
    c = EV.pivot(index="image", columns="concept_index", values="gt_label").reindex(order).to_numpy()
    y = (EV[["image", "y_true"]].drop_duplicates("image")
         .set_index("image").reindex(order).y_true.to_numpy(dtype=int))
    names = (EV[["concept_index", "concept_name"]].drop_duplicates()
             .sort_values("concept_index").concept_name.tolist())
    if names != expected_names:
        raise RuntimeError("evaluation concept names/order disagree with parts.json")
    print(f"[diag_common] eval: {len(order)} images, {len(names)} concepts, "
          f"{len(np.unique(y))} species")
    return z, c, y, [str(i) for i in order], names


def _equal_value(a, b) -> bool:
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    return bool(a == b)


def _record_identity(record: dict) -> tuple[str, int, object]:
    image = str(record.get("image", record.get("img_path", ""))).replace("\\", "/")
    return image, int(record["class_label"]), record.get("id")


def load_conflict_rates() -> pd.DataFrame:
    """Per-exact-concept label/visibility conflict rates (Figure 6b quantities),
    recomputed after verifying the two label views row by row."""
    cache = out_dir() / "conflict_exact.csv"
    curated = curated_root()
    names, spans = load_concepts()
    concept_part = {name: part for part, (lo, hi) in spans.items() for name in names[lo:hi]}
    standard_input = curated / "koh_joint_inputs" / "funnybirds" / "standard"
    visibility_input = curated / "koh_joint_inputs" / "funnybirds" / "rlv2"
    positive = np.zeros(len(names), dtype=int)
    changed = np.zeros(len(names), dtype=int)
    for split in ["train", "val"]:
        std = pickle.loads(_require(standard_input / f"{split}.pkl", "prepare standard labels").read_bytes())
        vis = pickle.loads(_require(visibility_input / f"{split}.pkl", "prepare rlv2 labels").read_bytes())
        if len(std) != len(vis):
            raise RuntimeError(f"standard/visibility-aware {split} lengths differ")
        for row_index, (a, b) in enumerate(zip(std, vis)):
            if _record_identity(a) != _record_identity(b):
                raise RuntimeError(
                    f"standard/RLv2 {split} row {row_index} identity differs: "
                    f"{_record_identity(a)!r} != {_record_identity(b)!r}")
            keys = set(a) | set(b)
            differing = [key for key in keys if key != "attribute_label" and
                          (key not in a or key not in b or not _equal_value(a[key], b[key]))]
            if differing:
                raise RuntimeError(
                    f"standard/RLv2 {split} row {row_index} differs outside "
                    f"attribute_label: {differing}")
            ca = np.asarray(a["attribute_label"])
            cb = np.asarray(b["attribute_label"])
            if ca.shape != (len(names),) or cb.shape != (len(names),):
                raise RuntimeError(f"{split} row {row_index} has wrong concept width")
            positive += (ca == 1)
            changed += ((ca == 1) & (cb == 0))
    out = pd.DataFrame({"concept": names,
                        "part": [concept_part[n] for n in names],
                        "value": [i - spans[concept_part[n]][0] for i, n in enumerate(names)],
                        "n_positive": positive, "n_changed": changed})
    out["conflict_rate"] = out.n_changed / out.n_positive.replace(0, np.nan)
    out.to_csv(cache, index=False)
    undefined = out.loc[out.conflict_rate.isna(), "concept"].tolist()
    print(f"[diag_common] conflict rates recomputed at {cache}; "
          f"undefined zero-positive concepts={undefined}")
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clustered_metric_interval(values: np.ndarray, groups: np.ndarray,
                              statistic, seed: int = FOLD_SEED,
                              repeats: int = 2000) -> tuple[float, float, float]:
    """Point estimate and 95% cluster-bootstrap interval over original images.

    This quantifies within-model sampling variation only. It is not training-seed
    uncertainty and must never be presented as such.
    """
    values = np.asarray(values)
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    point = float(statistic(values))
    rng = np.random.default_rng(seed)
    draws = []
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([positions[group] for group in sampled])
        draws.append(float(statistic(values[idx])))
    return point, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
