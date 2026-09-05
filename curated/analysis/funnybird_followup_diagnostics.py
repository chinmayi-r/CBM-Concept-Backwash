#!/usr/bin/env python3
"""Focused read-only follow-ups for the FunnyBird Standard-CBM chapter.

This script intentionally contains the complete analysis in one place.  It reads
the accepted Koh Joint ResNet-50 seed-1 evaluation and fixed-render swap table;
it does not train a CBM, render an image, or submit a cluster job.

Outputs are five figures and their source tables:

1. species information available beyond binary labels, and use by saved Wz+b;
2. off-target source evidence used by that saved head during controlled swaps;
2b. direct frozen-head replay after erasing only those off-target scores;
3. exact-value visibility-label conflict versus matched response components;
4. grouped held-out predictability from the measured contributor families.

The fitted logistic/ridge models are analysis-time diagnostics only.  They do
not replace or modify the accepted CBM.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260903
N_FOLDS = 5
ORDER = ["tail", "wing", "beak", "foot", "eye"]
COLORS = {
    "tail": "#6A0DAD",
    "wing": "#0072B2",
    "beak": "#E69F00",
    "foot": "#009E73",
    "eye": "#CC79A7",
}
OUTCOMES = ["donor wins", "donorward, source wins", "no donorward move"]


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"required input is missing or empty: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def controlled_event(frame: pd.DataFrame) -> np.ndarray:
    """Donorward movement while the final donor-minus-source margin is negative."""
    return ((frame["response_delta"] > 0) & (frame["m_cf"] < 0)).to_numpy()


def load_schema(curated: Path) -> tuple[list[str], OrderedDict[str, tuple[int, int]]]:
    module_root = Path(__file__).resolve().parents[1] / "data" / "funnybirds"
    sys.path.insert(0, str(module_root))
    import funnybirds_concepts as fbc

    funnybirds = Path(os.environ.get("FUNNYBIRDS_ROOT", curated / "FunnyBirds"))
    parts = fbc.load_parts(require_file(funnybirds / "parts.json").parent)
    names = fbc.concept_names(parts)
    spans = fbc.group_slices(parts)
    if names != [f"{part}_{value}" for part in spans for value in range(spans[part][1] - spans[part][0])]:
        raise RuntimeError("parts.json does not produce the expected part_value names")
    if len(names) != 26 or set(spans) != set(ORDER):
        raise RuntimeError(f"expected 26 concepts across {ORDER}; got {len(names)} and {list(spans)}")
    return names, spans


def load_evaluation(
    model_root: Path, names: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_parquet(require_file(model_root / "final_test.parquet"))
    needed = {"image", "y_true", "y_pred", "concept_index", "concept_name", "z", "gt_label"}
    missing = needed - set(table)
    if missing:
        raise RuntimeError(f"final_test.parquet lacks {sorted(missing)}")
    if table.duplicated(["image", "concept_index"]).any():
        raise RuntimeError("final_test.parquet repeats an image/concept pair")
    image_order = table["image"].drop_duplicates().tolist()
    if len(image_order) != 500 or len(table) != 500 * 26:
        raise RuntimeError(f"expected 500 x 26 evaluation rows; found {len(image_order)} x {len(table) // max(len(image_order), 1)}")
    concept_table = table[["concept_index", "concept_name"]].drop_duplicates().sort_values("concept_index")
    if concept_table.concept_index.tolist() != list(range(26)):
        raise RuntimeError("evaluation concept indices are not exactly 0..25")
    if concept_table.concept_name.tolist() != names:
        raise RuntimeError("evaluation concept names disagree with parts.json")
    z = table.pivot(index="image", columns="concept_index", values="z").reindex(image_order).to_numpy(float)
    c = table.pivot(index="image", columns="concept_index", values="gt_label").reindex(image_order).to_numpy(int)
    image_rows = table[["image", "y_true", "y_pred"]].drop_duplicates("image").set_index("image").reindex(image_order)
    y = image_rows.y_true.to_numpy(int)
    y_pred = image_rows.y_pred.to_numpy(int)
    if set(y) != set(range(50)) or np.bincount(y, minlength=50).min() < N_FOLDS:
        raise RuntimeError("evaluation must contain all 50 species with enough rows for five folds")
    if not np.isfinite(z).all() or not np.isin(c, [0, 1]).all():
        raise RuntimeError("evaluation logits must be finite and labels binary")
    return table, z, c, y, y_pred


def block_columns(table: pd.DataFrame, part: str, width: int) -> list[str]:
    columns = sorted(
        [name for name in table if name.startswith(f"z_cf_{part}_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if columns != [f"z_cf_{part}_{value}" for value in range(width)]:
        raise RuntimeError(f"{part}: expected {width} ordered post-swap logits; got {columns}")
    return columns


def load_swaps(
    swap_root: Path,
    spans: OrderedDict[str, tuple[int, int]],
    csv_name: str = "funnybirds-cbm-s1.csv",
) -> tuple[pd.DataFrame, str]:
    table = pd.read_csv(require_file(swap_root / csv_name))
    needed = {
        "part", "var_src", "var_donor", "sid_src", "sid_donor",
        "z_new", "z_old", "z_new_orig", "z_old_orig", "margin",
        "response_delta", "pixel_count_cf", "orig_render_id",
    }
    missing = needed - set(table)
    if missing:
        raise RuntimeError(f"swap table lacks {sorted(missing)}")
    if len(table) != 5000 or table.orig_render_id.nunique() != 250:
        raise RuntimeError(
            f"expected 5,000 swaps from 250 originals; found {len(table)} and "
            f"{table.orig_render_id.nunique()}"
        )
    if table.groupby("part").size().reindex(ORDER).tolist() != [1000] * 5:
        raise RuntimeError("expected exactly 1,000 accepted rows for each part")
    if table.groupby("orig_render_id").sid_src.nunique().max() != 1:
        raise RuntimeError("one original render maps to multiple source species")
    table = table.copy()
    table["m_orig"] = table.z_new_orig - table.z_old_orig
    table["m_cf"] = table.z_new - table.z_old
    table["donor_gain"] = table.z_new - table.z_new_orig
    table["source_decrease"] = table.z_old_orig - table.z_old
    if not np.allclose(table.m_cf, table.margin, atol=1e-8, rtol=1e-8):
        raise RuntimeError("stored margin is not z_new - z_old")
    if not np.allclose(
        table.m_cf,
        table.m_orig + table.donor_gain + table.source_decrease,
        atol=1e-8,
        rtol=1e-8,
    ):
        raise RuntimeError("m_cf decomposition does not close")
    table["controlled_event"] = controlled_event(table)
    table["outcome"] = np.select(
        [table.m_cf > 0, table.controlled_event],
        OUTCOMES[:2],
        default=OUTCOMES[2],
    )
    for part, (lo, hi) in spans.items():
        width = hi - lo
        block_columns(table, part, width)
        rows = table.part == part
        if not table.loc[rows, "var_src"].between(0, width - 1).all():
            raise RuntimeError(f"{part}: source value outside 0..{width - 1}")
        if not table.loc[rows, "var_donor"].between(0, width - 1).all():
            raise RuntimeError(f"{part}: donor value outside 0..{width - 1}")
    return table, "orig_render_id"


def same_record_except_label(left: dict, right: dict) -> bool:
    keys = set(left) | set(right)
    for key in keys - {"attribute_label"}:
        if key not in left or key not in right:
            return False
        a, b = left[key], right[key]
        if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                return False
        elif a != b:
            return False
    return True


def load_label_conflict(
    curated: Path, names: list[str], spans: OrderedDict[str, tuple[int, int]]
) -> pd.DataFrame:
    positive = np.zeros(26, dtype=int)
    hidden_positive = np.zeros(26, dtype=int)
    positive_species = [set() for _ in range(26)]
    for split in ("train", "val"):
        paths = {
            label: curated / "koh_joint_inputs" / "funnybirds" / label / f"{split}.pkl"
            for label in ("standard", "rlv2")
        }
        standard = pickle.loads(require_file(paths["standard"]).read_bytes())
        rlv2 = pickle.loads(require_file(paths["rlv2"]).read_bytes())
        if len(standard) != len(rlv2):
            raise RuntimeError(f"{split}: Standard and RLv2 row counts differ")
        for row_index, (left, right) in enumerate(zip(standard, rlv2)):
            if not same_record_except_label(left, right):
                raise RuntimeError(f"{split} row {row_index}: views differ outside attribute_label")
            c_standard = np.asarray(left["attribute_label"], dtype=int)
            c_rlv2 = np.asarray(right["attribute_label"], dtype=int)
            if c_standard.shape != (26,) or c_rlv2.shape != (26,):
                raise RuntimeError(f"{split} row {row_index}: concept width is not 26")
            if not np.isin(c_standard, [0, 1]).all() or not np.isin(c_rlv2, [0, 1]).all():
                raise RuntimeError(f"{split} row {row_index}: labels are not binary")
            positive += c_standard == 1
            hidden_positive += (c_standard == 1) & (c_rlv2 == 0)
            species = int(left["class_label"])
            for concept in np.flatnonzero(c_standard == 1):
                positive_species[concept].add(species)
    part_for_index = {
        index: part for part, (lo, hi) in spans.items() for index in range(lo, hi)
    }
    rows = []
    for index, name in enumerate(names):
        part = part_for_index[index]
        rows.append(
            {
                "concept_index": index,
                "concept": name,
                "part": part,
                "value": index - spans[part][0],
                "positive_images": int(positive[index]),
                "hidden_positive_images": int(hidden_positive[index]),
                "conflict_rate": (
                    float(hidden_positive[index] / positive[index])
                    if positive[index]
                    else np.nan
                ),
                "species_support": len(positive_species[index]),
            }
        )
    result = pd.DataFrame(rows)
    if result.conflict_rate.isna().any():
        concepts = result.loc[result.conflict_rate.isna(), "concept"].tolist()
        raise RuntimeError(f"conflict is undefined for zero-positive concepts: {concepts}")
    return result


def load_saved_head(model_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the accepted Koh model after installing its actual module paths."""
    curated_repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(curated_repo / "compat"))
    sys.path.insert(0, str(curated_repo / "external" / "ConceptBottleneck"))
    import torch

    model_path = require_file(model_root / "final_model_1.pth")
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location="cpu")
    if getattr(model, "curated_framework", None) != "koh_joint":
        raise RuntimeError("checkpoint is not marked as the accepted Koh Joint framework")
    if getattr(model, "curated_backbone", None) != "resnet50":
        raise RuntimeError("checkpoint is not marked as the accepted ResNet-50 backbone")
    if not hasattr(model, "sec_model") or not hasattr(model.sec_model, "linear"):
        raise RuntimeError("checkpoint has no Koh sec_model.linear class head")
    head = model.sec_model.linear
    weight = head.weight.detach().cpu().numpy()
    bias = head.bias.detach().cpu().numpy()
    if weight.shape != (50, 26) or bias.shape != (50,):
        raise RuntimeError(f"unexpected saved-head shapes: {weight.shape}, {bias.shape}")
    return weight, bias


def residualize_from_training(
    z_train: np.ndarray,
    c_train: np.ndarray,
    z_test: np.ndarray,
    c_test: np.ndarray,
) -> np.ndarray:
    residual = np.empty_like(z_test, dtype=float)
    for concept in range(z_train.shape[1]):
        for label in (0, 1):
            train_values = z_train[c_train[:, concept] == label, concept]
            if not len(train_values):
                raise RuntimeError(f"no training rows for concept {concept}, label {label}")
            selected = c_test[:, concept] == label
            residual[selected, concept] = z_test[selected, concept] - train_values.mean()
    return residual


def diagnostic_classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, random_state=SEED),
    )


def conditional_information(
    z: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    spans: OrderedDict[str, tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    def evaluate(columns: np.ndarray) -> float:
        label_prob = np.zeros((len(y), 50))
        combined_prob = np.zeros((len(y), 50))
        for train, test in splitter.split(z, y):
            residual = residualize_from_training(
                z[train][:, columns], c[train][:, columns],
                z[test][:, columns], c[test][:, columns],
            )
            label_model = diagnostic_classifier()
            combined_model = diagnostic_classifier()
            label_model.fit(c[train][:, columns], y[train])
            train_residual = residualize_from_training(
                z[train][:, columns], c[train][:, columns],
                z[train][:, columns], c[train][:, columns],
            )
            combined_model.fit(
                np.column_stack([c[train][:, columns], train_residual]), y[train]
            )
            label_prob[test] = label_model.predict_proba(c[test][:, columns])
            combined_prob[test] = combined_model.predict_proba(
                np.column_stack([c[test][:, columns], residual])
            )
        return float(log_loss(y, label_prob) - log_loss(y, combined_prob))

    full_rows = []
    subset_rows = []
    for part, (lo, hi) in spans.items():
        columns = np.arange(lo, hi)
        full_gain = evaluate(columns)
        full_rows.append(
            {
                "part": part,
                "coordinates": len(columns),
                "conditional_logloss_gain": full_gain,
                "gain_per_coordinate_descriptive": full_gain / len(columns),
            }
        )
        combinations = list(itertools.combinations(columns.tolist(), 3))
        if len(combinations) > 40:
            rng = np.random.default_rng(SEED)
            chosen = rng.choice(len(combinations), size=40, replace=False)
            combinations = [combinations[index] for index in sorted(chosen)]
        gains = [evaluate(np.asarray(combo)) for combo in combinations]
        subset_rows.append(
            {
                "part": part,
                "three_coordinate_subsets": len(gains),
                "mean_conditional_gain": float(np.mean(gains)),
                "min_conditional_gain": float(np.min(gains)),
                "max_conditional_gain": float(np.max(gains)),
            }
        )
    return pd.DataFrame(full_rows), pd.DataFrame(subset_rows)


def saved_head_use(
    z: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    exported_prediction: np.ndarray,
    spans: OrderedDict[str, tuple[int, int]],
    weight: np.ndarray,
    bias: np.ndarray,
) -> pd.DataFrame:
    raw_logits = z @ weight.T + bias
    raw_prediction = raw_logits.argmax(axis=1)
    if not np.array_equal(raw_prediction, exported_prediction):
        disagreement = int(np.sum(raw_prediction != exported_prediction))
        raise RuntimeError(f"saved Wz+b disagrees with exported predictions for {disagreement} images")
    raw_probability = stable_softmax(raw_logits)
    specifications: list[tuple[str, np.ndarray]] = [("all 26", np.arange(26))]
    specifications.extend(
        (part, np.arange(lo, hi)) for part, (lo, hi) in spans.items()
    )
    altered_logits = {name: np.full_like(raw_logits, np.nan) for name, _ in specifications}
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for train, test in splitter.split(z, y):
        means = np.empty((26, 2), dtype=float)
        for concept in range(26):
            for label in (0, 1):
                values = z[train][c[train, concept] == label, concept]
                if not len(values):
                    raise RuntimeError(f"no fold mean for concept {concept}, label {label}")
                means[concept, label] = values.mean()
        expected = means[np.arange(26)[None, :], c[test]]
        for name, columns in specifications:
            altered = z[test].copy()
            altered[:, columns] = expected[:, columns]
            altered_logits[name][test] = altered @ weight.T + bias
    rows = []
    raw_accuracy = accuracy_score(y, raw_prediction)
    for name, columns in specifications:
        logits = altered_logits[name]
        if not np.isfinite(logits).all():
            raise RuntimeError(f"incomplete out-of-fold saved-head replacement: {name}")
        prediction = logits.argmax(axis=1)
        probability = stable_softmax(logits)
        rows.append(
            {
                "replaced_block": name,
                "coordinates_replaced": len(columns),
                "raw_accuracy": raw_accuracy,
                "accuracy_after_replacement": accuracy_score(y, prediction),
                "top1_change_rate": float(np.mean(prediction != raw_prediction)),
                "mean_probability_mass_moved": float(
                    np.mean(0.5 * np.abs(probability - raw_probability).sum(axis=1))
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_information_and_use(
    information: pd.DataFrame,
    subsets: pd.DataFrame,
    head_use: pd.DataFrame,
    output: Path,
) -> None:
    info = information.set_index("part").reindex(ORDER).reset_index()
    sub = subsets.set_index("part").reindex(ORDER).reset_index()
    head = head_use.set_index("replaced_block")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(info.part, info.conditional_logloss_gain, color=[COLORS[p] for p in info.part])
    axes[0, 0].set_ylabel("held-out log-loss improvement")
    axes[0, 0].set_title("A · Extra species information after 0/1 labels are known")
    axes[0, 1].bar(sub.part, sub.mean_conditional_gain, color=[COLORS[p] for p in sub.part])
    axes[0, 1].vlines(sub.part, sub.min_conditional_gain, sub.max_conditional_gain, color="black", lw=1)
    axes[0, 1].set_ylabel("mean gain; line = subset range")
    axes[0, 1].set_title("B · Same three-coordinate budget for every part")
    all_row = head.loc["all 26"]
    axes[1, 0].bar(
        ["raw scores", "label-conditioned\nmeans"],
        [all_row.raw_accuracy, all_row.accuracy_after_replacement],
        color=["#333333", "#BBBBBB"],
    )
    axes[1, 0].set_ylim(0.985, 1.001)
    axes[1, 0].set_ylabel("saved CBM head accuracy")
    axes[1, 0].set_title("C · Accuracy before/after removing within-label magnitudes")
    for index, value in enumerate(
        [all_row.raw_accuracy, all_row.accuracy_after_replacement]
    ):
        axes[1, 0].text(index, value + 0.00025, f"{value:.3f}", ha="center")
    mass_rows = head.loc[["all 26"] + ORDER].reset_index()
    axes[1, 1].bar(
        mass_rows.replaced_block,
        100 * mass_rows.mean_probability_mass_moved,
        color=["#333333"] + [COLORS[p] for p in ORDER],
    )
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].set_ylabel("mean class-probability mass moved (%)")
    axes[1, 1].set_title("D · Saved-head sensitivity to magnitude removal")
    for index, value in enumerate(100 * mass_rows.mean_probability_mass_moved):
        axes[1, 1].text(index, value + 0.025, f"{value:.2f}%", ha="center", fontsize=8)
    fig.suptitle("Follow-up 1 · Species fingerprints: available information versus actual saved-head use")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def absent_label_means(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    means = np.empty(z.shape[1], dtype=float)
    for concept in range(z.shape[1]):
        values = z[c[:, concept] == 0, concept]
        if not len(values):
            raise RuntimeError(f"concept {concept} has no absent-label reference rows")
        means[concept] = values.mean()
    return means


def off_target_saved_head(
    swaps: pd.DataFrame,
    spans: OrderedDict[str, tuple[int, int]],
    absent_means: np.ndarray,
    weight: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for index, record in swaps.iterrows():
        part = str(record.part)
        lo, hi = spans[part]
        width = hi - lo
        values = record[block_columns(swaps, part, width)].to_numpy(float)
        residual = values - absent_means[lo:hi]
        keep = np.ones(width, dtype=bool)
        keep[[int(record.var_src), int(record.var_donor)]] = False
        difference = weight[int(record.sid_src), lo:hi] - weight[int(record.sid_donor), lo:hi]
        evidence = float(difference[keep] @ residual[keep])
        rows.append(
            {
                "row_index": index,
                "part": part,
                "var_src": int(record.var_src),
                "var_donor": int(record.var_donor),
                "original_image": str(record.orig_render_id),
                "outcome": record.outcome,
                "controlled_event": bool(record.controlled_event),
                "m_cf": float(record.m_cf),
                "offtarget_coordinates": int(keep.sum()),
                "offtarget_source_minus_donor_evidence": evidence,
            }
        )
    detail = pd.DataFrame(rows)
    pair = ["part", "var_src", "var_donor"]
    detail["evidence_within_pair"] = detail.offtarget_source_minus_donor_evidence - detail.groupby(pair).offtarget_source_minus_donor_evidence.transform("mean")
    detail["margin_within_pair"] = detail.m_cf - detail.groupby(pair).m_cf.transform("mean")
    summary = []
    for part, group in detail.groupby("part"):
        correlation = group.evidence_within_pair.rank().corr(group.margin_within_pair.rank())
        group = group.copy()
        group["evidence_fifth"] = pd.qcut(
            group.evidence_within_pair, 5, labels=False, duplicates="raise"
        ) + 1
        for fifth, selected in group.groupby("evidence_fifth"):
            summary.append(
                {
                    "part": part,
                    "rank_correlation_with_final_margin": correlation,
                    "evidence_fifth": int(fifth),
                    "n_rows": len(selected),
                    "n_originals": selected.original_image.nunique(),
                    "mean_evidence_within_pair": selected.evidence_within_pair.mean(),
                    "controlled_event_rate": selected.controlled_event.mean(),
                    "median_margin_within_pair": selected.margin_within_pair.median(),
                }
            )
    return detail, pd.DataFrame(summary)


def direct_offtarget_erasure(
    swaps: pd.DataFrame,
    spans: OrderedDict[str, tuple[int, int]],
    absent_means: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    model_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Replay swaps, erase only same-part off-target scores, and rerun saved Wz+b."""
    import torch
    from torchvision import transforms

    needed = {"image_cf_sha256", "image_cf_path", "z_old", "z_new"}
    if missing := needed - set(swaps):
        raise RuntimeError(f"direct erasure needs swap columns {sorted(missing)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "direct off-target erasure requires CUDA to replay the accepted CUDA inference; "
            "it performs no training"
        )
    curated_repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(curated_repo / "compat"))
    sys.path.insert(0, str(curated_repo / "external" / "ConceptBottleneck"))
    model_path = require_file(model_root / "final_model_1.pth")
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)
    model = model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[2.0] * 3),
        ]
    )

    unique = swaps[["image_cf_sha256", "image_cf_path"]].drop_duplicates(
        "image_cf_sha256"
    )
    if len(unique) != 3040:
        raise RuntimeError(f"expected 3040 unique accepted replacement images; got {len(unique)}")
    replay = {}
    with torch.no_grad():
        for row in unique.itertuples(index=False):
            path = require_file(Path(row.image_cf_path))
            output = model(transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device))
            if not isinstance(output, (list, tuple)) or len(output) != 27:
                raise RuntimeError("unexpected Koh output contract during direct erasure replay")
            replay[str(row.image_cf_sha256)] = torch.cat(
                [value.reshape(-1, 1) for value in output[1:]], dim=1
            )[0].detach().cpu().numpy()
    z = np.vstack([replay[str(value)] for value in swaps.image_cf_sha256]).astype(np.float64)
    w = weight.astype(np.float64)
    b = bias.astype(np.float64)
    absent = absent_means.astype(np.float64)

    source_replay = []
    donor_replay = []
    erased = z.copy()
    rows = []
    for position, record in enumerate(swaps.itertuples()):
        part = str(record.part)
        lo, hi = spans[part]
        source_local = int(record.var_src)
        donor_local = int(record.var_donor)
        source_replay.append(z[position, lo + source_local])
        donor_replay.append(z[position, lo + donor_local])
        keep = np.ones(hi - lo, dtype=bool)
        keep[[source_local, donor_local]] = False
        columns = np.arange(lo, hi)[keep]
        residual = z[position, columns] - absent[columns]
        difference = w[int(record.sid_src), columns] - w[int(record.sid_donor), columns]
        evidence = float(difference @ residual)
        erased[position, columns] = absent[columns]
        rows.append(
            {
                "part": part,
                "source_species": int(record.sid_src),
                "donor_species": int(record.sid_donor),
                "original_image": str(record.orig_render_id),
                "off_target_coordinates": len(columns),
                "off_target_source_evidence": evidence,
            }
        )
    source_replay = np.asarray(source_replay)
    donor_replay = np.asarray(donor_replay)
    replay_audit = {
        "device": str(device),
        "unique_replacement_images": len(unique),
        "source_coordinate_maximum_absolute_difference": float(
            np.max(np.abs(source_replay - swaps.z_old.to_numpy()))
        ),
        "donor_coordinate_maximum_absolute_difference": float(
            np.max(np.abs(donor_replay - swaps.z_new.to_numpy()))
        ),
    }
    if (
        replay_audit["source_coordinate_maximum_absolute_difference"] > 0.02
        or replay_audit["donor_coordinate_maximum_absolute_difference"] > 0.02
    ):
        raise RuntimeError(f"direct-erasure replay exceeds 0.02 raw-logit tolerance: {replay_audit}")

    detail = pd.DataFrame(rows)
    before = z @ w.T + b
    after = erased @ w.T + b
    index = np.arange(len(swaps))
    source = detail.source_species.to_numpy(int)
    donor = detail.donor_species.to_numpy(int)
    gap_before = before[index, source] - before[index, donor]
    gap_after = after[index, source] - after[index, donor]
    identity = gap_before - gap_after
    if not np.allclose(identity, detail.off_target_source_evidence, rtol=1e-12, atol=1e-10):
        raise RuntimeError("direct-erasure class-logit identity failed")
    source_share_before = 1.0 / (1.0 + np.exp(-np.clip(gap_before, -700, 700)))
    source_share_after = 1.0 / (1.0 + np.exp(-np.clip(gap_after, -700, 700)))
    detail["source_minus_donor_logit_before"] = gap_before
    detail["source_minus_donor_logit_after"] = gap_after
    detail["pairwise_source_share_reduction"] = source_share_before - source_share_after
    detail["top1_changed"] = before.argmax(1) != after.argmax(1)
    detail["source_to_donor_pair_flip"] = (gap_before > 0) & (gap_after <= 0)
    summary = (
        detail.groupby("part")
        .agg(
            n_swaps=("part", "size"),
            n_original_images=("original_image", "nunique"),
            off_target_coordinates=("off_target_coordinates", "first"),
            mean_e=("off_target_source_evidence", "mean"),
            median_e=("off_target_source_evidence", "median"),
            fraction_e_positive=("off_target_source_evidence", lambda x: float((x > 0).mean())),
            median_absolute_e=("off_target_source_evidence", lambda x: float(np.median(np.abs(x)))),
            median_absolute_source_minus_donor_gap=(
                "source_minus_donor_logit_before", lambda x: float(np.median(np.abs(x)))
            ),
            mean_pairwise_source_share_reduction=("pairwise_source_share_reduction", "mean"),
            top1_change_rate=("top1_changed", "mean"),
            source_to_donor_pair_flip_rate=("source_to_donor_pair_flip", "mean"),
        )
        .reindex(ORDER)
        .reset_index()
    )
    return detail, summary, replay_audit


def plot_direct_offtarget_erasure(detail: pd.DataFrame, output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    columns = [
        ("off_target_source_evidence", 1.0, "off-target source evidence e_i (class-logit units)",
         "A · Actual saved-head source evidence after the swap"),
        ("pairwise_source_share_reduction", 100.0,
         "reduction in pairwise source share (percentage points)",
         "B · Erase only off-target scores; frozen head rerun"),
        ("source_minus_donor_logit_before", 1.0,
         "absolute source-minus-donor class-logit gap before erasure",
         "C · Existing gap sets probability sensitivity"),
    ]
    for axis, (column, scale, ylabel, subtitle) in zip(axes, columns):
        values = []
        for part in ORDER:
            part_values = detail.loc[detail.part == part, column].to_numpy()
            if column == "source_minus_donor_logit_before":
                part_values = np.abs(part_values)
            values.append(scale * part_values)
        boxes = axis.boxplot(values, tick_labels=ORDER, showfliers=False, patch_artist=True)
        for patch, part in zip(boxes["boxes"], ORDER):
            patch.set_facecolor(COLORS[part])
        if column != "source_minus_donor_logit_before":
            axis.axhline(0, color="black", lw=0.8)
        axis.set_ylabel(ylabel)
        axis.set_title(subtitle)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_off_target(summary: pd.DataFrame, output: Path) -> None:
    correlations = summary.groupby("part").rank_correlation_with_final_margin.first().reindex(ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(correlations.index, correlations.values, color=[COLORS[p] for p in correlations.index])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("within-pair rank correlation")
    axes[0].set_title("A · Negative: source evidence accompanies a lower donor margin")
    for index, value in enumerate(correlations.values):
        axes[0].text(index, value - 0.004, f"{value:.3f}", ha="center", va="top", fontsize=8)
    for part in ORDER:
        group = summary[summary.part == part].sort_values("evidence_fifth")
        axes[1].plot(
            group.evidence_fifth,
            group.controlled_event_rate,
            "o-",
            color=COLORS[part],
            label=part,
        )
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xlabel("off-target source-evidence fifth: low → high")
    axes[1].set_ylabel("controlled-backwash fraction")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("B · Event rate across the same evidence ordering")
    axes[1].legend(fontsize=8)
    fig.suptitle("Follow-up 2 · Does the saved class head use a source fingerprint during swaps?")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def conflict_response_table(
    swaps: pd.DataFrame, conflict: pd.DataFrame
) -> pd.DataFrame:
    donor = (
        swaps.groupby(["part", "var_donor"])
        .agg(
            mean_donor_gain=("donor_gain", "mean"),
            mean_visible_pixels=("pixel_count_cf", "mean"),
            donor_rows=("donor_gain", "size"),
        )
        .reset_index()
        .rename(columns={"var_donor": "value"})
    )
    source = (
        swaps.groupby(["part", "var_src"])
        .agg(
            mean_source_decrease=("source_decrease", "mean"),
            source_rows=("source_decrease", "size"),
        )
        .reset_index()
        .rename(columns={"var_src": "value"})
    )
    result = conflict.merge(donor, on=["part", "value"], validate="one_to_one")
    result = result.merge(source, on=["part", "value"], validate="one_to_one")
    if len(result) != 26:
        raise RuntimeError(f"expected 26 exact-value conflict rows; found {len(result)}")
    return result


def plot_conflict_response(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = [
        ("mean_donor_gain", "mean rise in inserted-value logit", "A · Conflict of the inserted value"),
        ("mean_source_decrease", "mean fall in removed-value logit", "B · Conflict of the removed value"),
    ]
    for axis, (column, ylabel, title) in zip(axes, panels):
        for part in ORDER:
            group = table[table.part == part]
            axis.scatter(
                group.conflict_rate,
                group[column],
                s=30 + 7 * group.species_support,
                color=COLORS[part],
                label=part,
                alpha=0.85,
            )
            # Printing all 26 nearly coincident value numbers was unreadable.
            # Tail is the high-conflict group under investigation; the source
            # CSV retains exact labels for every non-tail value.
            for row in group[group.part == "tail"].itertuples():
                axis.annotate(
                    f"tail_{row.value}",
                    (row.conflict_rate, getattr(row, column)),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_xlabel("fraction of positive training labels hidden by the part mask")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    axes[0].legend(title="part", fontsize=8)
    fig.suptitle("Follow-up 3 · Does label–visibility conflict match the score movement it could weaken?")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def ordinary_value_recognition(
    z: np.ndarray,
    c: np.ndarray,
    spans: OrderedDict[str, tuple[int, int]],
) -> dict[tuple[str, int], float]:
    result = {}
    for part, (lo, hi) in spans.items():
        block_labels = c[:, lo:hi]
        positives_per_image = block_labels.sum(axis=1)
        # Standard labels have exactly one value per part. RLv2 deliberately
        # turns the positive value off when that part is invisible, so a row
        # may legitimately have zero positive values. Multiple positives would
        # still violate the exact-value schema and remain an error.
        if np.any(positives_per_image > 1):
            raise RuntimeError(f"ordinary labels contain multiple positives within {part}")
        predicted_local = z[:, lo:hi].argmax(axis=1)
        for value in range(hi - lo):
            selected = block_labels[:, value] == 1
            if not selected.any():
                raise RuntimeError(f"no ordinary held-out images for {part}_{value}")
            result[(part, value)] = float(np.mean(predicted_local[selected] == value))
    return result


def add_descriptors(
    swaps: pd.DataFrame,
    conflict: pd.DataFrame,
    recognition: dict[tuple[str, int], float],
) -> pd.DataFrame:
    lookup = conflict.set_index(["part", "value"])
    frame = swaps.copy()
    for side, value_column in (("donor", "var_donor"), ("source", "var_src")):
        keys = list(zip(frame.part, frame[value_column].astype(int)))
        frame[f"{side}_support"] = [lookup.loc[key, "species_support"] for key in keys]
        frame[f"{side}_conflict"] = [lookup.loc[key, "conflict_rate"] for key in keys]
        frame[f"{side}_ordinary_recognition"] = [recognition[key] for key in keys]
    frame["log_visible_pixels"] = np.log1p(frame.pixel_count_cf)
    return frame


def make_transform(frame: pd.DataFrame, numeric: list[str], categorical: list[str]):
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            ("categorical", encoder, categorical)
        )
    return ColumnTransformer(transformers)


def source_stratified_folds(frame: pd.DataFrame) -> np.ndarray:
    """Assign whole original images to folds while balancing source species."""
    units = (
        frame[["orig_render_id", "sid_src"]]
        .drop_duplicates()
        .sort_values("orig_render_id")
        .reset_index(drop=True)
    )
    if units.groupby("orig_render_id").sid_src.nunique().max() != 1:
        raise RuntimeError("one original image maps to more than one source species")
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    mapping = {}
    for fold, (_, test) in enumerate(splitter.split(units.orig_render_id, units.sid_src)):
        mapping.update({str(value): fold for value in units.iloc[test].orig_render_id})
    folds = frame.orig_render_id.astype(str).map(mapping).to_numpy()
    if pd.isna(folds).any() or set(folds) != set(range(N_FOLDS)):
        raise RuntimeError("source-stratified image fold assignment is incomplete")
    return folds.astype(int)


def grouped_predictions(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    target: str,
    classification: bool,
    folds: np.ndarray,
) -> np.ndarray:
    prediction = np.full(len(frame), np.nan)
    y = frame[target].to_numpy()
    for fold in range(N_FOLDS):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        transform = make_transform(frame, numeric, categorical)
        if classification:
            model = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
        else:
            model = Ridge(alpha=10.0)
        pipeline = make_pipeline(transform, model)
        pipeline.fit(frame.iloc[train], y[train])
        if classification:
            prediction[test] = pipeline.predict_proba(frame.iloc[test])[:, 1]
        else:
            prediction[test] = pipeline.predict(frame.iloc[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("grouped prediction did not cover every swap row")
    return prediction


def fold_mean_predictions(
    frame: pd.DataFrame,
    target: str,
    folds: np.ndarray,
) -> np.ndarray:
    """Predict each held-out fold using only the other folds' target mean."""
    prediction = np.full(len(frame), np.nan)
    values = frame[target].to_numpy(float)
    for fold in range(N_FOLDS):
        train = folds != fold
        test = folds == fold
        prediction[test] = values[train].mean()
    if not np.isfinite(prediction).all():
        raise RuntimeError("fold-mean baseline did not cover every swap row")
    return prediction


def prediction_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = OrderedDict(
        [
            ("part", ([], ["part"])),
            ("starting margin", (["m_orig"], [])),
            ("visibility", (["log_visible_pixels"], [])),
            ("support", (["donor_support", "source_support"], [])),
            ("label conflict", (["donor_conflict", "source_conflict"], [])),
            (
                "ordinary value recognition",
                (["donor_ordinary_recognition", "source_ordinary_recognition"], []),
            ),
            ("source species", ([], ["sid_src"])),
        ]
    )
    all_numeric = [column for numeric, _ in families.values() for column in numeric]
    all_categorical = [column for _, categorical in families.values() for column in categorical]
    specifications = [
        ("part only", [], ["part"]),
        ("full measured set", all_numeric, all_categorical),
    ]
    for family, (numeric, categorical) in families.items():
        specifications.append(
            (
                f"full minus {family}",
                [column for column in all_numeric if column not in numeric],
                [column for column in all_categorical if column not in categorical],
            )
        )
    rows = []
    predictions = []
    folds = source_stratified_folds(frame)
    baseline_margin = fold_mean_predictions(frame, "m_cf", folds)
    baseline_event = fold_mean_predictions(frame, "controlled_event", folds)
    rows.append(
        {
            "model": "overall mean only",
            "numeric_features": "none",
            "categorical_features": "none",
            "margin_RMSE": float(
                np.sqrt(np.mean((frame.m_cf - baseline_margin) ** 2))
            ),
            "margin_MAE": float(np.mean(np.abs(frame.m_cf - baseline_margin))),
            "event_Brier": brier_score_loss(frame.controlled_event, baseline_event),
        }
    )
    predictions.append(
        pd.DataFrame(
            {
                "row_index": frame.index,
                "model": "overall mean only",
                "m_cf": frame.m_cf,
                "margin_prediction": baseline_margin,
                "controlled_event": frame.controlled_event,
                "event_probability": baseline_event,
            }
        )
    )
    for name, numeric, categorical in specifications:
        margin_prediction = grouped_predictions(
            frame, numeric, categorical, "m_cf", classification=False, folds=folds
        )
        event_prediction = grouped_predictions(
            frame,
            numeric,
            categorical,
            "controlled_event",
            classification=True,
            folds=folds,
        )
        rows.append(
            {
                "model": name,
                "numeric_features": ", ".join(numeric) or "none",
                "categorical_features": ", ".join(categorical) or "none",
                "margin_RMSE": float(np.sqrt(np.mean((frame.m_cf - margin_prediction) ** 2))),
                "margin_MAE": float(np.mean(np.abs(frame.m_cf - margin_prediction))),
                "event_Brier": brier_score_loss(frame.controlled_event, event_prediction),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "row_index": frame.index,
                    "model": name,
                    "m_cf": frame.m_cf,
                    "margin_prediction": margin_prediction,
                    "controlled_event": frame.controlled_event,
                    "event_probability": event_prediction,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def value_holdout_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Stress test a generic descriptor model on one unseen inserted value."""
    numeric = [
        "m_orig", "log_visible_pixels", "donor_support", "source_support",
        "donor_conflict", "source_conflict", "donor_ordinary_recognition",
        "source_ordinary_recognition",
    ]
    rows = []
    for (part, value), test_rows in frame.groupby(["part", "var_donor"]):
        test = test_rows.index
        train = frame.index.difference(test)
        transform = make_transform(frame, numeric, ["part"])
        pipeline = make_pipeline(transform, Ridge(alpha=10.0))
        pipeline.fit(frame.loc[train], frame.loc[train, "m_cf"])
        prediction = pipeline.predict(frame.loc[test])
        rows.append(
            {
                "held_out_part": part,
                "held_out_donor_value": int(value),
                "n_rows": len(test),
                "n_originals": frame.loc[test, "orig_render_id"].nunique(),
                "RMSE": float(np.sqrt(np.mean((frame.loc[test, "m_cf"] - prediction) ** 2))),
                "MAE": float(np.mean(np.abs(frame.loc[test, "m_cf"] - prediction))),
            }
        )
    return pd.DataFrame(rows)


def plot_prediction_audit(table: pd.DataFrame, output: Path) -> None:
    full = table.loc[table.model == "full measured set"].iloc[0]
    omissions = table[table.model.str.startswith("full minus ")].copy()
    omissions["RMSE_increase_when_omitted"] = omissions.margin_RMSE - full.margin_RMSE
    omissions["Brier_increase_when_omitted"] = omissions.event_Brier - full.event_Brier
    labels = omissions.model.str.replace("full minus ", "", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(labels, omissions.RMSE_increase_when_omitted, color="#4477AA")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("increase in held-out RMSE when omitted")
    axes[0].set_title(
        f"A · Final-margin prediction; full RMSE = {full.margin_RMSE:.3f}"
    )
    axes[1].barh(labels, omissions.Brier_increase_when_omitted, color="#CC6677")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("increase in held-out Brier error when omitted")
    axes[1].set_title(
        f"B · Controlled-event prediction; full Brier = {full.event_Brier:.3f}"
    )
    fig.suptitle("Figure 9 · Which measured families add held-out predictive value?")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_method_readme(output: Path, regime: str) -> None:
    text = f"""# FunnyBird {regime} CBM focused follow-ups

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

Panel C is deliberately zoomed and labels both values because removing the
magnitudes changes only four of 500 top predictions. Panel D reports total
class-probability mass moved as a percentage. That is saved-head sensitivity,
not accuracy and not itself a backwash rate.

More explicitly, Panel C does not fit a new classifier. It first sends each
image's original 26 raw concept scores through the CBM's unchanged saved
species head, `Wz+b`. It then replaces every score by the training-fold average
score among images with the same official 0/1 label and sends that altered
26-score vector through the same saved head. For example, an original
`tail_4=+8.0` with label 1 may become the average positive `tail_4` score
`+4.5`. The replacement keeps the entire yes/no concept pattern but removes
unusually strong or weak within-label magnitudes. Accuracy changes from 0.992
(496/500) to 1.000 (500/500): four predictions change, all from wrong to
correct. Thus the saved head is sensitive to magnitude fingerprints, but they
were not needed for ordinary-image accuracy in this sample. This tests actual
saved-head use, whereas Panels A/B test information that a newly fitted probe
could recover; it is not itself the controlled-swap backwash test.

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
logit falls. Point size is species support. The plot labels only tail values
because all 26 numerals overlap near zero; the source table still names every
exact value. The causal test is the matched Standard-versus-RLv2 replay in
notebook 02rl, not this association.

## Follow-up 4: held-out predictability, not causal contribution

Five-fold evaluation keeps every swap from an original image together. The full
diagnostic uses part, starting margin, visible area, source/donor support,
source/donor label conflict, ordinary exact-value recognition, and source
species. Each bar is how much held-out error changes when one entire family is
removed. Correlated families can substitute for each other, so bars are not
causal percentages and do not transfer automatically to CUB. The separate
value-holdout table is only a stress test for a new exact value inside this
FunnyBird system.

The figure first compares the full diagnostic with two references: a
training-fold overall-mean prediction and a part-only prediction. RMSE is in
raw-logit units. Brier error is the mean squared difference between a predicted
event probability and the observed 0/1 event; lower is better. No prediction
for a held-out original image uses swaps from that image.
"""
    (output / "METHOD.md").write_text(text, encoding="utf-8")


def describe_files(paths: Iterable[Path]) -> list[dict]:
    return [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-data", type=Path, default=os.environ.get("CURATED_DATA"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--regime",
        choices=("standard", "rlv2"),
        default="standard",
        help="Accepted FunnyBird label/model regime. The analysis is otherwise identical.",
    )
    args = parser.parse_args()
    if args.curated_data is None:
        raise SystemExit("ERROR: pass --curated-data or export CURATED_DATA")
    curated = args.curated_data
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS.json").exists():
        raise SystemExit(f"ERROR: output already completed: {output}")

    regime = args.regime
    model_root = (
        curated
        / "koh_joint_resnet_accelerated_converged_v1"
        / "funnybirds"
        / regime
        / "seed1"
    )
    swap_root = curated / "swap_koh_joint_resnet_accelerated_converged_v1_seed1"
    require_file(model_root / "SUCCESS.json")
    require_file(swap_root / "SUCCESS.json")
    names, spans = load_schema(curated)
    _evaluation, z, c, y, exported = load_evaluation(model_root, names)
    swap_csv = (
        "funnybirds-cbm-s1.csv"
        if regime == "standard"
        else "funnybirds-cbm-rlv2matched-s1.csv"
    )
    swaps, source_id = load_swaps(swap_root, spans, swap_csv)
    if source_id != "orig_render_id":
        raise RuntimeError("unexpected original-image identity column")
    conflict = load_label_conflict(curated, names, spans)
    weight, bias = load_saved_head(model_root)

    print("[1/4] species information beyond labels and actual saved-head use")
    information, subsets = conditional_information(z, c, y, spans)
    head_use = saved_head_use(z, c, y, exported, spans, weight, bias)
    information.to_csv(output / "followup1_information.csv", index=False)
    subsets.to_csv(output / "followup1_three_coordinate_sensitivity.csv", index=False)
    head_use.to_csv(output / "followup1_saved_head_use.csv", index=False)
    plot_information_and_use(information, subsets, head_use, output / "followup1_information_and_use.png")
    print(information.round(4).to_string(index=False))
    print(head_use.round(4).to_string(index=False))

    print("[2/4] off-target source evidence through the unchanged saved head")
    detail, off_target = off_target_saved_head(swaps, spans, absent_label_means(z, c), weight)
    detail.to_csv(output / "followup2_offtarget_rows.csv", index=False)
    off_target.to_csv(output / "followup2_offtarget_summary.csv", index=False)
    plot_off_target(off_target, output / "followup2_offtarget_saved_head.png")
    print(off_target.round(4).to_string(index=False))

    print("[2b/4] direct off-target erasure through the unchanged saved head")
    erasure_rows, erasure_summary, replay_audit = direct_offtarget_erasure(
        swaps, spans, absent_label_means(z, c), weight, bias, model_root
    )
    erasure_rows.to_csv(output / "followup2b_direct_erasure_rows.csv", index=False)
    erasure_summary.to_csv(output / "followup2b_direct_erasure_summary.csv", index=False)
    plot_direct_offtarget_erasure(
        erasure_rows,
        output / "followup2b_direct_erasure.png",
        f"{regime} · Does the post-swap fingerprint push the saved species head toward source?",
    )
    print("direct-erasure replay audit:", replay_audit)
    print(erasure_summary.round(4).to_string(index=False))

    print("[3/4] label–visibility conflict versus matched response components")
    conflict_response = conflict_response_table(swaps, conflict)
    conflict_response.to_csv(output / "followup3_conflict_response.csv", index=False)
    plot_conflict_response(conflict_response, output / "followup3_conflict_response.png")
    print(conflict_response.round(4).to_string(index=False))

    print("[4/4] grouped held-out predictive audit of all measured contributors")
    recognition = ordinary_value_recognition(z, c, spans)
    features = add_descriptors(swaps, conflict, recognition)
    prediction, prediction_rows = prediction_audit(features)
    value_holdout = value_holdout_audit(features)
    prediction.to_csv(output / "followup4_prediction_models.csv", index=False)
    prediction_rows.to_csv(output / "followup4_prediction_rows.csv", index=False)
    value_holdout.to_csv(output / "followup4_value_holdout.csv", index=False)
    plot_prediction_audit(prediction, output / "followup4_predictive_value.png")
    print(prediction.round(4).to_string(index=False))
    print("Value-holdout stress test:")
    print(value_holdout.round(4).to_string(index=False))

    write_method_readme(output, regime)
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    input_paths = [
        model_root / "SUCCESS.json",
        model_root / "final_test.parquet",
        model_root / "final_model_1.pth",
        swap_root / "SUCCESS.json",
        swap_root / swap_csv,
        Path(os.environ.get("FUNNYBIRDS_ROOT", curated / "FunnyBirds")) / "parts.json",
    ]
    for labels in ("standard", "rlv2"):
        for split in ("train", "val"):
            input_paths.append(curated / "koh_joint_inputs" / "funnybirds" / labels / f"{split}.pkl")
    output_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SUCCESS.json"
    )
    manifest = {
        "regime": regime,
        "status": "SUCCESS",
        "scope": f"post-hoc read-only FunnyBird {regime}-CBM follow-ups",
        "framework": "Koh Joint ResNet-50",
        "seed": 1,
        "training": False,
        "rendering": False,
        "slurm": False,
        "git_commit": commit,
        "inputs": describe_files(input_paths),
        "scripts": describe_files(
            [
                Path(__file__),
                repo / "curated" / "notebooks" / "run_funnybird_followup_diagnostics.sh",
                repo / "curated" / "notebooks" / "run_02rl_notebook.sh",
            ]
        ),
        "outputs": describe_files(output_paths),
        "uncertainty": "single trained seed; no row-resampling error bars",
        "causal_boundary": "controlled swaps establish the event; follow-ups are predictive/descriptive associations",
    }
    atomic_json(output / "SUCCESS.json", manifest)
    print(f"[SUCCESS] {output / 'SUCCESS.json'}")


if __name__ == "__main__":
    main()
