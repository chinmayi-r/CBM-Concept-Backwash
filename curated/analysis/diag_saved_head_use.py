"""D6.5 — Does the unchanged saved CBM class head use within-label magnitudes?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import diag_common as dc


def stable_softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_head(model_path):
    import torch

    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location="cpu")
    if not hasattr(model, "sec_model") or not hasattr(model.sec_model, "linear"):
        raise RuntimeError("checkpoint is not the expected Koh Joint model with sec_model.linear")
    head = model.sec_model.linear
    weight = head.weight.detach().cpu().numpy()
    bias = head.bias.detach().cpu().numpy()
    if weight.shape != (50, 26) or bias.shape != (50,):
        raise RuntimeError(f"unexpected saved class-head shapes {weight.shape}, {bias.shape}")
    return weight, bias


def main():
    z, c, y, image_ids, _ = dc.load_eval()
    _, spans = dc.load_concepts()
    root = (dc.curated_root() / "koh_joint_resnet_accelerated_converged_v1" /
            "funnybirds" / "standard" / "seed1")
    model_path = dc._require(root / "final_model_1.pth", "complete accepted convergence")
    evaluation = pd.read_parquet(root / "final_test.parquet")
    exported_rows = evaluation[["image", "y_pred"]].drop_duplicates("image").copy()
    exported_rows["image"] = exported_rows.image.astype(str)
    exported = (exported_rows.set_index("image").reindex(image_ids)
                .y_pred.to_numpy(dtype=int))
    weight, bias = load_head(model_path)
    raw_logits = z @ weight.T + bias
    raw_prediction = raw_logits.argmax(axis=1)
    if not np.array_equal(raw_prediction, exported):
        raise RuntimeError("reconstructed saved-head predictions disagree with export")
    raw_probability = stable_softmax(raw_logits)

    splitter = StratifiedKFold(
        n_splits=dc.N_FOLDS, shuffle=True, random_state=dc.FOLD_SEED)
    specifications = [("all_26", np.arange(26))]
    specifications += [(part, np.arange(lo, hi)) for part, (lo, hi) in spans.items()]
    altered_logits = {name: np.full_like(raw_logits, np.nan) for name, _ in specifications}
    for train, test in splitter.split(z, y):
        expected = z.copy()
        for concept in range(26):
            for label in (0, 1):
                selected = train[c[train, concept].astype(int) == label]
                if not len(selected):
                    raise RuntimeError(
                        f"no training-fold rows for concept {concept}, label {label}")
                selected_test = test[c[test, concept].astype(int) == label]
                expected[selected_test, concept] = z[selected, concept].mean()
        for name, columns in specifications:
            altered = z[test].copy()
            altered[:, columns] = expected[test][:, columns]
            altered_logits[name][test] = altered @ weight.T + bias

    rows = []
    raw_accuracy = float(np.mean(raw_prediction == y))
    for name, columns in specifications:
        logits = altered_logits[name]
        if np.isnan(logits).any():
            raise RuntimeError(f"saved-head OOF replacement is incomplete for {name}")
        prediction = logits.argmax(axis=1)
        probability_shift = 0.5 * np.abs(
            raw_probability - stable_softmax(logits)).sum(axis=1)
        _, shift_lo, shift_hi = dc.clustered_metric_interval(
            probability_shift, np.asarray(image_ids), np.mean)
        rows.append({
            "replaced_block": name,
            "dimensions_replaced": len(columns),
            "raw_saved_head_accuracy": raw_accuracy,
            "accuracy_after_replacement": float(np.mean(prediction == y)),
            "accuracy_drop": raw_accuracy - float(np.mean(prediction == y)),
            "top1_prediction_change_rate": float(np.mean(prediction != raw_prediction)),
            "mean_probability_redistributed": float(np.mean(probability_shift)),
            "probability_redistributed_ci_low": shift_lo,
            "probability_redistributed_ci_high": shift_hi,
        })
    table = pd.DataFrame(rows)
    table.round(5).to_csv(dc.out_dir() / "d65_saved_head_use.csv", index=False)
    print("\nD6.5 · unchanged saved Wz+b head after within-label magnitudes are removed")
    print(table.round(4).to_string(index=False))
    print("\nReading rule: this tests actual use by the saved CBM species head, not just "
          "information available to a newly trained probe. An unchanged top-1 species "
          "can coexist with redistributed probability mass. The replacement is an "
          "analysis-time bottleneck intervention, not a new image or retrained model. "
          "Intervals resample held-out images and are not training-seed uncertainty.")


if __name__ == "__main__":
    main()
