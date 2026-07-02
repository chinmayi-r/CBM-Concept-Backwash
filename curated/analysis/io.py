"""Normalized evaluation table: the single currency the notebooks consume.

Both frameworks dump per-image, per-concept inference into one long-format table
so all downstream plotting/occlusion code is framework-agnostic.

EVAL_SCHEMA (one row per image x concept):
    image          str   image stem (joins to the visibility tables)
    class_label    int
    concept_idx    int
    concept_name   str
    part           str   coarse body part (CUB) or group (FunnyBirds); may be ""
    z              float pre-sigmoid concept logit / latent (CBM: c_logit; MCBM: z)
    prob           float sigmoid(z)
    gt_label       int   ground-truth concept label (0/1)
    pred_label     int   prob >= 0.5
And image-level columns repeated on every row: y_true, y_pred (int).

`build_eval_table` is the one function to implement per framework -- it runs the
trained model over a split and emits the schema above. Stubs below document the
exact entry points; fill them in on the cluster where the checkpoints live.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd

EVAL_COLUMNS = [
    "image", "class_label", "concept_idx", "concept_name", "part",
    "z", "prob", "gt_label", "pred_label", "y_true", "y_pred",
]


def load_eval_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(EVAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"eval table {path} missing columns: {sorted(missing)}")
    return df


def save_eval_table(df: pd.DataFrame, path: str | Path) -> None:
    df[EVAL_COLUMNS].to_parquet(path, index=False)


# ----------------------------------------------------------------------------
# Framework-specific builders (run on the cluster; documented entry points).
# ----------------------------------------------------------------------------
def _emit(image, class_label, concept_names, z, prob, gt, y_true, y_pred,
          part_of=None) -> list[dict]:
    """Expand one image's per-concept arrays into EVAL_SCHEMA rows."""
    rows = []
    for j, name in enumerate(concept_names):
        rows.append({
            "image": image, "class_label": int(class_label),
            "concept_idx": j, "concept_name": name,
            "part": (part_of(name) if part_of else "") or "",
            "z": float(z[j]), "prob": float(prob[j]),
            "gt_label": int(gt[j]), "pred_label": int(prob[j] >= 0.5),
            "y_true": int(y_true), "y_pred": int(y_pred),
        })
    return rows


def build_eval_table_cbm(model_ckpt, data_pkl, concept_names, model2_ckpt=None,
                         part_of=None, batch_size=64, device="cuda",
                         image_dir="images") -> pd.DataFrame:
    """Run an official ConceptBottleneck checkpoint over a split -> EVAL_SCHEMA.

    Faithful to external/ConceptBottleneck/CUB/inference.py (verified):
      outputs = model(x);  attr logits = outputs[1:] (one tensor per concept),
      stacked to [B, n_attr]; sigmoid -> prob. For Independent/Sequential the
      class head is a second model over the concept vector; for Joint the class
      logits are outputs[0].

    `data_pkl`  : the split pickle produced by our build scripts (has image stem,
                  class_label, attribute_label). We reuse the OFFICIAL loader +
                  transforms so inference matches training exactly.
    `model2_ckpt`: C->Y checkpoint for Independent/Sequential; None for Joint.

    ADROIT NOTE: confirm `image_dir`/`-data_dir2` so the official dataset resolves
    img_path (it splits on 'CUB_200_2011'; FunnyBirds paths differ -- see
    data/README). Everything else is framework-verified.
    """
    import pickle
    import numpy as np
    import torch
    from curated.compat import add_cbm_to_path
    add_cbm_to_path()
    from CUB.dataset import load_data  # official loader + transforms

    model = torch.load(model_ckpt, map_location=device); model.eval().to(device)
    model2 = None
    if model2_ckpt is not None:
        model2 = torch.load(model2_ckpt, map_location=device); model2.eval().to(device)

    loader = load_data([data_pkl], use_attr=True, no_img=False, batch_size=batch_size,
                       image_dir=image_dir, n_class_attr=2, resampling=False)
    recs = pickle.loads(Path(data_pkl).read_bytes())
    stems = [Path(r["img_path"]).stem for r in recs]

    rows, ptr = [], 0
    with torch.no_grad():
        for data in loader:
            inputs, class_labels, attr_labels = data[0], data[1], data[2]
            inputs = inputs.to(device)
            outputs = model(inputs)
            attr_logits = torch.cat([o.unsqueeze(1) for o in outputs[1:]], dim=1)
            probs = torch.sigmoid(attr_logits)
            if model2 is not None:
                class_logits = model2(torch.sigmoid(attr_logits))
            else:
                class_logits = outputs[0]
            y_pred = class_logits.argmax(1).cpu().numpy()
            z = attr_logits.cpu().numpy(); p = probs.cpu().numpy()
            attr_np = torch.stack([a if torch.is_tensor(a) else torch.tensor(a)
                                   for a in attr_labels], 1).cpu().numpy() \
                if isinstance(attr_labels, (list, tuple)) else np.asarray(attr_labels)
            yt = np.asarray(class_labels)
            for b in range(inputs.shape[0]):
                rows += _emit(stems[ptr], yt[b], concept_names, z[b], p[b],
                              attr_np[b], yt[b], y_pred[b], part_of)
                ptr += 1
    return pd.DataFrame(rows, columns=EVAL_COLUMNS)


def build_eval_table_mcbm(model, loader, concept_names, image_stems,
                          part_of=None, device="cuda") -> pd.DataFrame:
    """Run a minimal_cbm MCBM model over a split -> EVAL_SCHEMA.

    Faithful to external/minimal_cbm/src/models/mcbm.py (verified):
      z = model.p_z_x(x);  c_logits, c_preds = model.q_c_z(z)   # sampling=False
    We dump the CLEAN z (the quantity the minimality MSE pulls to +/-3), and
    sigmoid(c_logits) as prob. Task head: model.q_y_z(z) if present else argmax of
    whatever class logits the model exposes.

    `model` : an already-instantiated, checkpoint-loaded MCBM (build it with the
              repo's own config/loader on adroit; passing the live object avoids
              guessing the checkpoint wrapper format).
    `loader`: yields (image, ...) batches aligned with `image_stems` order.
    """
    import numpy as np
    import torch
    model.eval().to(device)
    rows, ptr = [], 0
    with torch.no_grad():
        for batch in loader:
            x = (batch["image"] if isinstance(batch, dict) else batch[0]).to(device)
            gt = (batch.get("concepts") if isinstance(batch, dict) else batch[1])
            z = model.p_z_x(x)
            c_logits, _ = model.q_c_z(z)
            prob = torch.sigmoid(c_logits)
            if hasattr(model, "q_y_z"):
                y_logits = model.q_y_z(z)
            elif hasattr(model, "q_y_c"):
                y_logits = model.q_y_c(prob)
            else:
                y_logits = None
            y_pred = (y_logits.argmax(1).cpu().numpy()
                      if y_logits is not None else np.full(x.shape[0], -1))
            zc = z.cpu().numpy(); pc = prob.cpu().numpy()
            gtc = np.asarray(gt.cpu() if torch.is_tensor(gt) else gt)
            yt = np.asarray(batch["class_idx"].cpu()) if isinstance(batch, dict) \
                and "class_idx" in batch else y_pred
            for b in range(x.shape[0]):
                rows += _emit(image_stems[ptr], yt[b], concept_names, zc[b], pc[b],
                              gtc[b], yt[b], y_pred[b], part_of)
                ptr += 1
    return pd.DataFrame(rows, columns=EVAL_COLUMNS)


def attach_part(df: pd.DataFrame, part_of) -> pd.DataFrame:
    """(Re)compute the `part` column from concept_name via a mapping callable."""
    df = df.copy()
    df["part"] = df["concept_name"].map(part_of).fillna("")
    return df
