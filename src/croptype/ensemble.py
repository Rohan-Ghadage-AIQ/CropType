"""Final crop model: an ensemble of three views of the same fields.

  field  - XGBoost on the field's median time series + phenology metrics
  cnn    - TempCNN on every pixel's time series, pixels vote per field
  pxgb   - XGBoost on every pixel's time series, pixels vote per field

Their class probabilities are averaged. Each prediction carries a confidence
(the winning probability), which is what makes the output usable: the
model is far more accurate on fields it is sure about (see summary.txt).

It also writes label_review.csv: fields whose satellite signature strongly
contradicts the surveyed crop, with the survey photo link, for re-checking.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .pixel_model import TempCNNClassifier, cap_pixels, field_vote, pixel_features, sample_weights

log = logging.getLogger(__name__)
CONF_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.85)


def _field_xgb(cfg):
    p = dict(cfg["model"]["xgb"])
    return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=cfg["model"]["random_state"], **p)


def _pixel_xgb(cfg):
    return XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.08, subsample=0.7,
                         colsample_bytree=0.3, max_bin=128, tree_method="hist", n_jobs=-1,
                         random_state=cfg["model"]["random_state"])


def _scores(y, proba, classes) -> dict:
    pred = classes[proba.argmax(1)]
    conf = proba.max(1)
    out = {"n_fields": int(len(y)), "accuracy": round(accuracy_score(y, pred), 4),
           "macro_f1": round(f1_score(y, pred, average="macro"), 4),
           "kappa": round(cohen_kappa_score(y, pred), 4), "by_confidence": {}}
    for q in CONF_LEVELS:
        k = conf >= q
        if k.any():
            out["by_confidence"][str(q)] = {"share_of_fields": round(float(k.mean()), 3),
                                            "accuracy": round(accuracy_score(y[k], pred[k]), 4)}
    return out


def train_ensemble(field_table: pd.DataFrame, field_cols: list[str], pix: dict,
                   labels: pd.DataFrame, cfg, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rs = cfg["model"]["random_state"]
    pcfg = cfg.get("pixel", {})
    classes = np.array(sorted(labels["label"].unique()))
    lab = labels.set_index("field_id")

    # ---------------- field member
    ft = field_table.set_index("field_id")
    is_tr = ft["split"].eq("train").to_numpy()
    yf = np.searchsorted(classes, ft["label"])
    field_model = _field_xgb(cfg)
    field_model.fit(ft.loc[is_tr, field_cols], yf[is_tr],
                    sample_weight=compute_sample_weight("balanced", yf[is_tr]))
    P = {"field": pd.DataFrame(field_model.predict_proba(ft.loc[~is_tr, field_cols]),
                               index=ft.index[~is_tr])}
    log.info("[ensemble] field model trained")

    # ---------------- pixel members
    keep = np.isin(pix["field_id"], lab.index)
    F, names = pixel_features(pix["X"][keep], list(pix["bands"]))
    fid = pix["field_id"][keep]
    y = lab.loc[fid, "label"].to_numpy()
    split = lab.loc[fid, "split"].to_numpy()
    train_f = np.unique(fid[split == "train"])
    val_f = np.random.default_rng(rs).choice(train_f, int(0.1 * len(train_f)), replace=False)
    is_val = np.isin(fid, val_f)
    tr_idx = np.flatnonzero((split == "train") & ~is_val)
    tr_idx = tr_idx[cap_pixels(fid[tr_idx], pcfg.get("max_pixels_per_field", 64), rs)]
    te = split == "test"
    sw = sample_weights(y[tr_idx], fid[tr_idx])
    log.info("[ensemble] pixel training set: %d pixels from %d fields", len(tr_idx),
             len(np.unique(fid[tr_idx])))

    cnn = TempCNNClassifier(n_epochs=pcfg.get("tempcnn_epochs", 40), random_state=rs)
    cnn.fit(F[tr_idx], y[tr_idx], sample_weight=sw, X_val=F[is_val], y_val=y[is_val])
    f, p = field_vote(cnn.predict_proba(F[te]), fid[te])
    P["cnn"] = pd.DataFrame(p, index=f)
    log.info("[ensemble] TempCNN trained")

    pxgb = _pixel_xgb(cfg)
    Xf = F.reshape(len(F), -1)
    pxgb.fit(Xf[tr_idx], np.searchsorted(classes, y[tr_idx]), sample_weight=sw)
    f, p = field_vote(pxgb.predict_proba(Xf[te]), fid[te])
    P["pxgb"] = pd.DataFrame(p, index=f)
    log.info("[ensemble] pixel XGBoost trained")

    # ---------------- combine (fields too small for any pixel use the field model alone)
    test_ids = P["field"].index
    ens = combine({k: v for k, v in P.items()}, test_ids)
    yt = lab.loc[test_ids, "label"].to_numpy()
    metrics = {k: _scores(lab.loc[v.index, "label"].to_numpy(), v.to_numpy(), classes)
               for k, v in P.items()}
    metrics["ensemble"] = _scores(yt, ens, classes)

    pred = pd.DataFrame(ens, index=test_ids, columns=classes)
    pred["label"] = yt
    pred["predicted"] = classes[ens.argmax(1)]
    pred["confidence"] = ens.max(1).round(3)
    pred.to_csv(out_dir / "test_predictions.csv")

    joblib.dump({"field_model": field_model, "field_cols": field_cols, "cnn": cnn, "pxgb": pxgb,
                 "classes": classes, "pixel_bands": list(pix["bands"]), "metrics": metrics},
                out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def combine(P: dict, ids) -> np.ndarray:
    """Average member probabilities per field; a member that has no row for
    a field (no pixel fell inside it) simply doesn't vote."""
    stack = [v.reindex(ids).to_numpy() for v in P.values()]
    s = np.nansum(stack, axis=0)
    n = np.sum([~np.isnan(a[:, 0]) for a in stack], axis=0)[:, None]
    return s / np.maximum(n, 1)


def predict_ensemble(bundle: dict, field_wide: pd.DataFrame, pix: dict) -> pd.DataFrame:
    classes = bundle["classes"]
    fw = field_wide.set_index("field_id")
    P = {"field": pd.DataFrame(bundle["field_model"].predict_proba(
        fw.reindex(columns=bundle["field_cols"])), index=fw.index)}
    if len(pix["X"]):
        F, _ = pixel_features(pix["X"], bundle["pixel_bands"])
        f, p = field_vote(bundle["cnn"].predict_proba(F), pix["field_id"])
        P["cnn"] = pd.DataFrame(p, index=f)
        f, p = field_vote(bundle["pxgb"].predict_proba(F.reshape(len(F), -1)), pix["field_id"])
        P["pxgb"] = pd.DataFrame(p, index=f)
    ens = combine(P, fw.index)
    out = pd.DataFrame(ens, index=fw.index, columns=classes)
    out["pred_crop"] = classes[ens.argmax(1)]
    out["confidence"] = ens.max(1).round(3)
    return out


def label_review(field_table: pd.DataFrame, field_cols: list[str], cfg, raw_meta: pd.DataFrame,
                 out_csv: Path, p_label_max: float = 0.10, p_other_min: float = 0.60) -> int:
    """Flag fields whose signature strongly contradicts their surveyed crop.

    Uses 5-fold out-of-fold predictions over ALL fields (each field is scored
    by a model that never saw it). Flagged = the surveyed crop gets < 10%
    probability while another crop gets > 60%.
    """
    ft = field_table.reset_index(drop=True)
    classes = np.array(sorted(ft["label"].unique()))
    y = np.searchsorted(classes, ft["label"])
    oof = np.zeros((len(ft), len(classes)))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=cfg["model"]["random_state"]).split(ft, y):
        m = _field_xgb(cfg)
        m.fit(ft.loc[a, field_cols], y[a], sample_weight=compute_sample_weight("balanced", y[a]))
        oof[b] = m.predict_proba(ft.loc[b, field_cols])
    p_label = oof[np.arange(len(y)), y]
    df = pd.DataFrame({"field_id": ft["field_id"], "surveyed_crop": ft["label"],
                       "satellite_says": classes[oof.argmax(1)],
                       "p_surveyed_crop": p_label.round(3), "p_satellite_crop": oof.max(1).round(3),
                       "split": ft["split"], "village": ft["village"], "district": ft["district"]})
    df = df.merge(raw_meta, on="field_id", how="left")
    flagged = df[(df.p_surveyed_crop < p_label_max) & (df.p_satellite_crop > p_other_min)]
    flagged.sort_values("p_satellite_crop", ascending=False).to_csv(out_csv, index=False)
    return len(flagged)
