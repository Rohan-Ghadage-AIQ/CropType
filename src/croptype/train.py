"""Train and evaluate crop classifiers on the signature matrix.

Three models are compared on the same field-level hold-out:
  * signature : nearest crop signature (interpretable baseline)
  * rf        : Random Forest
  * xgb       : XGBoost (usually best on tabular temporal features)

Evaluation is always per FIELD. With pixel-level data, pixels inherit their
field's split (no pixel of a test field is ever seen in training) and pixel
probabilities are averaged to one prediction per field.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,  # noqa: E402
                             classification_report, cohen_kappa_score, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from .signatures import SignatureClassifier  # noqa: E402

log = logging.getLogger(__name__)


def make_models(cfg, n_classes: int) -> dict:
    m = cfg["model"]
    rs = m["random_state"]
    return {
        "signature": SignatureClassifier(),
        "rf": RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample",
                                     random_state=rs, **m["rf"]),
        "xgb": XGBClassifier(objective="multi:softprob", num_class=n_classes,
                             tree_method="hist", n_jobs=-1, random_state=rs,
                             eval_metric="mlogloss", **m["xgb"]),
    }


def split_fields(table: pd.DataFrame, cfg) -> pd.Series:
    """Boolean mask: True = train row. Split is decided per field."""
    m = cfg["model"]
    if m.get("use_given_split", True) and "split" in table:
        return table["split"].eq("train")
    fields = table.drop_duplicates("field_id")[["field_id", "label"]]
    tr, _ = train_test_split(fields, test_size=m["test_size"], stratify=fields["label"],
                             random_state=m["random_state"])
    return table["field_id"].isin(tr["field_id"])


def _field_level(table, proba, classes):
    p = pd.DataFrame(proba, columns=classes)
    p["field_id"] = table["field_id"].to_numpy()
    p = p.groupby("field_id").mean()
    truth = table.groupby("field_id")["label"].first().loc[p.index]
    return truth, p


def evaluate(truth, proba_df) -> dict:
    pred = proba_df.idxmax(axis=1)
    return {
        "n_fields": int(len(truth)),
        "accuracy": round(accuracy_score(truth, pred), 4),
        "macro_f1": round(f1_score(truth, pred, average="macro"), 4),
        "weighted_f1": round(f1_score(truth, pred, average="weighted"), 4),
        "kappa": round(cohen_kappa_score(truth, pred), 4),
    }, pred


def spatial_cv(table, feature_cols, cfg, model_name="xgb", n_splits=5) -> dict:
    """Cross-validation grouped by village: an honest estimate of how the model
    does in villages it has never seen (neighbouring fields share soil, sowing
    date and weather, so a random split is optimistic)."""
    le = LabelEncoder().fit(table["label"])
    y = le.transform(table["label"])
    groups = table["village"].fillna(table["field_id"])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=cfg["model"]["random_state"])
    scores = []
    for tr, te in cv.split(table, y, groups):
        model = make_models(cfg, len(le.classes_))[model_name]
        Xtr = table.iloc[tr][feature_cols]
        model.fit(Xtr, y[tr], **_fit_kw(model_name, y[tr]))
        truth, p = _field_level(table.iloc[te], model.predict_proba(table.iloc[te][feature_cols]),
                                le.classes_)
        scores.append(evaluate(truth, p)[0]["macro_f1"])
    return {"model": model_name, "macro_f1_mean": round(float(np.mean(scores)), 4),
            "macro_f1_std": round(float(np.std(scores)), 4), "folds": scores}


def _fit_kw(name, y):
    if name == "signature":
        return {}
    if name == "xgb":
        return {"sample_weight": compute_sample_weight("balanced", y)}
    return {}


def train(table: pd.DataFrame, feature_cols: list[str], cfg, out_dir: Path,
          tag: str = "polygon", run_spatial_cv: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    is_train = split_fields(table, cfg)
    tr, te = table[is_train], table[~is_train]
    le = LabelEncoder().fit(table["label"])
    ytr = le.transform(tr["label"])
    log.info("[%s] train rows=%d (fields=%d)  test rows=%d (fields=%d)", tag, len(tr),
             tr.field_id.nunique(), len(te), te.field_id.nunique())

    results, fitted = {}, {}
    for name, model in make_models(cfg, len(le.classes_)).items():
        model.fit(tr[feature_cols], ytr, **_fit_kw(name, ytr))
        truth, proba = _field_level(te, model.predict_proba(te[feature_cols]), le.classes_)
        metrics, pred = evaluate(truth, proba)
        results[name] = metrics
        fitted[name] = (model, truth, pred)
        log.info("[%s] %-9s %s", tag, name, metrics)

    best = max(results, key=lambda k: results[k]["macro_f1"])
    model, truth, pred = fitted[best]

    report = classification_report(truth, pred, digits=3, zero_division=0)
    (out_dir / "report.txt").write_text(
        f"best model: {best}\n\n{json.dumps(results, indent=2)}\n\n{report}", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(truth, pred, ax=ax, xticks_rotation=45,
                                            normalize="true", values_format=".2f",
                                            colorbar=False)
    ax.set_title(f"{tag} / {best}: row-normalised confusion (test fields)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion.png", dpi=130)
    plt.close(fig)

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        imp.to_csv(out_dir / "importance.csv", header=["importance"])
        base = imp.groupby(imp.index.str.replace(r"_t\d+$", "", regex=True)).sum()
        base.sort_values(ascending=False).to_csv(out_dir / "importance_by_feature.csv",
                                                 header=["importance"])

    if run_spatial_cv:
        cv = spatial_cv(tr, feature_cols, cfg, best)
        results["spatial_cv_on_train"] = cv
        log.info("[%s] village-grouped CV: %s", tag, cv)

    joblib.dump({"model": model, "model_name": best, "feature_cols": feature_cols,
                 "classes": list(le.classes_), "tag": tag, "metrics": results},
                out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
