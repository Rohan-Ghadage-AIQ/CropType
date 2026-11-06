"""Crop signatures: per-crop temporal profiles, separability, and a
signature-matching classifier that uses them directly.

signatures.csv has exactly the rows x columns layout of the concept:
one row per (crop, date), columns NDVI, EVI, NDMI, NDRE, SAVI, GNDVI, VV, VH,
VH_VV, RVI, B1 ... B12 holding that crop's median value.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .features import FEATURES  # noqa: E402

PLOT_FEATURES = ["NDVI", "EVI", "NDMI", "NDRE", "GNDVI", "SAVI", "VV", "VH", "VH_VV",
                 "B4", "B8", "B11"]


def class_signatures(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median signature table and an IQR table, per (label, t, date)."""
    feats = [f for f in FEATURES if f in long]
    g = long.groupby(["label", "t", "date"])[feats]
    med = g.median().reset_index()
    iqr = (g.quantile(0.75) - g.quantile(0.25)).reset_index()
    n = long.groupby("label")["field_id"].nunique().rename("n_fields")
    med = med.merge(n, on="label")
    return med, iqr


def plot_signatures(long: pd.DataFrame, out_png: Path, features=PLOT_FEATURES):
    feats = [f for f in features if f in long]
    labels = long["label"].value_counts().index.tolist()
    cmap = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    dates = pd.to_datetime(long.groupby("t")["date"].first())
    for ax, f in zip(axes.flat, feats):
        q = long.groupby(["label", "t"])[f].quantile([0.25, 0.5, 0.75]).unstack()
        for i, lab in enumerate(labels):
            if lab not in q.index.get_level_values(0):
                continue
            s = q.loc[lab]
            x = dates.loc[s.index]
            ax.plot(x, s[0.5], color=cmap(i), lw=1.8, label=lab)
            ax.fill_between(x, s[0.25], s[0.75], color=cmap(i), alpha=0.12)
        ax.set_title(f)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Crop temporal signatures (median, shaded = IQR)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _jm(x1: np.ndarray, x2: np.ndarray) -> float:
    """Jeffries-Matusita distance between two Gaussian classes (0..2)."""
    m1, m2 = x1.mean(0), x2.mean(0)
    c1 = np.atleast_2d(np.cov(x1, rowvar=False)) + 1e-6 * np.eye(x1.shape[1])
    c2 = np.atleast_2d(np.cov(x2, rowvar=False)) + 1e-6 * np.eye(x1.shape[1])
    c = (c1 + c2) / 2
    dm = (m1 - m2)[:, None]
    _, ld = np.linalg.slogdet(c)
    _, ld1 = np.linalg.slogdet(c1)
    _, ld2 = np.linalg.slogdet(c2)
    b = 0.125 * (dm.T @ np.linalg.solve(c, dm)).item() +0.5 * (ld - 0.5 * (ld1 + ld2))
    return float(2 * (1 - np.exp(-b)))


def separability(wide: pd.DataFrame, n_t: int) -> pd.DataFrame:
    """Pairwise JM distance between crops using each feature's full time
    series. JM > 1.9 = well separable, < 1.0 = poorly separable."""
    labels = sorted(wide["label"].unique())
    rows = []
    for f in FEATURES:
        cols = [f"{f}_t{i:02d}" for i in range(n_t)]
        if not set(cols) <= set(wide):
            continue
        for a, b in combinations(labels, 2):
            xa = wide.loc[wide.label == a, cols].dropna().to_numpy()
            xb = wide.loc[wide.label == b, cols].dropna().to_numpy()
            if min(len(xa), len(xb)) <= n_t:
                continue
            rows.append({"feature": f, "crop_a": a, "crop_b": b, "jm": _jm(xa, xb)})
    return pd.DataFrame(rows)


class SignatureClassifier(BaseEstimator, ClassifierMixin):
    """Nearest-signature classifier: each crop's signature is the median of its
    standardized temporal feature vector; a field gets the crop whose signature
    it is closest to. Transparent baseline for the ML models."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.scaler_ = StandardScaler().fit(X)
        Z = np.nan_to_num(self.scaler_.transform(X))
        self.classes_ = np.unique(y)
        self.signatures_ = np.stack([np.median(Z[y == c], 0) for c in self.classes_])
        return self

    def _dist(self, X):
        Z = np.nan_to_num(self.scaler_.transform(np.asarray(X, float)))
        return np.sqrt(((Z[:, None, :] - self.signatures_[None]) ** 2).mean(-1))

    def predict_proba(self, X):
        d = -self._dist(X) / self.temperature
        e = np.exp(d - d.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._dist(X).argmin(1)]
