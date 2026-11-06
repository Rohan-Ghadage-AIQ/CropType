"""Pixel-level crop models.

Every 10 m pixel inside a field is its own training sample (about 30 per field
instead of one field average), so the model sees many more examples of each
crop's seasonal curve. At prediction time the pixels of a field vote: their
class probabilities are averaged into one answer per field.

Two pixel models are trained and later ensembled with the field-level model:
  * TempCNN  - 1-D convolutional network over the time axis, the standard
               deep-learning baseline for satellite crop time series
               (Pelletier et al. 2019). Learns shapes such as "green-up three
               weeks after the rains" directly from the curves.
  * XGBoost  - gradient-boosted trees on the flattened pixel time series.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

PIXEL_INDICES = ["NDVI", "EVI", "NDMI", "NDRE", "SAVI", "GNDVI", "VH_VV"]


# ---------------------------------------------------------------- features
def interpolate_time(a: np.ndarray) -> np.ndarray:
    """Vectorised linear interpolation of NaNs along the last axis (time);
    leading/trailing gaps take the nearest valid value."""
    shape = a.shape
    a = a.reshape(-1, shape[-1])
    T = a.shape[1]
    idx = np.arange(T)
    ok = ~np.isnan(a)
    prev = np.maximum.accumulate(np.where(ok, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(ok, idx, T)[:, ::-1], axis=1)[:, ::-1]
    vp = np.take_along_axis(a, np.clip(prev, 0, T - 1), 1)
    vn = np.take_along_axis(a, np.clip(nxt, 0, T - 1), 1)
    hp, hn = prev >= 0, nxt < T
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (idx - prev) / np.maximum(nxt - prev, 1)
    filled = np.where(hp & hn, vp + w * (vn - vp), np.where(hp, vp, np.where(hn, vn, np.nan)))
    out = np.where(ok, a, filled)  # observed values are never touched
    return out.reshape(shape).astype(np.float32)


def pixel_features(X: np.ndarray, bands: list[str]) -> tuple[np.ndarray, list[str]]:
    """(pixels, dates, bands) reflectance/dB -> (pixels, features, dates) with
    the vegetation indices added, gaps filled over time."""
    b = {name: X[:, :, i] for i, name in enumerate(bands)}
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = lambda x, y: (x - y) / (x + y)  # noqa: E731
        idx = {
            "NDVI": nd(b["B8"], b["B4"]),
            "EVI": 2.5 * (b["B8"] - b["B4"]) / (b["B8"] + 6 * b["B4"] - 7.5 * b["B2"] + 1),
            "NDMI": nd(b["B8"], b["B11"]),
            "NDRE": nd(b["B8"], b["B5"]),
            "SAVI": 1.5 * (b["B8"] - b["B4"]) / (b["B8"] + b["B4"] + 0.5),
            "GNDVI": nd(b["B8"], b["B3"]),
            "VH_VV": b["VH"] - b["VV"],
        }
    names = list(bands) + PIXEL_INDICES
    F = np.stack([b[n] for n in bands] + [idx[n] for n in PIXEL_INDICES], axis=1)  # (P, F, T)
    F[~np.isfinite(F)] = np.nan
    np.clip(F, -50, 50, out=F)  # guard against division blow-ups
    return interpolate_time(F), names


def cap_pixels(field_id: np.ndarray, max_per_field: int, seed: int = 0) -> np.ndarray:
    """Indices keeping at most `max_per_field` random pixels per field, so a
    few big fields cannot dominate training."""
    rng = np.random.default_rng(seed)
    keep = []
    for f in np.unique(field_id):
        ii = np.flatnonzero(field_id == f)
        keep.append(ii if len(ii) <= max_per_field else rng.choice(ii, max_per_field, replace=False))
    return np.sort(np.concatenate(keep))


def sample_weights(y: np.ndarray, field_id: np.ndarray) -> np.ndarray:
    """Each field weighs the same however many pixels it has, and each class
    weighs the same however many fields it has."""
    _, inv, cnt = np.unique(field_id, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    for c in np.unique(y):
        m = y == c
        w[m] /= w[m].sum()
    return (w / w.mean()).astype(np.float32)


def field_vote(proba: np.ndarray, field_id: np.ndarray):
    """Average pixel probabilities per field -> (field_ids, field_proba)."""
    fids, inv = np.unique(field_id, return_inverse=True)
    out = np.zeros((len(fids), proba.shape[1]))
    np.add.at(out, inv, proba)
    out /= np.bincount(inv)[:, None]
    return fids, out


# ---------------------------------------------------------------- TempCNN
class TempCNNClassifier:
    """sklearn-style wrapper around a small 1-D CNN over time."""

    def __init__(self, n_epochs=30, batch_size=512, lr=1e-3, hidden=64, dropout=0.3,
                 weight_decay=1e-4, patience=6, random_state=42, threads=None):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _net(self, n_feat, n_time, n_cls):
        import torch.nn as nn
        h, d = self.hidden, self.dropout
        block = lambda i, o: [nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(d)]  # noqa: E731
        return nn.Sequential(*block(n_feat, h), *block(h, h), *block(h, h), nn.Flatten(),
                             nn.Linear(h * n_time, 256), nn.BatchNorm1d(256), nn.ReLU(),
                             nn.Dropout(d), nn.Linear(256, n_cls))

    def _prep(self, X):
        Z = (X - self.mu_) / self.sd_
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        import torch
        torch.manual_seed(self.random_state)
        if self.threads:
            torch.set_num_threads(self.threads)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.mu_ = np.nanmean(X, axis=(0, 2), keepdims=True)
        self.sd_ = np.nanstd(X, axis=(0, 2), keepdims=True) + 1e-6
        Xt = torch.from_numpy(self._prep(X))
        yt = torch.from_numpy(yi)
        wt = torch.from_numpy(sample_weight if sample_weight is not None
                              else np.ones(len(y), np.float32))
        self.n_time_ = X.shape[2]
        self.net_ = self._net(X.shape[1], X.shape[2], len(self.classes_))
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr * 3, total_steps=self.n_epochs * ((len(y) + self.batch_size - 1) // self.batch_size))
        lossf = torch.nn.CrossEntropyLoss(reduction="none")
        best, best_state, bad = -1.0, None, 0
        for ep in range(self.n_epochs):
            self.net_.train()
            perm = torch.randperm(len(y))
            for i in range(0, len(y), self.batch_size):
                b = perm[i:i + self.batch_size]
                xb = Xt[b]
                xb = xb + 0.03 * torch.randn_like(xb)  # light noise augmentation
                loss = (lossf(self.net_(xb), yt[b]) * wt[b]).sum() / wt[b].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if X_val is not None:
                acc = (self.predict(X_val) == y_val).mean()
                if acc > best:
                    best, bad = acc, 0
                    best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                else:
                    bad += 1
                log.info("  tempcnn epoch %d  val pixel acc %.4f", ep + 1, acc)
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        out = []
        with torch.no_grad():
            Z = torch.from_numpy(self._prep(X))
            for i in range(0, len(Z), 4096):
                out.append(torch.softmax(self.net_(Z[i:i + 4096]), 1).numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def __getstate__(self):  # make joblib-able without pickling torch internals oddly
        s = self.__dict__.copy()
        if "net_" in s:
            s["net_state_"] = {k: v.cpu() for k, v in s.pop("net_").state_dict().items()}
            s["net_shape_"] = (self.mu_.shape[1], self.n_time_, len(self.classes_))
        return s

    def __setstate__(self, s):
        state = s.pop("net_state_", None)
        shape = s.pop("net_shape_", None)
        self.__dict__.update(s)
        if state is not None:
            self.net_ = self._net(*shape)
            self.net_.load_state_dict(state)
