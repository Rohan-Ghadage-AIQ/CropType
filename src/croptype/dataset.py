"""Turn the long signature table into a model-ready matrix.

Long table (one row per sample x period, columns = NDVI, EVI, ..., B12) is what
you inspect and plot. For the model each sample becomes one row whose columns
are every feature at every date (NDVI_t00 ... B12_t13) plus a few phenology
metrics derived from the NDVI curve. That full temporal vector is the crop's
"unique signature": wheat, gram and onion look alike on any single date but
differ in when they green up, how high they peak, and when they senesce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES

ID_COLS = ["field_id", "sample_mode"]


def gap_fill(long: pd.DataFrame, keys=("field_id", "sample_mode"),
             max_missing_frac: float = 0.5) -> pd.DataFrame:
    """Linear interpolation over time per sample (clouds leave holes in S2;
    S1 can miss a period when only one satellite is flying), then edge fill.
    Samples missing more than `max_missing_frac` of S2 dates are dropped."""
    keys = list(keys) + (["pixel_id"] if "pixel_id" in long else [])
    long = long.sort_values(keys + ["t"]).copy()
    feats = [f for f in FEATURES if f in long]
    long[feats] = long[feats].apply(pd.to_numeric, errors="coerce")

    miss = long.groupby(keys)["NDVI"].apply(lambda s: s.isna().mean())
    good = miss[miss <= max_missing_frac].index
    long = long.set_index(keys).loc[good].reset_index()

    long[feats] = (long.groupby(keys, group_keys=False)[feats]
                   .apply(lambda g: g.interpolate(limit_direction="both")))
    return long


def phenology(wide: pd.DataFrame, n_t: int) -> pd.DataFrame:
    """Simple, robust phenology metrics from the NDVI curve."""
    ndvi = wide[[f"NDVI_t{i:02d}" for i in range(n_t)]].to_numpy()
    vh = wide[[f"VH_t{i:02d}" for i in range(n_t)]].to_numpy()
    d = np.diff(ndvi, axis=1)
    out = pd.DataFrame(index=wide.index)
    out["ph_ndvi_max"] = ndvi.max(1)
    out["ph_ndvi_min"] = ndvi.min(1)
    out["ph_ndvi_amp"] = out["ph_ndvi_max"] - out["ph_ndvi_min"]
    out["ph_t_peak"] = ndvi.argmax(1)
    out["ph_ndvi_auc"] = ndvi.sum(1)
    out["ph_greenup_rate"] = d.max(1)
    out["ph_senescence_rate"] = d.min(1)
    out["ph_t_greenup"] = d.argmax(1)
    out["ph_t_senescence"] = d.argmin(1)
    # Length of season: periods above half the amplitude.
    thr = out["ph_ndvi_min"].to_numpy()[:, None] + 0.5 * out["ph_ndvi_amp"].to_numpy()[:, None]
    out["ph_season_len"] = (ndvi > thr).sum(1)
    out["ph_vh_mean"] = vh.mean(1)
    out["ph_vh_std"] = vh.std(1)
    return out


def to_wide(long: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keys = ID_COLS + (["pixel_id"] if "pixel_id" in long else [])
    feats = [f for f in FEATURES if f in long]
    n_t = int(long["t"].max()) + 1
    wide = long.pivot_table(index=keys, columns="t", values=feats)
    wide.columns = [f"{f}_t{int(t):02d}" for f, t in wide.columns]
    wide = wide.reset_index()
    wide = pd.concat([wide, phenology(wide, n_t)], axis=1)
    feature_cols = [c for c in wide.columns if c not in keys]
    return wide, feature_cols


def build_matrix(long: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Gap-fill, pivot, and attach labels/split. Returns (table, feature_cols)."""
    wide, feature_cols = to_wide(gap_fill(long))
    meta = labels[["field_id", "label", "crop", "split", "district", "village"]]
    wide = wide.merge(meta, on="field_id", how="inner")
    return wide, feature_cols
