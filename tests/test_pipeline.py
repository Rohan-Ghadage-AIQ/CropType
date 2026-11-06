"""Offline end-to-end test: real labels + synthetic Sentinel time series.

Earth Engine is not called. A fake signature table with crop-specific NDVI /
VH curves (plus clouds = NaNs) stands in for the GEE output, so this checks
geometry, gap-filling, pivoting, signatures, separability and training wiring.
Run:  python -m pytest -q tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from croptype import cli  # noqa: E402
from croptype.features import FEATURES, S2_BANDS  # noqa: E402
from croptype.gee_extract import periods  # noqa: E402

SHP = ROOT / "data/raw/farm_boundaries_FINAL_Squares_27Oct_updated.shp"

# peak period, peak NDVI, season length (periods) - rough rabi phenology
PHENO = {"Wheat": (7, 0.80, 7), "Gram": (5, 0.60, 5), "Onion": (8, 0.55, 6),
         "Sorghum": (5, 0.70, 6), "Maize": (6, 0.82, 7), "Cotton": (1, 0.55, 5),
         "Sugarcane": (6, 0.75, 20)}


def fake_long(labels: pd.DataFrame, per, rng) -> pd.DataFrame:
    t = np.arange(len(per))
    rows = []
    for fid, lab in zip(labels.field_id, labels.label):
        pk, mx, ln = PHENO.get(lab, (6, 0.5, 6))
        pk += rng.normal(0, 0.8)
        ndvi = 0.18 + (mx - 0.18) * np.exp(-((t - pk) / (ln / 2.3)) ** 2) + rng.normal(0, 0.04, len(t))
        df = pd.DataFrame({"field_id": fid, "sample_mode": "polygon", "t": t,
                           "date": [a.date() for a, _ in per]})
        df["NDVI"] = ndvi
        df["EVI"] = ndvi * 0.9
        df["NDMI"] = ndvi * 0.6 - 0.1
        df["NDRE"] = ndvi * 0.7
        df["SAVI"] = ndvi * 0.8
        df["GNDVI"] = ndvi * 0.85
        df["VH"] = -22 + 8 * ndvi + rng.normal(0, 0.7, len(t))
        df["VV"] = -14 + 3 * ndvi + rng.normal(0, 0.7, len(t))
        df["VH_VV"] = df.VH - df.VV
        df["RVI"] = 0.3 + ndvi * 0.5
        for i, b in enumerate(S2_BANDS):
            df[b] = 0.05 + 0.02 * i + 0.1 * ndvi * (b in ("B8", "B8A", "B7", "B6"))
        cloud = rng.random(len(t)) < 0.25
        df.loc[cloud, [f for f in FEATURES if f not in ("VV", "VH", "VH_VV", "RVI")]] = np.nan
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def fake_pixels(long, per):
    """3 pixels per field: reflectances that reproduce the field's NDVI curve."""
    from croptype.gee_extract import PIXEL_FEATURES
    piv = long.pivot(index="field_id", columns="t", values="NDVI").interpolate(axis=1, limit_direction="both")
    rng = np.random.default_rng(1)
    nd = np.repeat(piv.to_numpy(), 3, axis=0) + rng.normal(0, 0.03, (len(piv) * 3, len(per)))
    X = np.full((len(nd), len(per), len(PIXEL_FEATURES)), 0.1, np.float32)
    b = {n: i for i, n in enumerate(PIXEL_FEATURES)}
    X[:, :, b["B4"]] = 0.08
    X[:, :, b["B8"]] = 0.08 * (1 + nd) / (1 - nd)
    X[:, :, b["VV"]], X[:, :, b["VH"]] = -12, -20 + 6 * nd
    X[:, 2, :] = np.nan  # a cloudy date
    return {"X": X, "field_id": np.repeat(piv.index.to_numpy().astype("U32"), 3),
            "bands": np.array(PIXEL_FEATURES)}


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["data"]["shapefile"] = str(SHP)
    cfg["paths"] = {k: str(tmp / k) for k in ("interim", "features", "outputs")}
    cfg["model"]["xgb"]["n_estimators"] = 80
    cfg["model"]["rf"]["n_estimators"] = 80
    p = tmp / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.mark.skipif(not SHP.exists(), reason="shapefile not extracted")
def test_end_to_end(cfg_path):
    from croptype.dataset import build_matrix
    from croptype.train import train
    cfg = cli.load_config(cfg_path)
    labels, samples = cli.prepare(cfg)
    assert set(samples) == {"polygon", "point"}
    w = cfg["season_window"]
    per = periods(w["start"], w["end"], w["step_days"])
    long = fake_long(labels, per, np.random.default_rng(0))

    cli.signatures(cfg, "polygon", long, labels)
    table, cols = build_matrix(long, labels)
    m = train(table, cols, cfg, cli._out(cfg, "polygon"), tag="polygon", run_spatial_cv=True)

    # Final ensemble on synthetic pixels built from the same curves.
    from croptype.ensemble import label_review, predict_ensemble, train_ensemble
    from croptype.gee_extract import PIXEL_FEATURES
    pix = fake_pixels(long, per)
    cfg["pixel"]["tempcnn_epochs"] = 2
    fin = train_ensemble(table, cols, pix, labels, cfg, cli._out(cfg, "final"))
    assert fin["ensemble"]["accuracy"] > 0.5 and "by_confidence" in fin["ensemble"]
    import joblib
    bundle = joblib.load(cli._out(cfg, "final") / "model.joblib")  # round-trips TempCNN
    wide = table[["field_id"] + cols].head(30)
    res = predict_ensemble(bundle, wide, {k: v[:90] for k, v in pix.items() if k != "bands"} | {"bands": pix["bands"]})
    assert len(res) == 30 and res["confidence"].between(0, 1).all()
    n = label_review(table, cols, cfg, labels[["field_id", "image_url"]], cli._out(cfg) / "review.csv")
    assert n >= 0

    out = cli._out(cfg, "polygon")
    sig = pd.read_csv(out / "signatures.csv")
    assert {"label", "date", "NDVI", "VH", "B12"} <= set(sig.columns)
    assert (out / "model.joblib").exists() and (out / "signatures.png").exists()
    # Synthetic classes are separable: xgb must clearly beat chance.
    assert m["xgb"]["macro_f1"] > 0.5


def test_periods_drop_stub():
    per = periods("2022-10-01", "2023-04-16", 15)
    assert len(per) == 13 and all((b - a).days == 15 for a, b in per)


def test_interpolate_time_keeps_observed_values():
    from croptype.pixel_model import interpolate_time
    a = np.array([[np.nan, 1, np.nan, 3, np.nan], [5, np.nan, np.nan, 8, 9], [np.nan] * 5], np.float32)
    r = interpolate_time(a)
    assert np.allclose(r[0], [1, 1, 2, 3, 3]) and np.allclose(r[1], [5, 6, 7, 8, 9])
    assert np.isnan(r[2]).all()
