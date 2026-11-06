"""Two commands, everything else is in config.yaml.

    python run.py                              # full pipeline: labels -> GEE -> signatures -> models
    python run.py predict new_fields.shp       # classify new farm polygons

The pipeline runs both sampling approaches (polygon square, GPS-point square)
and compares them. Downloaded Earth Engine data is cached, so re-running only
redoes the fast local steps.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings

import geopandas as gpd
import joblib
import pandas as pd

from .config import load_config
from .dataset import build_matrix, gap_fill, to_wide
from .geometry import point_samples, polygon_samples
from .labels import load_fields

log = logging.getLogger("croptype")
MODES = ("polygon", "point")


def _out(cfg, mode=None):
    p = cfg.path("paths", "outputs")
    p = p / mode if mode else p
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepare(cfg):
    """Labels + sampling geometries for both approaches."""
    fields = load_fields(cfg)
    s = cfg.sampling
    samples = {
        "polygon": polygon_samples(fields, s["polygon_inner_buffer_m"], s["polygon_mode"],
                                   s["min_square_half_m"]),
        "point": point_samples(fields, s["point_square_half_m"], s["point_clip_to_polygon"]),
    }
    interim = cfg.path("paths", "interim")
    fields.to_file(interim / "labels.gpkg", driver="GPKG")
    for mode, gdf in samples.items():
        gdf.to_file(interim / f"samples_{mode}.gpkg", driver="GPKG")
        log.info("%-7s sampling: %d fields, median %.1f pixels each", mode, len(gdf),
                 gdf.approx_pixels.median())
    return fields.drop(columns="geometry"), samples


def extract(cfg, mode, samples) -> pd.DataFrame:
    from . import gee_extract
    out_csv = cfg.path("paths", "features") / f"signature_table_{mode}.csv"
    log.info("downloading %s signatures from Earth Engine ...", mode)
    return gee_extract.extract(samples, cfg, out_csv, level="field")


def signatures(cfg, mode, long, labels):
    from .signatures import class_signatures, plot_signatures, separability
    out = _out(cfg, mode)
    filled = gap_fill(long).merge(labels[["field_id", "label", "split"]], on="field_id")
    train = filled[filled["split"] == "train"]  # signatures from training fields only
    med, iqr = class_signatures(train)
    med.to_csv(out / "signatures.csv", index=False)
    iqr.to_csv(out / "signatures_iqr.csv", index=False)
    plot_signatures(train, out / "signatures.png")
    wide, _ = to_wide(train)
    wide = wide.merge(labels[["field_id", "label"]], on="field_id")
    jm = separability(wide, int(long["t"].max()) + 1)
    jm.to_csv(out / "separability.csv", index=False)


def run_all(cfg, args):
    from . import gee_extract
    from .ensemble import label_review, train_ensemble
    from .train import train
    labels, samples = prepare(cfg)
    gee_extract.initialize(cfg.gee["project"])

    # 1. Compare the two sampling approaches with simple field-level models.
    compare, tables = {}, {}
    for mode in MODES:
        long = extract(cfg, mode, samples[mode])
        signatures(cfg, mode, long, labels)
        tables[mode] = build_matrix(long, labels)
        compare[mode] = train(*tables[mode], cfg, _out(cfg, mode), tag=mode,
                              run_spatial_cv=cfg.model.get("spatial_cv", False))

    # 2. Final model: field + pixel ensemble on the polygon sampling.
    log.info("downloading pixel time series from Earth Engine ...")
    pix = gee_extract.extract_pixels(samples["polygon"], cfg,
                                     cfg.path("paths", "features") / "pixels_polygon.npz")
    final = train_ensemble(*tables["polygon"], pix, labels, cfg, _out(cfg, "final"))

    # 3. Fields whose satellite signature contradicts the survey label.
    meta = labels[["field_id", "crop_stage", "surveyor", "survey_time", "image_url"]]
    n_flag = label_review(*tables["polygon"], cfg, meta, _out(cfg) / "label_review.csv")

    text = _summary_text(compare, final, n_flag, len(labels))
    (_out(cfg) / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n\nAll outputs: {_out(cfg)}")


def _summary_text(compare, final, n_flag, n_fields) -> str:
    rows = [{"model": f"{m} / {k}", "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
            for m, res in compare.items() for k, r in res.items() if "accuracy" in r]
    rows += [{"model": f"final / {k}", "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
             for k, r in final.items()]
    tbl = pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_string(index=False)
    e = final["ensemble"]
    conf = "\n".join(f"  confidence >= {q:<5} {v['share_of_fields']:6.1%} of fields   accuracy {v['accuracy']:.3f}"
                     for q, v in e["by_confidence"].items())
    return (f"FINAL MODEL (field + pixel ensemble), {e['n_fields']} test fields\n"
            f"  accuracy {e['accuracy']:.3f}   macro F1 {e['macro_f1']:.3f}   kappa {e['kappa']:.3f}\n\n"
            f"Accuracy when only confident predictions are kept:\n{conf}\n\n"
            f"Label review: {n_flag} of {n_fields} fields ({n_flag / n_fields:.0%}) have a satellite\n"
            f"signature that strongly contradicts the surveyed crop -> outputs/label_review.csv\n\n"
            f"All models on the same test fields:\n{tbl}")


def predict(cfg, args):
    """Classify new polygons with the final ensemble (new fields come with
    boundaries, not GPS points, so the polygon sampling is used)."""
    from . import gee_extract
    from .ensemble import predict_ensemble
    bundle = joblib.load(_out(cfg, "final") / "model.joblib")
    fields = gpd.read_file(args.shapefile).to_crs(4326)
    if "field_id" not in fields:
        fields["field_id"] = [f"N{i:06d}" for i in range(len(fields))]
    fields["field_id"] = fields["field_id"].astype(str)
    s = cfg.sampling
    samples = polygon_samples(fields, s["polygon_inner_buffer_m"], s["polygon_mode"],
                              s["min_square_half_m"])
    out_dir = _out(cfg, "predictions")
    gee_extract.initialize(cfg.gee["project"])
    long = gee_extract.extract(samples, cfg, out_dir / "signature_table.csv")
    pix = gee_extract.extract_pixels(samples, cfg, out_dir / "pixels.npz")
    wide, _ = to_wide(gap_fill(long, max_missing_frac=1.0))
    res = predict_ensemble(bundle, wide, pix).reset_index()
    res["field_id"] = res["field_id"].astype(str)
    fields.merge(res[["field_id", "pred_crop", "confidence"]], on="field_id", how="left") \
          .to_file(out_dir / "predictions.gpkg", driver="GPKG")
    res.to_csv(out_dir / "predictions.csv", index=False)
    print(res["pred_crop"].value_counts().to_string())
    print(f"\nPredictions: {out_dir / 'predictions.gpkg'}  (use the confidence column to filter)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="default: config.yaml")
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("predict", help="classify new farm polygons")
    sp.add_argument("shapefile")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("googleapiclient").setLevel(logging.ERROR)
    logging.getLogger("pyogrio").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    (predict if args.cmd == "predict" else run_all)(cfg, args)


if __name__ == "__main__":
    main()
