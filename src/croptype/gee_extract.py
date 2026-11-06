"""Google Earth Engine extraction of the Sentinel-1/2 time-series stack.

For every period (default: 15 days) of the season we build one composite:
  * Sentinel-2 L2A (S2_SR_HARMONIZED), clouds/shadows masked with Cloud Score+,
    per-pixel median of the 12 bands and 6 indices.
  * Sentinel-1 GRD IW, VV/VH averaged in linear power then converted to dB,
    plus VH/VV ratio and RVI.
All periods are stacked into one multi-band image (<feature>_tNN) and reduced
over each sampling geometry, giving one row per (field, period) in the output:

    field_id, sample_mode, t, date, NDVI, EVI, NDMI, NDRE, SAVI, GNDVI,
    VV, VH, VH_VV, RVI, B1 ... B12, n_valid

That long table is the "signature table". Chunks are cached on disk so an
interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .features import FEATURES, INDICES, S1_BANDS, S2_BANDS

log = logging.getLogger(__name__)
NODATA = -9999.0

try:
    import ee
except ImportError:  # keeps the rest of the package importable without GEE
    ee = None


def initialize(project: str | None):
    if ee is None:
        raise ImportError("pip install earthengine-api")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def periods(start: str, end: str, step_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    end_ts = pd.Timestamp(end)
    t0 = pd.date_range(start, end_ts, freq=f"{step_days}D", inclusive="left")
    out = []
    for a in t0:
        b = a + pd.Timedelta(days=step_days)
        if b > end_ts and (end_ts - a).days < step_days / 2:
            break  # a stub period of a few days would be mostly empty
        out.append((a, b if b < end_ts else end_ts))
    return out


# --------------------------------------------------------------------------- S2
def add_indices(img):
    b = {k: img.select(k) for k in ("B2", "B3", "B4", "B5", "B8", "B11")}
    nd = lambda x, y: img.normalizedDifference([x, y])  # noqa: E731
    evi = img.expression("2.5*(N-R)/(N+6*R-7.5*B+1)", {"N": b["B8"], "R": b["B4"], "B": b["B2"]})
    savi = img.expression("1.5*(N-R)/(N+R+0.5)", {"N": b["B8"], "R": b["B4"]})
    return img.addBands([
        nd("B8", "B4").rename("NDVI"), evi.rename("EVI"), nd("B8", "B11").rename("NDMI"),
        nd("B8", "B5").rename("NDRE"), savi.rename("SAVI"), nd("B8", "B3").rename("GNDVI"),
    ])


def s2_collection(region, start, end, cs_threshold: float):
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(start, end)
           .linkCollection(cs, ["cs_cdf"]))

    def prep(img):
        refl = img.select(S2_BANDS).divide(10000)
        # toFloat(): every image (and the empty placeholder merged in later)
        # must share one plain Float type, or GEE rejects the collection.
        return (add_indices(refl).toFloat().updateMask(img.select("cs_cdf").gte(cs_threshold))
                .copyProperties(img, ["system:time_start"]))

    return col.map(prep).select(S2_BANDS + INDICES)


# --------------------------------------------------------------------------- S1
def s1_collection(region, start, end, orbit_pass: str):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    if orbit_pass in ("ASCENDING", "DESCENDING"):
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    def to_linear(img):
        db = img.select(S1_BANDS)
        db = db.updateMask(db.gt(-30))  # drop border noise / no-data
        return (ee.Image(10).pow(db.divide(10)).toFloat()
                .copyProperties(img, ["system:time_start"]))

    return col.map(to_linear)


def _masked(bands):
    return ee.Image.constant([0] * len(bands)).rename(bands).updateMask(ee.Image(0)).toFloat()


def period_composite(region, t0, t1, cfg_gee):
    s2 = (s2_collection(region, t0, t1, cfg_gee["cloud_score_threshold"])
          .merge(ee.ImageCollection([_masked(S2_BANDS + INDICES)])).median())
    s1 = (s1_collection(region, t0, t1, cfg_gee["s1_orbit_pass"])
          .merge(ee.ImageCollection([_masked(S1_BANDS)])).mean())
    vv, vh = s1.select("VV"), s1.select("VH")
    s1_feats = ee.Image.cat([
        vv.log10().multiply(10).rename("VV"),
        vh.log10().multiply(10).rename("VH"),
        vh.divide(vv).log10().multiply(10).rename("VH_VV"),
        vh.multiply(4).divide(vv.add(vh)).rename("RVI"),
    ])
    return s2.addBands(s1_feats).select(FEATURES).toFloat()


def build_stack(region, cfg, features=FEATURES):
    w = cfg["season_window"]
    per = periods(w["start"], w["end"], w["step_days"])
    imgs = []
    for i, (a, b) in enumerate(per):
        comp = period_composite(region, a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), cfg["gee"])
        imgs.append(comp.select(features).rename([f"{f}_t{i:02d}" for f in features]))
    return ee.Image.cat(imgs), per


# --------------------------------------------------------------------- extract
def _to_fc(gdf: gpd.GeoDataFrame):
    feats = [ee.Feature(ee.Geometry(geom.__geo_interface__),
                        {"field_id": fid, "sample_mode": mode})
             for fid, mode, geom in zip(gdf["field_id"], gdf["sample_mode"], gdf.geometry)]
    return ee.FeatureCollection(feats)


def _reduce_chunk(gdf, cfg, level: str) -> pd.DataFrame:
    fc = _to_fc(gdf)
    region = fc.geometry().bounds()
    stack, _ = build_stack(region, cfg)
    scale = cfg["gee"]["scale"]
    if level == "field":
        reducer = ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True)
        out = stack.reduceRegions(collection=fc, reducer=reducer, scale=scale, tileScale=4)
    else:  # pixel: sampleRegions drops any pixel with a masked band, so fill first
        out = stack.unmask(NODATA).sampleRegions(
            collection=fc, properties=["field_id", "sample_mode"],
            scale=scale, tileScale=4, geometries=False)
    rows = [f["properties"] for f in out.getInfo()["features"]]
    return pd.DataFrame(rows).replace(NODATA, float("nan"))


def wide_to_long(df: pd.DataFrame, per, level: str) -> pd.DataFrame:
    """<feat>_tNN[_median|_count] columns -> one row per (sample, period)."""
    id_cols = ["field_id", "sample_mode"] + (["pixel_id"] if level == "pixel" else [])
    if level == "pixel":
        df = df.copy()
        df["pixel_id"] = df.groupby("field_id").cumcount()
    frames = []
    for i, (a, _) in enumerate(per):
        sfx = "_median" if level == "field" else ""
        cols = {f"{f}_t{i:02d}{sfx}": f for f in FEATURES}
        part = df[id_cols].copy()
        for src, dst in cols.items():
            # A band with no data anywhere in the chunk (e.g. no S1 pass that
            # period) is absent from the GEE response -> float NaN column.
            part[dst] = pd.to_numeric(df[src], errors="coerce") if src in df else np.nan
        if level == "field":
            part["n_valid_s2"] = df.get(f"NDVI_t{i:02d}_count", 0)
            part["n_valid_s1"] = df.get(f"VV_t{i:02d}_count", 0)
        part.insert(len(id_cols), "t", i)
        part.insert(len(id_cols) + 1, "date", (a + pd.Timedelta(days=(per[i][1] - a).days / 2)).date())
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


_TRANSIENT = ("timed out", "timeout", "too many", "quota", "rate limit", "memory limit",
              "503", "502", "429", "unavailable", "connection", "internal error")


def _retryable(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in _TRANSIENT)


def _cache_dir(out_path: Path, cfg, extra: dict) -> Path:
    """Chunk cache folder keyed on every setting that changes the values, so
    editing config.yaml never silently reuses stale chunks."""
    gee = {k: v for k, v in dict(cfg["gee"]).items() if k not in ("project", "chunk_size", "workers")}
    key = json.dumps({"w": dict(cfg["season_window"]), "g": gee, **extra}, sort_keys=True, default=str)
    tag = hashlib.md5(key.encode()).hexdigest()[:8]
    d = out_path.parent / f"_chunks_{out_path.stem}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _spatial_sort(samples):
    # Compact chunks = smaller S1/S2 searches per request.
    c = samples.geometry.representative_point()
    return samples.assign(_k=(c.y.round(1) * 1000 + c.x.round(1))).sort_values("_k")


def _run_chunks(chunks: list, paths: list[Path], fetch, workers: int, label: str,
                max_retries: int = 4):
    """Run fetch(sub, path) for every chunk not yet on disk, `workers` at a time,
    retrying transient Earth Engine errors. Raises if any chunk still fails."""
    todo = [i for i, p in enumerate(paths) if not p.exists()]
    if len(todo) < len(paths):
        log.info("%s: %d/%d chunks already downloaded, reusing them", label,
                 len(paths) - len(todo), len(paths))

    def job(i):
        for attempt in range(max_retries):
            try:
                fetch(chunks[i], paths[i])
                return True
            except Exception as e:
                if not _retryable(e):
                    raise  # a bug in the request: retrying won't help
                wait = 2 ** attempt * 10
                log.warning("chunk %d failed (%s); retry in %ds", i, str(e)[:160], wait)
                time.sleep(wait)
        return False

    from concurrent.futures import ThreadPoolExecutor, as_completed
    failed, done = [], 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(job, i): i for i in todo}
        for f in as_completed(futs):
            done += 1
            if not f.result():
                failed.append(futs[f])
            if done % 10 == 0 or done == len(todo):
                log.info("%s: %d/%d chunks downloaded", label, done, len(todo))
    if failed:
        # Never hand a partial table to training: it would silently drop fields.
        raise RuntimeError(f"{len(failed)} of {len(paths)} chunks failed (Earth Engine busy?). "
                           "Run the same command again; finished chunks are kept.")


def extract(samples: gpd.GeoDataFrame, cfg, out_csv: Path, level: str = "field",
            cache_dir: Path | None = None) -> pd.DataFrame:
    """Field-level signature table: median of each feature per field and date."""
    w = cfg["season_window"]
    per = periods(w["start"], w["end"], w["step_days"])
    chunk = cfg["gee"]["chunk_size"]
    cache_dir = Path(cache_dir or _cache_dir(out_csv, cfg, {"level": level, "n": len(samples),
                                                            "chunk": chunk}))
    samples = _spatial_sort(samples)
    chunks = [samples.iloc[i:i + chunk] for i in range(0, len(samples), chunk)]
    paths = [cache_dir / f"chunk_{i:04d}.csv" for i in range(len(chunks))]

    def fetch(sub, path):
        wide_to_long(_reduce_chunk(sub, cfg, level), per, level).to_csv(path, index=False)

    _run_chunks(chunks, paths, fetch, cfg["gee"].get("workers", 4), out_csv.stem)
    out = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    out.to_csv(out_csv, index=False)
    log.info("wrote %s (%d rows)", out_csv, len(out))
    return out


PIXEL_FEATURES = S2_BANDS + S1_BANDS  # indices are recomputed locally from these


def extract_pixels(samples: gpd.GeoDataFrame, cfg, out_npz: Path,
                   max_pixels_per_request: int = 3500) -> dict:
    """Pixel-level time series: every 10 m pixel inside each sampling geometry.

    Saved as a compact array X[pixel, date, band] (float32) plus the field_id
    of each pixel, so a model can learn from every pixel and vote per field.
    """
    w = cfg["season_window"]
    per = periods(w["start"], w["end"], w["step_days"])
    feats = PIXEL_FEATURES
    cache_dir = _cache_dir(out_npz, cfg, {"level": "pixel", "n": len(samples), "f": feats})
    samples = _spatial_sort(samples)

    # Chunks sized by pixel count: Earth Engine returns at most 5000 features.
    chunks, cur, n = [], [], 0.0
    for i, px in enumerate(samples["approx_pixels"].to_numpy()):
        if cur and n + px > max_pixels_per_request:
            chunks.append(samples.iloc[cur])
            cur, n = [], 0.0
        cur.append(i)
        n += px
    if cur:
        chunks.append(samples.iloc[cur])
    paths = [cache_dir / f"chunk_{i:04d}.npz" for i in range(len(chunks))]
    cols = [f"{f}_t{t:02d}" for t in range(len(per)) for f in feats]

    def fetch(sub, path):
        fc = _to_fc(sub)
        stack, _ = build_stack(fc.geometry().bounds(), cfg, feats)
        out = stack.unmask(NODATA).sampleRegions(collection=fc, properties=["field_id"],
                                                 scale=cfg["gee"]["scale"], tileScale=4,
                                                 geometries=False)
        df = pd.DataFrame([f["properties"] for f in out.getInfo()["features"]])
        if df.empty:
            X, fid = np.zeros((0, len(per), len(feats)), np.float32), np.array([], "U32")
        else:
            X = df.reindex(columns=cols).to_numpy(np.float32)
            X[X == NODATA] = np.nan
            X = X.reshape(len(df), len(per), len(feats))
            fid = df["field_id"].astype(str).to_numpy().astype("U32")  # not object: no pickle
        np.savez_compressed(path, X=X, field_id=fid)

    _run_chunks(chunks, paths, fetch, cfg["gee"].get("workers", 4), out_npz.stem)
    parts = [np.load(p, allow_pickle=False) for p in paths]
    X = np.concatenate([p["X"] for p in parts])
    fid = np.concatenate([p["field_id"] for p in parts])
    dates = np.array([str((a + (b - a) / 2).date()) for a, b in per])
    np.savez_compressed(out_npz, X=X, field_id=fid, bands=np.array(feats), dates=dates)
    log.info("wrote %s (%d pixels from %d fields)", out_npz, len(X), len(np.unique(fid)))
    return {"X": X, "field_id": fid, "bands": np.array(feats), "dates": dates}
