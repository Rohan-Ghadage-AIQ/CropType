"""Load and clean the ground-truth farm shapefile."""
from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

# The shapefile truncates column names to 10 chars; give them readable names.
RENAME = {
    "Crop Name": "crop",
    "Crop Seaso": "crop_season",
    "Crop Type": "crop_type",
    "Crop Stage": "crop_stage",
    "Date Sowin": "sowing_date",
    "Mobile Tim": "survey_time",
    "season_yea": "season_year",
    "boundary_s": "boundary_source",
    "District N": "district",
    "Taluka Nam": "taluka",
    "Village Na": "village",
    "Latitude": "lat",
    "Longitude": "lon",
    "Surveyor N": "surveyor",
    "Image Url": "image_url",
}

KEEP = ["field_id", "crop", "season_year", "crop_type", "crop_stage", "sowing_date",
        "survey_time", "boundary_source", "district", "taluka", "village", "lat", "lon",
        "area_ha", "qc_flags", "train_ok", "split", "surveyor", "image_url", "geometry"]


def load_fields(cfg) -> gpd.GeoDataFrame:
    """Return one clean row per labelled field for the configured season."""
    gdf = gpd.read_file(cfg.path("data", "shapefile"))
    gdf = gdf.rename(columns=RENAME)[KEEP].to_crs(4326)
    n0 = len(gdf)

    d = cfg.data
    gdf = gdf[gdf["season_year"] == d["season"]]
    if d.get("only_train_ok", True):
        gdf = gdf[gdf["train_ok"].astype(bool)]
    if d.get("only_mono", True):
        gdf = gdf[gdf["crop_type"] == "Mono"]
    gdf = gdf[~gdf["crop"].isin(d.get("exclude_crops", []))]
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf["geometry"] = gdf.geometry.make_valid()

    gdf["sowing_date"] = pd.to_datetime(gdf["sowing_date"].str[:10], errors="coerce")
    gdf["survey_time"] = pd.to_datetime(gdf["survey_time"], format="%b %d, %Y %I:%M:%S %p",
                                        errors="coerce")

    # Rare crops -> "Other" (or drop). Counted on the training split only so the
    # test set never influences the class list.
    train_counts = gdf.loc[gdf["split"] == "train", "crop"].value_counts()
    keep = set(train_counts[train_counts >= d["min_class_count"]].index)
    rare = ~gdf["crop"].isin(keep)
    if d.get("other_policy", "group") == "group":
        gdf["label"] = gdf["crop"].where(~rare, "Other")
    else:
        gdf = gdf[~rare]
        gdf["label"] = gdf["crop"]

    log.info("fields: %d raw -> %d after filters (season=%s)", n0, len(gdf), d["season"])
    log.info("label counts:\n%s", gdf["label"].value_counts().to_string())
    return gdf.reset_index(drop=True)
