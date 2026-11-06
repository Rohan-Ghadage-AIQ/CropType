"""Feature definitions shared by extraction, signatures and training.

Sentinel-2 L2A has 12 surface-reflectance bands: B1-B9, B8A, B11, B12.
There is no B13-B15, and B10 (cirrus) is dropped from L2A because it holds no
surface signal. So "B1..B15" in practice means these 12 bands.
"""
from __future__ import annotations

import numpy as np

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S1_BANDS = ["VV", "VH"]              # backscatter in dB
S1_DERIVED = ["VH_VV", "RVI"]        # VH-VV ratio (dB) and radar vegetation index
INDICES = ["NDVI", "EVI", "NDMI", "NDRE", "SAVI", "GNDVI"]

FEATURES = INDICES + S1_BANDS + S1_DERIVED + S2_BANDS


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def compute_indices(df):
    """Add the vegetation indices to a frame of S2 reflectances (0-1 scale).

    Same formulas as the Earth Engine side (gee_extract.add_indices), used when
    you have reflectances from another source (local GeoTIFFs, Sentinel Hub...).
    """
    b2, b3, b4, b5, b8, b11 = (df[b] for b in ("B2", "B3", "B4", "B5", "B8", "B11"))
    df = df.copy()
    df["NDVI"] = _nd(b8, b4)
    df["EVI"] = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1)
    df["NDMI"] = _nd(b8, b11)
    df["NDRE"] = _nd(b8, b5)
    df["SAVI"] = 1.5 * (b8 - b4) / (b8 + b4 + 0.5)
    df["GNDVI"] = _nd(b8, b3)
    return df
