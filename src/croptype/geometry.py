"""Sampling geometries for the two approaches.

Approach 1 (polygon): field polygon -> inner (negative) buffer -> largest
axis-aligned square that fits inside. The buffer removes edge pixels mixed with
bunds, roads and neighbouring fields; the square gives a compact, pure core.

Approach 2 (point): ground-truth GPS point -> square buffer (default 30 x 30 m,
i.e. 3x3 Sentinel-2 pixels), optionally clipped to the farm polygon.

All metric work happens in the local UTM zone; outputs are back in EPSG:4326.
"""
from __future__ import annotations

import math

import geopandas as gpd
import shapely
from shapely.geometry import Point, box

PIXEL_AREA_M2 = 100.0  # one 10 m Sentinel-2 pixel


def largest_inner_square(geom, min_half_m: float = 0.0):
    """Largest axis-aligned square inscribed in the maximum inscribed circle.

    Returns (square, half_side_m). Cheap and robust: a square of half-side
    r/sqrt(2) centred on the pole of inaccessibility is always fully inside.
    """
    if geom.is_empty:
        return geom, 0.0
    line = shapely.maximum_inscribed_circle(geom, tolerance=0.5)
    centre = Point(line.coords[0])
    radius = line.length
    half = radius / math.sqrt(2)
    if half < min_half_m:
        return None, half
    return box(centre.x - half, centre.y - half, centre.x + half, centre.y + half), half


def polygon_samples(fields: gpd.GeoDataFrame, inner_buffer_m: float, mode: str = "square",
                    min_square_half_m: float = 5.0) -> gpd.GeoDataFrame:
    """Approach 1 geometries, one per field, with QC columns."""
    utm = fields.estimate_utm_crs()
    f = fields.to_crs(utm)
    geoms, method, half_sides = [], [], []
    for g in f.geometry:
        inner = g.buffer(-inner_buffer_m)
        used = "inner_buffer"
        if inner.is_empty or inner.area < PIXEL_AREA_M2:
            # Field too small or too thin to shrink: keep the raw polygon, flag it.
            inner, used = g, "no_buffer_small_field"
        half = math.nan
        if mode == "square":
            sq, half = largest_inner_square(inner, min_square_half_m)
            if sq is not None:
                inner, used = sq, used + "+square"
        geoms.append(inner)
        method.append(used)
        half_sides.append(half)

    out = f[["field_id"]].copy()
    out = gpd.GeoDataFrame(out, geometry=geoms, crs=utm)
    out["sample_mode"] = "polygon"
    out["geom_method"] = method
    out["square_half_m"] = half_sides
    out["approx_pixels"] = (out.geometry.area / PIXEL_AREA_M2).round(1)
    return out.to_crs(4326)


def point_samples(fields: gpd.GeoDataFrame, half_m: float = 15.0,
                  clip_to_polygon: bool = True) -> gpd.GeoDataFrame:
    """Approach 2 geometries: square buffer around each ground-truth point."""
    pts = gpd.GeoSeries(gpd.points_from_xy(fields["lon"], fields["lat"]), crs=4326)
    utm = fields.estimate_utm_crs()
    pts_u = pts.to_crs(utm)
    polys_u = fields.geometry.to_crs(utm)

    geoms, method, dist = [], [], []
    for p, poly in zip(pts_u, polys_u):
        sq = box(p.x - half_m, p.y - half_m, p.x + half_m, p.y + half_m)
        d = poly.distance(p)
        used = "point_square"
        if clip_to_polygon and d == 0:
            clipped = sq.intersection(poly)
            if clipped.area >= PIXEL_AREA_M2:
                sq, used = clipped, "point_square_clipped"
        if d > 0:
            used += "_point_outside_polygon"
        geoms.append(sq)
        method.append(used)
        dist.append(round(d, 1))

    out = gpd.GeoDataFrame(fields[["field_id"]].copy(), geometry=geoms, crs=utm)
    out["sample_mode"] = "point"
    out["geom_method"] = method
    out["point_dist_to_polygon_m"] = dist
    out["approx_pixels"] = (out.geometry.area / PIXEL_AREA_M2).round(1)
    return out.to_crs(4326)
