"""Audit NWP disconnection and solve four corridors using actual monthly CMIP6.

Geometry is refined independently of climate resolution. Native ocean-cell
values are nearest-neighbour mapped on a sphere; this is not downscaling.
No schematic fallback or relaxation of the ice threshold is permitted.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from chokepoints24 import HERE, EARTH_KM, xyz
from analyze_ethan_trajectory_arctic_embedding import haversine_vec

OUT = HERE / "data/cmip6_monthly_repair"
LAND = HERE / "data/public/naturalearth_10m/ne_10m_land.shp"
MODELS = ["ACCESS-CM2", "BCC-CSM2-MR", "CNRM-ESM2-1", "GFDL-CM4", "IPSL-CM6A-LR", "MIROC6"]
DEFINITIONS = {
    "NWP": dict(bounds=(-175, -45, 55, 77), step=.1, endpoints=[(-170, 66), (-55, 58)]),
    "NEP": dict(bounds=(0, 195, 58, 83), step=.25, endpoints=[(-170, 66), (0, 62)]),
    "NSR": dict(bounds=(35, 195, 60, 83), step=.25, endpoints=[(-170, 66), (35, 70)]),
    "TSR": dict(bounds=(0, 360, 58, 89), step=.25, endpoints=[(-170, 66), (0, 62)], polar=True),
}


def normalise(lon):
    return (np.asarray(lon) + 180) % 360 - 180


def prepare_land():
    frame = gpd.read_file(LAND)
    clip = shapely.box(-180, 54, 180, 90)
    land = shapely.union_all([g.intersection(clip) for g in frame.geometry if g.bounds[3] >= 54])
    shapely.prepare(land)
    return land


def crosses_land(land, a, b):
    result = np.zeros(len(a), dtype=bool)
    wrapping = np.abs(a[:, 0] - b[:, 0]) > 180
    for start in range(0, len(a), 100000):
        ids = np.arange(start, min(start + 100000, len(a)))
        ids = ids[~wrapping[ids]]
        lines = shapely.linestrings(np.stack([a[ids], b[ids]], axis=1))
        result[ids] = shapely.intersects(land, lines)
    for i in np.flatnonzero(wrapping):
        p, q = a[i].copy(), b[i].copy()
        shift = ((q[0] - p[0] + 180) % 360) - 180
        boundary = 180 if shift > 0 else -180
        f = (boundary - p[0]) / shift
        lat = p[1] + f * (q[1] - p[1])
        result[i] = bool(shapely.intersects(land, shapely.LineString([p, (boundary, lat)])) or
            shapely.intersects(land, shapely.LineString([(-boundary, lat), q])))
    return result


def make_grid(route, land):
    cache = OUT / f"geometry_{route}.npz"
    if cache.exists():
        return dict(np.load(cache))
    cfg = DEFINITIONS[route]
    lo, hi, bottom, top = cfg["bounds"]
    step = cfg["step"]
    x = np.arange(lo, hi + (0 if route == "TSR" else step / 2), step)
    y = np.arange(bottom, top + step / 2, step)
    xx, yy = np.meshgrid(x, y)
    coords = np.c_[normalise(xx.ravel()), yy.ravel()]
    ocean = ~shapely.contains_xy(land, coords[:, 0], coords[:, 1])
    active = np.flatnonzero(ocean)
    compact = np.full(len(coords), -1, dtype=np.int32)
    compact[active] = np.arange(len(active))
    row, col = [], []
    ids = np.arange(len(coords)).reshape(xx.shape)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = ids[:len(y) - dy or None]
        b = np.roll(ids, -dx, axis=1)[dy:]
        if route != "TSR" and dx:
            a, b = (a[:, :-1], b[:, :-1]) if dx > 0 else (a[:, 1:], b[:, 1:])
        a, b = a.ravel(), b.ravel()
        keep = ocean[a] & ocean[b]
        a, b = a[keep], b[keep]
        keep = ~crosses_land(land, coords[a], coords[b])
        row.append(compact[a[keep]])
        col.append(compact[b[keep]])
    row, col = np.concatenate(row), np.concatenate(col)
    points = coords[active]
    km = haversine_vec(points[row, 0], points[row, 1], points[col, 0], points[col, 1])
    tree = cKDTree(xyz(points))
    _, endpoints = tree.query(xyz(np.asarray(cfg["endpoints"])))
    polar = points[:, 1] >= 85 if route == "TSR" else np.zeros(len(points), dtype=bool)
    output = dict(points=points, row=row, col=col, km=km, endpoints=endpoints, polar=polar)
    np.savez_compressed(cache, **output)
    print(route, "water nodes", len(points), "land-safe edges", len(row), flush=True)
    return output


def coordinate_arrays(ds):
    lon = next(ds[n].values for n in ("longitude", "lon", "nav_lon") if n in ds)
    lat = next(ds[n].values for n in ("latitude", "lat", "nav_lat") if n in ds)
    if lon.ndim == lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    return normalise(lon).ravel(), lat.ravel()


def open_data(path):
    for engine in ("h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, engine=engine)
        except (OSError, ValueError, TypeError):
            continue
    raise RuntimeError(f"Unreadable climate input {path}")


def prepare_model(model, grids, inventory):
    target = OUT / f"native_{model}.npz"
    if target.exists():
        return
    r = inventory[(inventory.model == model) & (inventory.experiment == "ssp585")].iloc[0]
    sic_ds, thick_ds = open_data(r.siconc), open_data(r.thickness)
    sic_time = pd.MultiIndex.from_arrays([sic_ds.time.dt.year.values, sic_ds.time.dt.month.values])
    thick_time = pd.MultiIndex.from_arrays([thick_ds.time.dt.year.values, thick_ds.time.dt.month.values])
    select = np.flatnonzero((sic_ds.time.dt.year.values >= 2091) & (sic_ds.time.dt.year.values <= 2100))
    ti = thick_time.get_indexer(sic_time[select])
    assert (ti >= 0).all() and len(select) == 120
    assert np.all(np.diff(select) == 1) and np.all(np.diff(ti) == 1)
    concentration = sic_ds.siconc.isel(time=slice(int(select[0]), int(select[-1]) + 1)).to_numpy().astype(np.float32).reshape(120, -1)
    units = str(sic_ds.siconc.attrs.get("units", "")).lower()
    if units in ("1", "fraction", "0-1"):
        concentration *= 100
    elif units not in ("%", "percent", "percentage"):
        raise ValueError(f"Unrecognised sea-ice units: {units}")
    variable = "simass" if r.thickness_source == "derived_from_simass" else "sithick"
    thickness = thick_ds[variable].isel(time=slice(int(ti[0]), int(ti[-1]) + 1)).to_numpy().astype(np.float32).reshape(120, -1)
    lon, lat = coordinate_arrays(sic_ds)
    lon_h, lat_h = coordinate_arrays(thick_ds)
    assert len(lon) == len(lon_h) and np.allclose(lon, lon_h, equal_nan=True) and np.allclose(lat, lat_h, equal_nan=True)
    if variable == "simass":
        thickness /= 900 * np.maximum(concentration / 100, .01)
    no_ice = np.isfinite(concentration) & (concentration <= 15)
    thickness = np.where(no_ice & ~np.isfinite(thickness), 0, thickness)
    known = np.isfinite(concentration) & np.isfinite(thickness)
    passes = known & (concentration <= 80) & (thickness <= 1.2)
    ocean = known.any(axis=0) & (lat >= 50) & np.isfinite(lon)
    indices = np.flatnonzero(ocean)
    tree = cKDTree(xyz(np.c_[lon[indices], lat[indices]]))
    records = dict(year=sic_ds.time.dt.year.values[select], month=sic_ds.time.dt.month.values[select])
    for route, grid in grids.items():
        chord, nearest = tree.query(xyz(grid["points"]))
        distance = 2 * EARTH_KM * np.arcsin(np.minimum(chord / 2, 1))
        node_indices = indices[nearest]
        mapped = passes[:, node_indices] & (distance[None, :] <= 150)
        coverage = known[:, node_indices] & (distance[None, :] <= 150)
        records[f"{route}_pass"] = np.packbits(mapped, axis=1)
        records[f"{route}_known"] = np.packbits(coverage, axis=1)
        records[f"{route}_sampling_km"] = distance.astype(np.float32)
    records["siconc_source"] = str(r.siconc)
    records["thickness_source"] = str(r.thickness)
    np.savez_compressed(target, **records)
    sic_ds.close()
    thick_ds.close()
    print("Mapped actual monthly fields:", model, flush=True)


def graph_for(grid, allowed, probability=None):
    row, col = grid["row"], grid["col"]
    keep = allowed[row] & allowed[col]
    weights = grid["km"][keep]
    if probability is not None:
        weights = weights * (1 + 2 * (1 - .5 * (probability[row[keep]] + probability[col[keep]])))
    return coo_matrix((np.r_[weights, weights], (np.r_[row[keep], col[keep]], np.r_[col[keep], row[keep]])),
        shape=(len(allowed), len(allowed))).tocsr()


def connectivity(grid, allowed):
    a, b = grid["endpoints"]
    if not allowed[a] or not allowed[b]:
        return False
    _, label = connected_components(graph_for(grid, allowed), directed=False)
    if label[a] != label[b]:
        return False
    return not grid["polar"].any() or np.any(allowed & grid["polar"] & (label == label[a]))


def solve_path(grid, allowed, probability=None):
    a, b = map(int, grid["endpoints"])
    if not allowed[a] or not allowed[b]:
        return []
    graph = graph_for(grid, allowed, probability)
    da, pa = dijkstra(graph, directed=False, indices=a, return_predecessors=True)
    if not np.isfinite(da[b]):
        return []
    def backtrack(pred, source, target):
        path = [int(target)]
        while path[-1] != source:
            p = int(pred[path[-1]])
            if p < 0:
                return []
            path.append(p)
        return path[::-1]
    if grid["polar"].any():
        db, pb = dijkstra(graph, directed=False, indices=b, return_predecessors=True)
        candidate = np.flatnonzero(grid["polar"] & allowed & np.isfinite(da + db))
        if not len(candidate):
            return []
        middle = int(candidate[np.argmin((da + db)[candidate])])
        first, second = backtrack(pa, a, middle), backtrack(pb, b, middle)
        path = first + second[-2::-1]
        if len(path) != len(set(path)):
            return []
        return path
    return backtrack(pa, a, b)


def audit_old_nwp():
    df = pd.read_csv(HERE / "data/cmip6_grid_feasible_months.csv")
    g = df[(df.experiment == "ssp585") & (df.period == 2100) & (df["class"] == "PC6") & (df.lon <= -35)]
    points = g[["lon", "lat"]].to_numpy()
    keys = {tuple(x): i for i, x in enumerate(points)}
    row, col = [], []
    for key, i in keys.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            j = keys.get((key[0] + dx, key[1] + dy))
            if j is not None:
                row.append(i)
                col.append(j)
    graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(points), len(points))).tocsr()
    _, label = connected_components(graph, directed=False)
    tree = cKDTree(xyz(points))
    endpoints = tree.query(xyz(np.asarray(DEFINITIONS["NWP"]["endpoints"])))[1]
    return dict(old_nwp_connected_even_with_all_ice_removed=bool(label[endpoints[0]] == label[endpoints[1]]),
        old_graph_components=int(label.max() + 1), old_node_count=len(points),
        old_endpoint_components=list(map(int, label[endpoints])),
        root_cause="one-degree nine-sample any-land mask and lost narrow-waterway topology; monthly fields were replaced by annual-month heuristics",
        prior_drawn_nwp_source="sampled corridor fallback, not a proven continuous sea-ice-feasible path")


def repaired_route_metadata():
    paths = pd.read_csv(OUT / "monthly_four_route_paths.csv")
    summary = pd.read_csv(OUT / "monthly_four_route_summary.csv")
    metadata = {}
    for r in summary.itertuples():
        if not r.cmip6_continuous:
            continue
        g = paths[(paths.route == r.route) & (paths.month == r.month)].sort_values("order")
        if g.empty:
            raise ValueError("Feasible route has no geometry")
        metadata[(r.route, int(r.month))] = dict(
            gateway_a_lon=float(g.iloc[0].lon), gateway_a_lat=float(g.iloc[0].lat),
            gateway_b_lon=float(g.iloc[-1].lon), gateway_b_lat=float(g.iloc[-1].lat),
            arctic_middle_km=float(r.path_km), path_mean_months=float(r.path_mean_months),
            threshold_used=float(r.threshold_used), path_source=r.path_source,
            cmip6_continuous=True, common_path_feasible_fraction=float(r.common_path_feasible_fraction))
    return paths, metadata


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    land = prepare_land()
    grids = {r: make_grid(r, land) for r in DEFINITIONS}
    audit = audit_old_nwp()
    audit["refined_ice_free_nwp_connected"] = connectivity(grids["NWP"], np.ones(len(grids["NWP"]["points"]), dtype=bool))
    inventory = pd.read_csv(HERE.parent / "12_CMIP6_Download/data/cmip6_future_model_inventory.csv")
    for model in MODELS:
        prepare_model(model, grids, inventory)
    members, path_rows, summaries = [], [], []
    for route, grid in grids.items():
        n = len(grid["points"])
        fields, labels = [], []
        for model in MODELS:
            with np.load(OUT / f"native_{model}.npz") as source:
                fields.append(np.unpackbits(source[f"{route}_pass"], axis=1, count=n).astype(bool))
                labels += [(model, int(y), int(m)) for y, m in zip(source["year"], source["month"])]
        fields = np.concatenate(fields)
        annual_months = fields.mean(axis=0) * 12
        for month in range(1, 13):
            selection = np.asarray([i for i, (_, _, m) in enumerate(labels) if m == month])
            values = fields[selection]
            probability = values.mean(axis=0)
            path = solve_path(grid, probability >= .5, probability)
            path_frequency = float(np.mean(np.all(values[:, path], axis=1))) if path else 0.0
            possible = []
            for index, value in zip(selection, values):
                feasible = connectivity(grid, value)
                possible.append(feasible)
                model, year, _ = labels[index]
                members.append(dict(model=model, experiment="ssp585", year=year, month=month, route=route,
                    ship_class="PC6", network_path_exists=feasible,
                    common_path_passable=bool(np.all(value[path])) if path else False))
            valid = bool(path) and path_frequency >= .5
            km = float(haversine_vec(grid["points"][path[:-1], 0], grid["points"][path[:-1], 1],
                grid["points"][path[1:], 0], grid["points"][path[1:], 1]).sum()) if path else np.nan
            source = "actual_monthly_CMIP6_land_checked_ensemble" if valid else "no_ensemble_supported_monthly_path"
            summary = dict(route=route, month=month, scenario="SSP5-8.5", period=2100, ship_class="PC6",
                path_km=km, path_mean_months=float(annual_months[path].mean()) if path else np.nan,
                threshold_used=.5, path_source=source, cmip6_continuous=valid,
                common_path_feasible_fraction=path_frequency, any_path_feasible_fraction=float(np.mean(possible)),
                n_model_years=len(values), models=";".join(MODELS), geometry_step_degrees=DEFINITIONS[route]["step"])
            summaries.append(summary)
            for order, node in enumerate(path):
                lon, lat = grid["points"][node]
                path_rows.append(dict(route=route, month=month, order=order, lon=lon, lat=lat, **{k: v for k, v in summary.items() if k not in ("route", "month")}))
            print(route, month, "common", path_frequency, "any", float(np.mean(possible)), "km", round(km, 1), flush=True)
    pd.DataFrame(path_rows).to_csv(OUT / "monthly_four_route_paths.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "monthly_four_route_summary.csv", index=False)
    pd.DataFrame(members).to_csv(OUT / "model_year_month_feasibility.csv", index=False)
    audit.update(models=MODELS, model_years_per_month=60, years="2091-2100", experiment="ssp585", ship_class="PC6",
        geography_source="https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
        ice_criterion="siconc <= 80 percent AND sithick <= 1.2 m; BCC simass/(900*siconc_fraction)",
        ice_criterion_is_operational_certification=False,
        climate_sampling="nearest finite native ocean cell in spherical distance, maximum 150 km; not climate downscaling",
        geometry_tests="each edge tested for land intersection; NWP constrained to Canadian archipelago; TSR must reach 85N",
        monthly_reliability_threshold=.5, missing_as_open_water=False,
        native_climate_resolution_unaltered=True, no_schematic_fallback=True)
    (OUT / "NWP_diagnostic.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
