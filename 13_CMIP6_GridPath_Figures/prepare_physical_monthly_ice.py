"""Actual CMIP6 monthly sea-ice masks on every northern sea-edge geometry."""
from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from physical_ocean_routing import HERE, OUT
from chokepoints24 import xyz, EARTH_KM
from repair_cmip6_monthly_routes import MODELS, open_data, coordinate_arrays

ALL_YEARS = np.uint64((1 << 60) - 1)


def bit_count(values):
    a = np.asarray(values, dtype=np.uint64)
    shape = a.shape
    return np.unpackbits(np.ascontiguousarray(a).view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).reshape(shape)


def main(network_dir=OUT):
    OUT=Path(network_dir)
    geom = np.load(OUT / "edge_geometry.npz")
    points, offsets = geom["geometry"], geom["offsets"]
    north_ids = np.flatnonzero(points[:, 1] >= 55)
    unique, inverse = np.unique(np.round(points[north_ids], 6), axis=0, return_inverse=True)
    masks = np.zeros((12, len(unique)), dtype=np.uint64)
    inventory = pd.read_csv(HERE.parent / "12_CMIP6_Download/data/cmip6_future_model_inventory.csv")
    sources = []
    for mi, model in enumerate(MODELS):
        row = inventory[(inventory.model == model) & (inventory.experiment == "ssp585")].iloc[0]
        sic_ds, h_ds = open_data(row.siconc), open_data(row.thickness)
        sel = np.flatnonzero((sic_ds.time.dt.year.values >= 2091) & (sic_ds.time.dt.year.values <= 2100))
        key = pd.MultiIndex.from_arrays([sic_ds.time.dt.year.values, sic_ds.time.dt.month.values])
        hkey = pd.MultiIndex.from_arrays([h_ds.time.dt.year.values, h_ds.time.dt.month.values])
        hi = hkey.get_indexer(key[sel])
        assert len(sel) == 120 and (hi >= 0).all()
        assert np.all(np.diff(sel)==1) and np.all(np.diff(hi)==1)
        sic = sic_ds.siconc.isel(time=slice(int(sel[0]),int(sel[-1])+1)).to_numpy().astype(np.float32).reshape(120, -1)
        units = str(sic_ds.siconc.attrs.get("units", "")).lower()
        if units in ("1", "fraction", "0-1"):
            sic *= 100
        elif units not in ("%", "percent", "percentage"):
            raise ValueError(f"Unrecognised concentration units: {units}")
        var = "simass" if row.thickness_source == "derived_from_simass" else "sithick"
        thick = h_ds[var].isel(time=slice(int(hi[0]),int(hi[-1])+1)).to_numpy().astype(np.float32).reshape(120, -1)
        if var == "simass":
            thick /= 900 * np.maximum(sic / 100, .01)
        thick = np.where(np.isfinite(sic) & (sic <= 15) & ~np.isfinite(thick), 0, thick)
        lon, lat = coordinate_arrays(sic_ds)
        hlon, hlat = coordinate_arrays(h_ds)
        assert np.allclose(lon,hlon,equal_nan=True) and np.allclose(lat,hlat,equal_nan=True)
        known = np.isfinite(sic) & np.isfinite(thick)
        keep = np.flatnonzero(known.any(axis=0) & np.isfinite(lon) & (lat >= 50))
        tree = cKDTree(xyz(np.c_[lon[keep], lat[keep]]))
        chord, nearest = tree.query(xyz(unique))
        mapping_valid = 2 * EARTH_KM * np.arcsin(np.minimum(chord / 2, 1)) <= 150
        for t, (year, month) in enumerate(key[sel]):
            col = keep[nearest]
            passed = mapping_valid & known[t, col] & (sic[t, col] >= 0) & (sic[t, col] <= 80) & (thick[t, col] >= 0) & (thick[t, col] <= 1.2)
            masks[month - 1, passed] |= np.uint64(1) << np.uint64(mi * 10 + year - 2091)
        sources.append(dict(model=model, siconc=row.siconc, thickness=row.thickness))
        sic_ds.close()
        h_ds.close()
        print("Monthly physical-edge ice", model, "northern sample points", len(unique), flush=True)
    edge_masks = np.full((12, len(offsets) - 1), ALL_YEARS, dtype=np.uint64)
    for month in range(12):
        node_masks = np.full(len(points), ALL_YEARS, dtype=np.uint64)
        node_masks[north_ids] = masks[month, inverse]
        edge_masks[month] = np.bitwise_and.reduceat(node_masks, offsets[:-1])
    np.savez_compressed(OUT / "edge_monthly_ice.npz", model_year_mask=edge_masks)
    (OUT / "monthly_ice_provenance.json").write_text(json.dumps(dict(models=MODELS, sources=sources,
        geometry_sha256=hashlib.sha256((OUT/"edge_geometry.npz").read_bytes()).hexdigest(),
        years="2091-2100", experiment="ssp585", screening="siconc <= 80%, sithick <= 1.2 m",
        north_of_degrees=55, ice_class="simplified_PC6", southern_access_ice_treatment="no sea-ice restriction below 55N",
        climate_nearest_native_cell_coverage_limit_km=150, arctic_access_distance_limit_km=None,
        distinction="150 km limits native climate sampling only, never physical sea access",
        native_resolution_not_downscaled=True), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--network-dir",type=Path,default=OUT)
    main(parser.parse_args().network_dir)
