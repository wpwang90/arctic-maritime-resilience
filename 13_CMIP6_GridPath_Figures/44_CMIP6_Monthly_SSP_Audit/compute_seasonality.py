"""Recompute complete monthly Arctic cores from the original CMIP6 fields."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import repair_cmip6_monthly_routes as routing

RESULTS = HERE / "results"
CACHE = HERE / "cache"
ROUTES = ["NWP", "NEP", "NSR", "TSR"]
PERIODS = {2050: (2041, 2050), 2100: (2091, 2100)}
SSPS = ["ssp245", "ssp585"]
SCREENS = ["ice_capable", "low_ice_sensitivity"]
INVENTORY = ROOT.parent / "12_CMIP6_Download/data/cmip6_future_model_inventory.csv"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def year_month(dataset):
    return pd.MultiIndex.from_arrays([dataset.time.dt.year.values, dataset.time.dt.month.values], names=["year", "month"])


def normalized_concentration(values, units):
    units = str(units).lower().strip()
    if units in ("1", "fraction", "0-1"):
        return values * 100
    if units in ("%", "percent", "percentage"):
        return values
    raise ValueError("Unrecognised concentration unit: " + units)


def convert_thickness(raw, sic, variable, units):
    if variable == "simass":
        assert units in ("kg m-2", "kg m^-2", "kg m**-2")
        raw = raw / (900 * np.maximum(sic/100, .01))
    else:
        assert variable == "sithick" and units == "m"
    # Preserve the existing low-concentration missing-thickness convention.
    return np.where(np.isfinite(sic) & (sic >= 0) & (sic <= 15) & ~np.isfinite(raw), 0, raw)


def map_source(row, period, grids):
    target = CACHE / f"{row.model}_{row.experiment}_{period}.npz"
    audit_path = target.with_suffix(".json")
    if target.exists() and audit_path.exists():
        return json.loads(audit_path.read_text())
    first, last = PERIODS[period]
    s, h = routing.open_data(row.siconc), routing.open_data(row.thickness)
    assert s.attrs["source_id"] == h.attrs["source_id"] == row.model
    assert s.attrs["experiment_id"] == h.attrs["experiment_id"] == row.experiment
    assert s.attrs["variant_label"] == h.attrs["variant_label"]
    skey, hkey = year_month(s), year_month(h)
    assert skey.is_unique and hkey.is_unique
    expected = pd.MultiIndex.from_product([range(first, last+1), range(1, 13)], names=["year", "month"])
    si, hi = skey.get_indexer(expected), hkey.get_indexer(expected)
    assert (si >= 0).all() and (hi >= 0).all() and len(si) == 120
    assert np.all(np.diff(si) == 1) and np.all(np.diff(hi) == 1)
    sv = s.siconc.isel(time=slice(int(si[0]), int(si[-1])+1))
    hv = h[row.thickness_variable].isel(time=slice(int(hi[0]), int(hi[-1])+1))
    sv = sv.transpose("time", *[d for d in sv.dims if d != "time"])
    hv = hv.transpose("time", *[d for d in hv.dims if d != "time"])
    sic = normalized_concentration(sv.to_numpy().astype(np.float32).reshape(120, -1), sv.attrs["units"])
    raw_h = hv.to_numpy().astype(np.float32).reshape(120, -1)
    missing_h = np.isfinite(sic) & (sic >= 0) & (sic <= 15) & ~np.isfinite(raw_h)
    thick = convert_thickness(raw_h, sic, row.thickness_variable, hv.attrs["units"])
    lon, lat = routing.coordinate_arrays(s)
    hlon, hlat = routing.coordinate_arrays(h)
    assert np.allclose(lon, hlon, equal_nan=True) and np.allclose(lat, hlat, equal_nan=True)
    finite = np.isfinite(sic) & np.isfinite(thick)
    valid_range = (sic >= 0) & (sic <= 100) & (thick >= 0)
    known = finite & valid_range
    ocean = np.flatnonzero(finite.any(axis=0) & np.isfinite(lon) & (lat >= 50))
    tree = cKDTree(routing.xyz(np.c_[lon[ocean], lat[ocean]]))
    records = dict(year=expected.get_level_values("year").to_numpy(), month=expected.get_level_values("month").to_numpy())
    winter = []
    compare = []
    old = None
    if period == 2100 and row.experiment == "ssp585":
        old = np.load(routing.OUT / f"native_{row.model}.npz")
        np.testing.assert_array_equal(old["year"], records["year"])
        np.testing.assert_array_equal(old["month"], records["month"])
    old_paths = pd.read_csv(routing.OUT / "monthly_four_route_paths.csv")
    for route, grid in grids.items():
        chord, near = tree.query(routing.xyz(grid["points"]))
        distance = 2*routing.EARTH_KM*np.arcsin(np.minimum(chord/2, 1))
        col = ocean[near]
        coverage = known[:, col] & (distance <= 150)[None, :]
        ice = coverage & (sic[:, col] <= 80) & (thick[:, col] <= 1.2)
        low = coverage & (sic[:, col] <= 15) & (thick[:, col] <= 1.2)
        records[route + "_ice_capable"] = np.packbits(ice, axis=1)
        records[route + "_low_ice_sensitivity"] = np.packbits(low, axis=1)
        records[route + "_known"] = np.packbits(coverage, axis=1)
        records[route + "_sampling_km"] = distance.astype(np.float32)
        if old is not None:
            prior = np.unpackbits(old[route+"_pass"], axis=1, count=len(col)).astype(bool)
            compare.append(dict(model=row.model, route=route, different_mapped_pass_bits=int(np.count_nonzero(prior != ice)),
                                tested_bits=int(prior.size), max_sampling_km=float(distance.max())))
            # Sample original January/February cores independently from the raw fields.
            gt = cKDTree(routing.xyz(grid["points"]))
            for month in [1, 2]:
                p = old_paths[old_paths.route.eq(route) & old_paths.month.eq(month)].sort_values("order")
                if p.empty:
                    continue
                _, ix = gt.query(routing.xyz(p[["lon", "lat"]].to_numpy()))
                cols = col[ix]
                for ti in np.flatnonzero(records["month"] == month):
                    good = coverage[ti, ix]
                    winter.append(dict(model=row.model, route=route, year=int(records["year"][ti]), month=month,
                        path_known=bool(good.all()), passable=bool(ice[ti, ix].all()),
                        max_siconc_pct=float(np.nanmax(sic[ti, cols])), max_thickness_m=float(np.nanmax(thick[ti, cols])),
                        concentration_blocking_nodes=int(np.count_nonzero(sic[ti, cols] > 80)),
                        thickness_blocking_nodes=int(np.count_nonzero(thick[ti, cols] > 1.2)),
                        thickness_filled_nodes=int(missing_h[ti, cols].sum()),
                        unknown_nodes=int((~good).sum()),
                        maximum_climate_sampling_km=float(distance[ix].max())))
    np.savez_compressed(target, **records)
    if old is not None:
        old.close()
    audit = dict(model=row.model, experiment=row.experiment, period=period, year_start=first, year_end=last,
                 climate_member=s.attrs["variant_label"], grid_label=s.attrs.get("grid_label"),
                 concentration_units=sv.attrs["units"], thickness_units=hv.attrs["units"],
                 thickness_variable=row.thickness_variable, calendar=str(s.time.encoding.get("calendar")),
                 monthly_time_alignment="PASS: exact year-month keys, 120 slices",
                 native_invalid_range_cells=int((finite & ~valid_range).sum()),
                 missing_thickness_filled_low_concentration_cells=int(missing_h.sum()),
                 source_files={str(Path(p)):sha(Path(p)) for p in [row.siconc, row.thickness]},
                 comparisons=compare, original_core_winter_values=winter)
    audit_path.write_text(json.dumps(audit, indent=2))
    s.close(); h.close()
    print("Mapped", row.model, row.experiment, first, last, flush=True)
    return audit


def solve_case(experiment, period, screen, route, grid):
    n = len(grid["points"])
    parts, known_parts, labels = [], [], []
    for model in routing.MODELS:
        with np.load(CACHE / f"{model}_{experiment}_{period}.npz") as source:
            parts.append(np.unpackbits(source[route+"_"+screen], axis=1, count=n).astype(bool))
            known_key = route+"_"+screen+"_known"
            if known_key not in source:
                known_key = route+"_known"
            known_parts.append(np.unpackbits(source[known_key], axis=1, count=n).astype(bool))
            labels += [(model, int(y), int(m)) for y, m in zip(source["year"], source["month"])]
    values, known = np.concatenate(parts), np.concatenate(known_parts)
    records, summaries, paths = [], [], []
    for month in range(1, 13):
        ix = np.array([i for i, (_, _, m) in enumerate(labels) if m == month])
        v, k = values[ix], known[ix]
        assert len(ix) == 60
        probability = v.mean(axis=0)
        path = routing.solve_path(grid, probability >= .5, probability)
        passed = v[:, path].all(axis=1) if path else np.zeros(60, bool)
        path_known = k[:, path].all(axis=1) if path else np.zeros(60, bool)
        network = np.array([routing.connectivity(grid, val) for val in v], bool)
        coverage_connected = np.array([routing.connectivity(grid, val) for val in k], bool)
        frequency = passed.mean()
        for i, global_i in enumerate(ix):
            model, year, _ = labels[global_i]
            records.append(dict(experiment=experiment, period=period, screen=screen, route=route, month=month,
                model=model, year=year, common_candidate_exists=bool(path), common_path_known=bool(path_known[i]),
                common_path_passable=bool(passed[i]), member_network_connected=bool(network[i]),
                member_known_coverage_connected=bool(coverage_connected[i])))
        summaries.append(dict(experiment=experiment, period=period, year_start=PERIODS[period][0], year_end=PERIODS[period][1],
            screen=screen, route=route, month=month, model_years=60, models=6,
            common_candidate_exists=bool(path), common_path_support_pct=float(100*frequency),
            admitted=bool(path) and frequency >= .5, supported_model_years=int(passed.sum()),
            common_path_known_model_years=int(path_known.sum()),
            member_network_connected_pct=float(100*network.mean()),
            known_network_coverage_pct=float(100*coverage_connected.mean()),
            data_status="complete_source_ensemble" if coverage_connected.all() else "source_complete_local_mapping_gaps",
            zero_meaning="no_majority_candidate_found" if not path else ("no_member_supports_candidate" if not passed.any() else "nonzero_support"),
            candidate_path_nodes=len(path)))
        for order, node in enumerate(path):
            lon, lat = grid["points"][node]
            paths.append(dict(experiment=experiment, period=period, screen=screen, route=route, month=month,
                              order=order, node=int(node), lon=lon, lat=lat, support_pct=float(100*frequency)))
    print("Solved", experiment, period, screen, route, flush=True)
    return records, summaries, paths


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    protected = [ROOT / "26_MainFig2_LatestNetwork/panel_b_monthly_core_support.csv",
                 routing.OUT / "monthly_four_route_paths.csv", routing.OUT / "model_year_month_feasibility.csv",
                 ROOT / "37_FiveClosures_BoxScatter/figures/MainFig2.pdf"]
    original_hashes = {str(p):sha(p) for p in protected}
    grids = {r:dict(np.load(routing.OUT / f"geometry_{r}.npz")) for r in ROUTES}
    inventory = pd.read_csv(INVENTORY)
    audits = []
    for period in PERIODS:
        for experiment in SSPS:
            for model in routing.MODELS:
                selected = inventory[inventory.model.eq(model) & inventory.experiment.eq(experiment)]
                assert len(selected) == 1
                audits.append(map_source(next(selected.itertuples(index=False)), period, grids))
    pd.DataFrame([x for a in audits for x in a["comparisons"]]).to_csv(RESULTS / "original_raw_field_bit_comparison.csv", index=False)
    winter = pd.DataFrame([x for a in audits for x in a["original_core_winter_values"]])
    winter.to_csv(RESULTS / "winter_original_core_raw_field_audit.csv", index=False)
    all_records, all_summaries, all_paths = [], [], []
    for period in PERIODS:
        for experiment in SSPS:
            for screen in SCREENS:
                for route, grid in grids.items():
                    rec, summ, paths = solve_case(experiment, period, screen, route, grid)
                    all_records += rec; all_summaries += summ; all_paths += paths
    members, summary, paths = pd.DataFrame(all_records), pd.DataFrame(all_summaries), pd.DataFrame(all_paths)
    members.to_csv(RESULTS / "model_year_month_feasibility.csv", index=False)
    summary.to_csv(RESULTS / "monthly_route_support.csv", index=False)
    paths.to_parquet(RESULTS / "candidate_core_paths.parquet", index=False)
    old = pd.read_csv(protected[0])
    principal = summary[summary.period.eq(2100) & summary.experiment.eq("ssp585") & summary.screen.eq("ice_capable")]
    compare = old.merge(principal, on=["route", "month"], validate="one_to_one")
    compare["difference_pp"] = compare.common_path_support_pct - compare.support_pct
    compare.to_csv(RESULTS / "mainfig2b_reproduction.csv", index=False)
    summary.groupby(["period", "experiment", "screen", "route"]).agg(admitted_months=("admitted", "sum"),
        summed_common_support_months=("common_path_support_pct", lambda x:x.sum()/100)).to_csv(RESULTS / "annual_month_summary.csv")
    # Climate classes are independent of vessel categories, not fleet certification.
    winter.groupby(["route", "month", "model"]).agg(supported_years=("passable", "sum"),
        evaluated_years=("year", "size"), highest_path_sic=("max_siconc_pct", "max"),
        highest_path_thickness=("max_thickness_m", "max")).to_csv(RESULTS / "winter_model_summary.csv")
    assert original_hashes == {str(p):sha(p) for p in protected}
    report = dict(status="computed_pending_independent_geometry_checks",
        mainfig2b_max_abs_difference_pp=float(compare.difference_pp.abs().max()),
        raw_pass_mask_different_bits=sum(x["different_mapped_pass_bits"] for a in audits for x in a["comparisons"]),
        scenarios=SSPS, periods=PERIODS, models=routing.MODELS, model_years_per_month=60,
        assumptions=dict(ice_capable="0 <= SIC <= 80%; 0 <= ice thickness <= 1.2 m",
            low_ice_sensitivity="Same thickness requirement; SIC <= 15% instead of 80%; not certified OW operability",
            admission_rule="Common complete candidate passes in at least 30 of 60 model-years",
            climate_sampling="Nearest spherical native climate cell within 150 km; not an access-distance cap or downscaling",
            averaging="Monthly mean fields; a passing month does not guarantee every day is safe",
            route_search="Monthly majority-node candidate, frequency-weighted shortest path; not maximization over all common paths",
            member_network_diagnostic="Connectivity; TSR also requires reachable latitude >=85N, not proof of a loop-free transpolar path"),
        unavailable_scenarios=["ssp126", "ssp370"], unavailable_policy="Not computed; no zero-filled or extrapolated scenarios",
        source_audits=audits, protected_input_sha256=original_hashes,
        geometry_sha256={r:sha(routing.OUT / f"geometry_{r}.npz") for r in ROUTES},
        unchanged="Main figures and downstream network/economic calculations not overwritten")
    (RESULTS / "calculation_audit.json").write_text(json.dumps(report, indent=2))
    print("MainFig2b max difference (percentage points):", report["mainfig2b_max_abs_difference_pp"], flush=True)


if __name__ == "__main__":
    main()
