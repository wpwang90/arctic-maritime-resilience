"""AIS-first 2024 exposure and 24 single-closure physical opportunity tests.

This is not a forecast of cargo, economic losses, or certified fleet navigation.
Published scalar detour distances never enter the calculation.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import time
import zipfile

import numpy as np
import pandas as pd

from chokepoints24 import Exposure, xyz, EARTH_KM
from physical_ocean_routing import length
from prepare_physical_monthly_ice import ALL_YEARS, bit_count, main as prepare_ice
from recompute_physical_arctic_embedding import PhysicalRouter, core_ice_masks
from repair_cmip6_monthly_routes import repaired_route_metadata

ROOT = Path(__file__).resolve().parent
NET = ROOT / "data/physical_arctic_embedding/oxmartrans_1390"
OUT = ROOT / "25_MainFig1_LatestNetwork"
SHIPS = ["container", "bulk", "tanker", "lng"]
ROUTES = ["NWP", "NEP", "NSR", "TSR"]
VERSION = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT / "routing").mkdir(exist_ok=True)
    climate = OUT / "climate"
    climate.mkdir(exist_ok=True)
    for name in ["edge_geometry.npz"]:
        target = climate / name
        if not target.exists():
            target.symlink_to(NET / name)
    manifest = {name: sha(NET / name) for name in [
        "physical_edges.parquet", "edge_geometry.npz", "node_coordinates.npy",
        "canonical_port_registry.csv", "monthly_od_repaired.parquet",
        "canonical_endpoint_access.csv", "original_AIS_member_to_main_port.parquet"]}
    manifest.update({"version": VERSION, "chokepoints": sha(ROOT / "data/chokepoints24/registry.csv"),
                     "ais_summary": sha(ROOT / "data/ethan_representative_trajectory_summary.parquet"),
                     "ais_exposure": sha(ROOT / "data/chokepoints24/ais_member_exposure.parquet")})
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2))
    old = climate / "monthly_ice_provenance.json"
    if not old.exists() or json.loads(old.read_text())["geometry_sha256"] != manifest["edge_geometry.npz"]:
        prepare_ice(climate)
    mapping = pd.read_parquet(NET / "original_AIS_member_to_main_port.parquet")
    mapping = mapping[mapping.accounting_bucket.eq("retained_interport")]
    summary = pd.read_parquet(ROOT / "data/ethan_representative_trajectory_summary.parquet")
    quality = summary[["member", "direct_quality", "endpoint_label_consistent", "raw_valid_distance_km",
                       "raw_points", "duration_days", "large_jump_count", "portcall_weight_matched"]]
    candidates = mapping.merge(quality, on="member", validate="one_to_one")
    candidates = candidates.merge(pd.read_parquet(ROOT / "data/chokepoints24/ais_member_exposure.parquet"),
                                  on="member", validate="one_to_one")
    candidates["accepted_observed"] = (candidates.direct_quality.eq(1)
        & candidates.endpoint_label_consistent.eq(1) & candidates.large_jump_count.eq(0)
        & candidates.raw_valid_distance_km.gt(0))
    group = ["shiptype", "start_month", "origin_port_id", "dest_port_id"]
    # A deterministic representative, not a synthetic average of different geometries.
    chosen = candidates[candidates.accepted_observed].sort_values(
        ["activity_weight", "raw_valid_distance_km", "member"], ascending=[False, True, True]).drop_duplicates(group)
    chosen = chosen[group + ["member", "raw_valid_distance_km", "mask_75", "mask_100", "mask_150", "raw_points"]]
    demand = pd.read_parquet(NET / "monthly_od_repaired.parquet")
    demand = demand.merge(chosen, on=group, how="left", validate="one_to_one")
    demand["has_observed"] = demand.member.notna()
    demand["od_id"] = np.arange(len(demand))
    demand.to_parquet(OUT / "demand_ais_first.parquet", index=False)
    candidates.to_parquet(OUT / "ais_candidate_audit.parquet", index=False)
    report = dict(od_month_type_rows=len(demand), matched_archive_members=len(mapping),
                  accepted_archive_members=int(candidates.accepted_observed.sum()),
                  observed_representative_rows=int(demand.has_observed.sum()),
                  fallback_rows=int((~demand.has_observed).sum()),
                  total_activity_weight=float(demand.activity_weight.sum()),
                  observed_activity_pct=float(100*demand.loc[demand.has_observed, "activity_weight"].sum()/demand.activity_weight.sum()),
                  selection="direct_quality=1; label-consistent endpoints; no >150 km jumps; largest support weight, then shortest actual polyline",
                  weights="Inherited capacity-normalized call activity index, not tonnes, revenue, or independent vessel calls",
                  month="Actual trajectory starting month, not every month crossed by a multi-month file")
    (OUT / "ais_preference_audit.json").write_text(json.dumps(report, indent=2))
    print("PREPARED", report, flush=True)


def group_run(ship, month):
    started = time.monotonic()
    prefix = OUT / "routing" / f"{ship}_{month:02d}"
    signature = sha(OUT / "input_manifest.json")
    marker = Path(str(prefix) + "_audit.json")
    if marker.exists() and json.loads(marker.read_text()).get("input_signature") == signature:
        return
    od = pd.read_parquet(OUT / "demand_ais_first.parquet")
    od = od[(od.shiptype == ship) & (od.start_month == month)].reset_index(drop=True)
    edges = pd.read_parquet(NET / "physical_edges.parquet")
    coords = np.load(NET / "node_coordinates.npy")
    geometry_ice = np.load(OUT / "climate/edge_monthly_ice.npz")["model_year_mask"][month-1]
    observed = pd.read_parquet(ROOT / "data/ethan_representative_ais_grid_edges_1deg.parquet",
                              filters=[("shiptype", "==", ship), ("month", "==", month)])
    active = observed.merge(pd.read_parquet(NET / "ais_water_edges.parquet"), on=["x1", "y1", "x2", "y2"])
    keys = set(zip(np.minimum(active.a, active.b), np.maximum(active.a, active.b)))
    monthly_supported = np.array([r.source != "AIS_supported_water_grid" or (r.a, r.b) in keys for r in edges.itertuples()])
    # Missing AIS sampling in a ship-month is not evidence that a waterway closes.
    # Retain the full validated geometry backbone; monthly observations determine
    # OD activity and the preferred observed baseline, not physical edge existence.
    # Conventional routes are a 2024 geographic stress test. Future climate is
    # applied only to the candidate Arctic layer, not retroactively to observed AIS.
    conventional = PhysicalRouter(edges, coords, np.full(len(geometry_ice), ALL_YEARS, np.uint64), 100)
    ice_edges = edges[bit_count(geometry_ice[edges.geometry_id.to_numpy()]) >= 30].reset_index(drop=True)
    arctic = PhysicalRouter(ice_edges, coords, geometry_ice, 100)
    endpoints = pd.read_csv(NET / "canonical_endpoint_access.csv").set_index("key")
    n = len(od)
    source = conventional.to_compact[endpoints.loc[od.origin_port_id, "node"].to_numpy(int)]
    target = conventional.to_compact[endpoints.loc[od.dest_port_id, "node"].to_numpy(int)]
    if (source < 0).any() or (target < 0).any():
        raise ValueError("A monthly public-backbone endpoint is missing")
    graph_base = conventional.solve(source, target, np.uint32(0))
    graph_km = np.asarray(conventional.length)[graph_base]
    base_km = np.where(od.has_observed, od.raw_valid_distance_km, graph_km)
    base_masks = np.where(od.has_observed, od.mask_100.fillna(0),
                          np.asarray(conventional.path_mask)[graph_base]).astype(np.uint32)
    od["baseline_km"] = base_km
    od["baseline_mask"] = base_masks
    od["baseline_source"] = np.where(od.has_observed, "observed_AIS_polyline", "water_network_shortest_path")
    od["baseline_graph_path_id"] = graph_base
    points, metadata = repaired_route_metadata()
    gateways = pd.read_csv(NET / "gateway_mapping.csv")
    classifier = Exposure(pd.read_csv(ROOT / "data/chokepoints24/registry.csv"))
    core_masks = core_ice_masks()
    ps = arctic.to_compact[endpoints.loc[od.origin_port_id, "node"].to_numpy(int)]
    pt = arctic.to_compact[endpoints.loc[od.dest_port_id, "node"].to_numpy(int)]
    valid = (ps >= 0) & (pt >= 0)
    unique = np.unique(np.r_[ps[ps >= 0], pt[pt >= 0]])
    slots = {x: i for i, x in enumerate(unique)}
    ss = np.array([slots.get(x, 0) for x in ps])
    tt = np.array([slots.get(x, 0) for x in pt])
    gates, cores = {}, {}
    for route in ROUTES:
        if (route, month) not in metadata:
            continue
        r = gateways[(gateways.route == route) & (gateways.month == month)].set_index("side")
        pair = []
        for side in ["a", "b"]:
            key = r.loc[side, "key"]
            node = int(arctic.to_compact[int(endpoints.loc[key, "node"])])
            if node < 0:
                break
            if key not in gates:
                gates[key] = dict(node=node, offset=len(gates)*len(unique))
            pair.append(gates[key])
        if len(pair) == 2:
            p = points[(points.route == route) & (points.month == month)].sort_values("order")[["lon", "lat"]].to_numpy()
            cores[route] = dict(pair=pair, km=length(p), ice=core_masks[(route, month)], mask=classifier.masks(p)[100])
    ars = np.concatenate([np.full(len(unique), g["node"]) for g in gates.values()]).astype(int) if gates else np.array([], int)
    art = np.tile(unique, len(gates))
    ars_base = arctic.solve(ars, art, np.uint32(0)) if gates else np.array([], int)
    solutions, sums = [], []
    loop_cache = {}
    rejected_loop = rejected_ice = 0
    registry = pd.read_csv(ROOT / "data/chokepoints24/registry.csv")
    for si in range(25):
        closed = np.uint32(0 if si == 0 else 1 << (si-1))
        affected = (base_masks & closed) != 0
        gid = graph_base.copy() if si == 0 else conventional.solve(source, target, closed, graph_base[None, :])
        # Unexposed observed routes must not be silently replaced by optimized paths.
        without = np.where(affected, np.asarray(conventional.length)[gid], base_km)
        candidate_ids = ars_base if si == 0 else (arctic.solve(ars, art, closed, ars_base[None, :]) if gates else ars_base)
        best = without.copy()
        code = np.zeros(n, np.int8)
        entry, exit_ = np.zeros(n, np.int32), np.zeros(n, np.int32)
        support = np.full(n, np.nan)
        lengths, ice = np.asarray(arctic.length), np.asarray(arctic.path_ice, np.uint64)
        for ri, route in enumerate(ROUTES, 1):
            if route not in cores or cores[route]["mask"] & int(closed):
                continue
            c = cores[route]
            for left, right in (c["pair"], c["pair"][::-1]):
                p = candidate_ids[left["offset"] + ss]
                q = candidate_ids[right["offset"] + tt]
                distance = lengths[p] + c["km"] + lengths[q]
                joint = bit_count(ice[p] & ice[q] & c["ice"])
                improve = valid & np.isfinite(distance) & (distance < best-1e-7)
                rejected_ice += int((improve & (joint < 30)).sum())
                improve &= joint >= 30
                for j in np.flatnonzero(improve):
                    pair = (int(p[j]), int(q[j]))
                    if pair not in loop_cache:
                        loop_cache[pair] = not bool(set(arctic.paths[pair[0]]) & set(arctic.paths[pair[1]]))
                    if not loop_cache[pair]:
                        improve[j] = False
                        rejected_loop += 1
                best[improve], code[improve] = distance[improve], ri
                entry[improve], exit_[improve] = p[improve], q[improve]
                support[improve] = joint[improve]/60
        if si == 0:
            open_code = code.copy()
        # All scenario rows are saved; panel c uses only baseline-exposed OD rows.
        row = pd.DataFrame(dict(od_id=od.od_id, scenario_index=si, affected=affected,
            baseline_km=base_km, without_km=without, with_km=best, route_code=code,
            conventional_path_id=gid, entry_path_id=entry, exit_path_id=exit_,
            joint_model_year_fraction=support, open_route_code=open_code))
        solutions.append(row)
        w = od.activity_weight.to_numpy()
        sums.append(dict(shiptype=ship, month=month, scenario_index=si,
            scenario="open" if si == 0 else registry.iloc[si-1]["key"],
            total_weight=float(w.sum()), affected_weight=float(w[affected].sum()),
            arctic_weight=float(w[affected & (code > 0)].sum()),
            new_arctic_weight=float(w[affected & (code > 0) & (open_code == 0)].sum()),
            no_bypass_weight=float(w[affected & ~np.isfinite(without)].sum()),
            rescued_weight=float(w[affected & ~np.isfinite(without) & np.isfinite(best)].sum()),
            residual_weight=float(w[affected & ~np.isfinite(best)].sum())))
    od.to_parquet(str(prefix)+"_baseline.parquet", index=False)
    pd.concat(solutions, ignore_index=True).to_parquet(str(prefix)+"_solutions.parquet", index=False)
    pd.DataFrame(sums).to_csv(str(prefix)+"_summary.csv", index=False)
    conventional.save_catalogue(Path(str(prefix)+"_conventional"))
    arctic.save_catalogue(Path(str(prefix)+"_arctic"))
    audit = dict(input_signature=signature, shiptype=ship, month=month, scenarios=25,
        od_rows=n, observed_rows=int(od.has_observed.sum()), conventional_edges=len(edges), arctic_edges=len(ice_edges),
        backbone_edges_without_this_ship_month_AIS_support=int((~monthly_supported).sum()),
        available_core_geometries=list(cores), conventional_trees=conventional.dijkstra_trees,
        arctic_trees=arctic.dijkstra_trees, joint_ice_rejections=rejected_ice,
        repeated_access_node_rejections=rejected_loop, elapsed_seconds=time.monotonic()-started)
    marker.write_text(json.dumps(audit, indent=2))
    print("COMPLETED", ship, month, n, round(audit["elapsed_seconds"], 1), flush=True)


def collect():
    baselines = list((OUT/"routing").glob("*_baseline.parquet"))
    assert len(baselines) == 48
    baseline = pd.concat([pd.read_parquet(f) for f in baselines], ignore_index=True)
    scenarios = pd.concat([pd.read_csv(f) for f in (OUT/"routing").glob("*_summary.csv")], ignore_index=True)
    registry = pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
    ports = pd.read_csv(NET/"canonical_port_registry.csv").set_index("key")
    gate_xyz = xyz(registry[["lon", "lat"]].to_numpy())
    endpoint_masks = np.zeros(len(baseline), np.uint32)
    for side in ["origin", "dest"]:
        xy = ports.loc[baseline[f"{side}_port_id"], ["water_lon", "water_lat"]].to_numpy()
        vectors = xyz(xy)
        assert np.isfinite(vectors).all() and np.isfinite(gate_xyz).all()
        dot = np.einsum("ij,kj->ik", vectors, gate_xyz, optimize=False)
        distances = np.arccos(np.clip(dot, -1, 1))*EARTH_KM
        for i in range(24):
            endpoint_masks[distances[:,i] <= 100] |= np.uint32(1 << i)
    baseline["endpoint_closure_mask"] = endpoint_masks
    baseline.to_parquet(OUT/"baseline_all.parquet", index=False)
    scenarios.to_csv(OUT/"single_closure_results.csv", index=False)
    total = baseline.groupby("shiptype").activity_weight.sum()
    records = []
    for s in SHIPS:
        d = baseline[baseline.shiptype == s]
        for gate in registry.itertuples():
            hit = (d.baseline_mask.to_numpy(np.uint32) & np.uint32(1 << (gate.number-1))) != 0
            records.append(dict(shiptype=s, gate=gate.key, number=gate.number, name=gate.name,
                exposed_weight=float(d.activity_weight.to_numpy()[hit].sum()), denominator=float(total[s]),
                exposure_pct=100*float(d.activity_weight.to_numpy()[hit].sum())/float(total[s]), exposed_rows=int(hit.sum())))
    pd.DataFrame(records).to_csv(OUT/"panel_b_exposure.csv", index=False)
    countries = []
    for side in ["origin", "dest"]:
        for country, d in baseline.groupby(f"{side}_iso3"):
            denom = float(d.activity_weight.sum())
            for gate in registry.itertuples():
                hit = (d.baseline_mask.to_numpy(np.uint32) & np.uint32(1 << (gate.number-1))) != 0
                countries.append(dict(country=country, side=side, gate=gate.key, number=gate.number,
                    exposure_pct=100*float(d.activity_weight.to_numpy()[hit].sum())/denom,
                    exposed_weight=float(d.activity_weight.to_numpy()[hit].sum()), denominator=denom))
    pd.DataFrame(countries).to_csv(OUT/"appendix_country_exposure.csv", index=False)
    monthly = baseline.groupby(["shiptype", "start_month"]).apply(lambda g: pd.Series({
        "rows": len(g), "observed_rows": int(g.has_observed.sum()), "weight": g.activity_weight.sum(),
        "observed_pct": 100*g.loc[g.has_observed, "activity_weight"].sum()/g.activity_weight.sum()}), include_groups=False).reset_index()
    monthly.to_csv(OUT/"appendix_monthly_coverage.csv", index=False)
    flows = []
    for f in (OUT/"routing").glob("*_solutions.parquet"):
        rows = pd.read_parquet(f)
        rows = rows[(rows.scenario_index > 0) & rows.affected].merge(
            baseline[["od_id", "activity_weight", "shiptype", "start_month", "endpoint_closure_mask"]], on="od_id", validate="many_to_one")
        rows["port_in_closure_zone"] = ((rows.endpoint_closure_mask.to_numpy(np.uint32)
            & np.left_shift(np.uint32(1), rows.scenario_index.to_numpy(np.uint32)-1)) != 0)
        missing = np.where(rows.port_in_closure_zone, "Port in closure zone", "No mapped network bypass")
        rows["without_state"] = np.where(np.isfinite(rows.without_km), "Conventional bypass", missing)
        rows["with_state"] = np.where(rows.route_code > 0, "Arctic distance option",
            np.where(np.isfinite(rows.with_km), "Conventional bypass", missing))
        flows.append(rows)
    affected = pd.concat(flows, ignore_index=True)
    affected.to_parquet(OUT/"affected_counterfactuals.parquet", index=False)
    agg = affected.groupby(["without_state", "with_state"]).activity_weight.sum().reset_index()
    agg["percent"] = 100*agg.activity_weight/agg.activity_weight.sum()
    agg.to_csv(OUT/"panel_c_flow.csv", index=False)
    checks = dict(od_rows=len(baseline), unique_observed_ports=len(set(baseline.origin_port_id)|set(baseline.dest_port_id)),
        weight_preserved=bool(np.isclose(baseline.activity_weight.sum(), pd.read_parquet(NET/"monthly_od_repaired.parquet").activity_weight.sum())),
        unique_od_ids=bool(baseline.od_id.is_unique), rows_with_finite_baseline=int(np.isfinite(baseline.baseline_km).sum()),
        independent_single_closures=24, ship_month_groups=48, total_scenario_groups=len(scenarios),
        paired_flow_sum_pct=float(agg.percent.sum()),
        arctic_never_lengthens=bool((affected.with_km <= affected.without_km+1e-4).all()),
        counterfactual_scope="24 independent single closures plus open control; not an updated pair/triple-closure experiment",
        outcome="Geometric distance opportunity under a common hypothetical ice-capable vessel screen, not cost-optimal or certified shipping",
        uncertainty="One SSP585 decade, six models x ten years. >=30/60 whole-route support; missing alternatives are not proof of infeasibility",
        inland_controls="Public lock/river links retain static connectivity; draft, bridge, lock capacity, policy and freshwater ice are unqualified. No operational cargo inference.")
    assert checks["weight_preserved"] and checks["unique_od_ids"] and checks["arctic_never_lengthens"]
    (OUT/"calculation_audit.json").write_text(json.dumps(checks, indent=2))
    print("COLLECTED", checks, flush=True)


def ais_raster_points():
    destination = OUT/"selected_AIS_map_points.parquet"
    if destination.exists():
        return
    d = pd.read_parquet(OUT/"demand_ais_first.parquet")
    d = d[d.has_observed]
    chunks = []
    point_count = 0
    with zipfile.ZipFile(ROOT/"data/ais_representative_trajectories_2024_ethan/source.zip") as z:
        for k, member in enumerate(d.member):
            p = np.loadtxt(io.BytesIO(z.read(member)), delimiter=",", skiprows=1, usecols=(2,3), ndmin=2)
            valid = np.isfinite(p).all(axis=1) & (np.abs(p[:,0])<=180) & (np.abs(p[:,1])<=90)
            p = p[valid]
            point_count += len(p)
            xy = np.floor((p+[180,90])/.0125).astype(np.int64)
            code = xy[:,1]*28800+xy[:,0]
            _, idx = np.unique(code, return_index=True)
            chunks.append(np.c_[p[idx], code[idx]])
            if k % 5000 == 0:
                print("AIS MAP", k, "of", len(d), "raw points", point_count, flush=True)
    points = np.vstack(chunks)
    _, idx = np.unique(points[:,2], return_index=True)
    p = points[idx,:2]
    pd.DataFrame(p, columns=["lon", "lat"]).to_parquet(destination, index=False)
    (OUT/"ais_map_audit.json").write_text(json.dumps(dict(members=len(d), raw_points=point_count,
        display_samples=len(p), display_dedup_degrees=.0125,
        no_synthetic_interpolation=True, source_archive_sha256=sha(ROOT/"data/ais_representative_trajectories_2024_ethan/source.zip")), indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "route", "collect", "ais-map"])
    p.add_argument("--ship", choices=SHIPS)
    p.add_argument("--month", type=int)
    a = p.parse_args()
    if a.stage == "prepare":
        prepare()
    elif a.stage == "route":
        for ship in ([a.ship] if a.ship else SHIPS):
            for month in ([a.month] if a.month else range(1,13)):
                group_run(ship, month)
    elif a.stage == "collect":
        collect()
    else:
        ais_raster_points()


if __name__ == "__main__":
    main()
