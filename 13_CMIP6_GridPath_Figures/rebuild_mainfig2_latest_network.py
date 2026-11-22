"""Matched physical routing with explicit monthly Arctic cores, Figure 2.

No published detour distances or assumed dollar tariffs enter this calculation.
The premium analysis is a break-even sensitivity, not a calibrated cost forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from chokepoints24 import Exposure
from physical_ocean_routing import length
from prepare_physical_monthly_ice import bit_count
from recompute_physical_arctic_embedding import PhysicalRouter, core_ice_masks
from repair_cmip6_monthly_routes import repaired_route_metadata

ROOT = Path(__file__).resolve().parent
NET = ROOT / "data/physical_arctic_embedding/oxmartrans_1390"
OLD = ROOT / "25_MainFig1_LatestNetwork"
OUT = ROOT / "26_MainFig2_LatestNetwork"
SHIPS = ["container", "bulk", "tanker", "lng"]
ROUTES = ["NWP", "NEP", "NSR", "TSR"]
VERSION = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Geometry:
    def __init__(self, network_dir=NET):
        self.edges = pd.read_parquet(network_dir / "physical_edges.parquet")
        data = np.load(network_dir / "edge_geometry.npz")
        self.xy, self.offsets = data["geometry"], data["offsets"]
        self.nodes = np.load(network_dir / "node_coordinates.npy")
        self.lookup = {(r.a, r.b): r for r in self.edges.itertuples()}

    def path(self, nodes):
        if len(nodes) == 1:
            return self.nodes[nodes].copy()
        pieces = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            edge = self.lookup[(min(a, b), max(a, b))]
            p = self.xy[self.offsets[edge.geometry_id]:self.offsets[edge.geometry_id + 1]]
            if a > b:
                p = p[::-1]
            if pieces:
                assert length([pieces[-1][-1], p[0]]) < .001
                p = p[1:]
            pieces.append(p)
        return np.vstack(pieces) if pieces else np.empty((0, 2))

    def qualification_flags(self, nodes):
        sources = {self.lookup[(min(a,b), max(a,b))].source for a,b in zip(nodes[:-1],nodes[1:])}
        return any(any(word in s for word in ("inland", "Caspian", "Great_Lakes", "Seaway", "USACE")) for s in sources)


def simple(p):
    p = p[np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]].copy()
    p[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(p[:, 0])))
    return bool(shapely.is_simple(shapely.LineString(p)))


def run_group(args):
    ship, month = args
    started = time.monotonic()
    prefix = OUT / "routing" / f"{ship}_{month:02d}"
    marker = Path(str(prefix) + "_audit.json")
    signature = sha(OUT / "input_manifest.json")
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature:
        return json.loads(marker.read_text())
    old_prefix = OLD / "routing" / f"{ship}_{month:02d}"
    base = pd.read_parquet(str(old_prefix) + "_baseline.parquet")
    saved = pd.read_parquet(str(old_prefix) + "_solutions.parquet")
    conventional = np.load(str(old_prefix) + "_conventional_paths.npz")
    endpoints = pd.read_csv(NET / "canonical_endpoint_access.csv").set_index("key")
    geo = Geometry()
    ice = np.load(OLD / "climate/edge_monthly_ice.npz")["model_year_mask"][month-1]
    edges = geo.edges[bit_count(ice[geo.edges.geometry_id.to_numpy()]) >= 30].reset_index(drop=True)
    router = PhysicalRouter(edges, geo.nodes, ice, 100)
    ps = router.to_compact[endpoints.loc[base.origin_port_id, "node"].to_numpy(int)]
    pt = router.to_compact[endpoints.loc[base.dest_port_id, "node"].to_numpy(int)]
    valid = (ps >= 0) & (pt >= 0)
    unique = np.unique(np.r_[ps[ps >= 0], pt[pt >= 0]])
    slots = {x: j for j, x in enumerate(unique)}
    ss, tt = np.array([slots.get(x, 0) for x in ps]), np.array([slots.get(x, 0) for x in pt])
    points, metadata = repaired_route_metadata()
    gateways = pd.read_csv(NET / "gateway_mapping.csv")
    classifier = Exposure(pd.read_csv(ROOT / "data/chokepoints24/registry.csv"))
    masks = core_ice_masks()
    gates, cores = {}, {}
    for route in ROUTES:
        if (route, month) not in metadata:
            continue
        rows = gateways[(gateways.route == route) & (gateways.month == month)].set_index("side")
        pair = []
        for side in ("a", "b"):
            key = rows.loc[side, "key"]
            node = int(router.to_compact[int(endpoints.loc[key, "node"])])
            if node < 0:
                break
            if key not in gates:
                gates[key] = dict(node=node, offset=len(gates)*len(unique))
            pair.append(gates[key])
        if len(pair) == 2:
            p = points[(points.route == route) & (points.month == month)].sort_values("order")[["lon", "lat"]].to_numpy(float)
            cores[route] = dict(pair=pair, points=p, km=length(p), ice=masks[(route, month)], mask=classifier.masks(p)[100])
    sources = np.concatenate([np.full(len(unique), g["node"]) for g in gates.values()]).astype(int) if gates else np.array([], int)
    targets = np.tile(unique, len(gates))
    opened = router.solve(sources, targets, np.uint32(0)) if gates else np.array([], int)

    @lru_cache(maxsize=512)
    def geometry(pid):
        return geo.path(router.active[router.paths[pid]])

    @lru_cache(maxsize=8192)
    def geometry_check(p, q, route, orientation):
        if set(router.paths[p]) & set(router.paths[q]):
            return False
        left, right = geometry(p)[::-1], geometry(q)
        core = cores[route]["points"] if orientation == 1 else cores[route]["points"][::-1]
        if not len(left) or not len(right):
            return False
        assert max(length([left[-1], core[0]]), length([core[-1], right[0]])) < .001
        full = np.vstack([left, core[1:], right[1:]])
        return simple(full)

    frames, rejects = [], dict(joint_ice=0, geometric_loop=0)
    for si in range(25):
        old = saved[saved.scenario_index == si].set_index("od_id").loc[base.od_id].reset_index()
        # Both counterfactuals optimize on the same physical backbone. The raw
        # AIS distance remains an audit column, not an inflated causal baseline.
        old_pid = old.conventional_path_id.to_numpy(int)
        without = conventional["physical_km"][old_pid]
        use = np.ones(len(base), bool) if si == 0 else old.affected.to_numpy(bool)
        if not use.any():
            continue
        closed = np.uint32(0 if si == 0 else 1 << (si-1))
        solved = opened if si == 0 else (router.solve(sources, targets, closed, opened[None, :]) if gates else opened)
        lengths, path_ice = np.asarray(router.length), np.asarray(router.path_ice, np.uint64)
        n = len(base)
        frame = pd.DataFrame(dict(od_id=base.od_id, scenario_index=si, affected=old.affected,
            without_km=without, conventional_path_id=old_pid,
            observed_or_fallback_baseline_km=old.baseline_km,
            shiptype=ship, month=month, activity_weight=base.activity_weight))
        for route in ROUTES:
            distance = np.full(n, np.inf)
            entry, exit_, orientation = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
            joint_best = np.zeros(n, float)
            if route in cores and not cores[route]["mask"] & int(closed):
                c = cores[route]
                for ori, (left, right) in enumerate((c["pair"], c["pair"][::-1]), 1):
                    p, q = solved[left["offset"]+ss], solved[right["offset"]+tt]
                    candidate = lengths[p]+c["km"]+lengths[q]
                    joint = bit_count(path_ice[p] & path_ice[q] & c["ice"])
                    potential = use & valid & np.isfinite(candidate) & (candidate < distance) & (candidate < without-1e-7)
                    rejects["joint_ice"] += int((potential & (joint < 30)).sum())
                    potential &= joint >= 30
                    for j in np.flatnonzero(potential):
                        if not geometry_check(int(p[j]), int(q[j]), route, ori):
                            potential[j] = False
                            rejects["geometric_loop"] += 1
                    distance[potential] = candidate[potential]
                    entry[potential], exit_[potential], orientation[potential] = p[potential], q[potential], ori
                    joint_best[potential] = joint[potential]/60
            frame[f"{route}_km"] = distance
            frame[f"{route}_entry"] = entry
            frame[f"{route}_exit"] = exit_
            frame[f"{route}_orientation"] = orientation
            frame[f"{route}_support"] = joint_best
        frames.append(frame.loc[use])
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(str(prefix) + "_candidates.parquet", index=False)
    router.save_catalogue(prefix)
    audit = dict(signature=signature, shiptype=ship, month=month, od_rows=len(base), output_rows=len(result),
                 candidate_rows_with_geometric_advantage=int(np.isfinite(result[[f"{r}_km" for r in ROUTES]]).any(axis=1).sum()),
                 available_cores=list(cores), rejected=rejects, elapsed_seconds=round(time.monotonic()-started, 2))
    marker.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)
    return audit


def summarize():
    data = pd.concat([pd.read_parquet(p) for p in sorted((OUT/"routing").glob("*_candidates.parquet"))], ignore_index=True)
    assert not data.duplicated(["od_id", "scenario_index"]).any()
    baseline = pd.read_parquet(OLD / "baseline_all.parquet")
    keys = ["od_id", "origin_port_id", "dest_port_id", "origin_name", "dest_name", "origin_iso3", "dest_iso3", "has_observed", "endpoint_closure_mask"]
    data = data.merge(baseline[keys], on="od_id", validate="many_to_one")
    masks = np.where(data.scenario_index > 0, np.left_shift(np.uint32(1), np.maximum(data.scenario_index.to_numpy()-1, 0)), 0)
    data["port_in_closure_zone"] = (data.endpoint_closure_mask.astype(np.uint32).to_numpy() & masks) != 0
    dist = data[[f"{r}_km" for r in ROUTES]].to_numpy()
    best = dist.min(axis=1)
    select = np.isfinite(best) & (best < data.without_km.to_numpy()-1e-7)
    data["arctic_km"] = np.where(select, best, np.nan)
    data["with_km"] = np.minimum(best, data.without_km)
    data["selected_route"] = np.where(select, np.asarray(ROUTES)[dist.argmin(axis=1)], "none")
    data["saved_km"] = np.where(select & np.isfinite(data.without_km), data.without_km - best, 0)
    data["break_even_premium"] = np.where(select, data.without_km/best - 1, 0)
    op = data[data.scenario_index == 0].set_index("od_id")
    data["open_selected_route"] = data.od_id.map(op.selected_route)
    data["open_break_even_premium"] = data.od_id.map(op.break_even_premium)
    data["incremental_selection"] = select & data.open_selected_route.eq("none")
    trade = pd.read_parquet(OLD / "panel_c_trade_weighted_od.parquet")
    if "od_id" not in trade:
        trade = trade.merge(baseline[["od_id", "shiptype", "start_month", "origin_port_id", "dest_port_id"]],
                            on=["shiptype", "start_month", "origin_port_id", "dest_port_id"], validate="one_to_one")
    trade = trade[trade.selection_status.eq("included")][["od_id", "trade_weight_usd"]]
    data = data.merge(trade, on="od_id", how="left", validate="many_to_one")
    data.to_parquet(OUT / "all_candidates_and_choices.parquet", index=False)
    # The main estimand is conditional on exposed OD-scenario observations.
    # Each of 24 closures is a separate stress test, not a probability-weighted forecast.
    exposed = data[data.scenario_index > 0].copy()
    monthly, sensitivity, mix, scenarios = [], [], [], []
    for (ship, month), g in exposed.groupby(["shiptype", "month"]):
        w = g.activity_weight
        selected = g.selected_route.ne("none")
        monthly.append(dict(shiptype=ship, month=month, denominator=float(w.sum()), rows=len(g),
            selected_weight=float(w[selected].sum()), selected_pct=100*w[selected].sum()/w.sum(),
            incremental_pct=100*w[g.incremental_selection].sum()/w.sum(),
            unresolved_pct=100*w[~np.isfinite(g.with_km)].sum()/w.sum()))
    for weight, col in (("activity", "activity_weight"), ("2024_BACI_x_2021_sea_share_subset", "trade_weight_usd")):
        for ship, g in exposed.groupby("shiptype"):
            g = g[g[col].notna()]
            w = g[col].to_numpy(float)
            total = w.sum()
            for premium in np.linspace(0, 1, 101):
                selected = (g.break_even_premium.to_numpy() > premium+1e-9)
                new = selected & (g.open_break_even_premium.to_numpy() <= premium+1e-9)
                sensitivity.append(dict(weight=weight, shiptype=ship, extra_unit_distance_cost_pct=100*premium,
                    denominator=float(total), selected_pct=100*w[selected].sum()/total,
                    incremental_pct=100*w[new].sum()/total))
            for route in ROUTES:
                value = g.loc[g.selected_route.eq(route), col].sum()
                mix.append(dict(weight=weight, shiptype=ship, route=route, selected_weight=float(value),
                    exposed_weight=float(total), pct_of_exposed=100*value/total,
                    pct_of_selected=100*value/g.loc[g.selected_route.ne("none"), col].sum() if g.selected_route.ne("none").any() else 0))
    registry = pd.read_csv(ROOT / "data/chokepoints24/registry.csv")
    for si, g in exposed.groupby("scenario_index"):
        w = g.activity_weight
        scenarios.append(dict(scenario_index=int(si), chokepoint=registry.iloc[int(si)-1]["name"],
            exposed_weight=float(w.sum()), selected_pct=100*w[g.selected_route.ne("none")].sum()/w.sum(),
            incremental_pct=100*w[g.incremental_selection].sum()/w.sum(),
            endpoint_in_closure_zone_pct=100*w[g.port_in_closure_zone].sum()/w.sum()))
    pd.DataFrame(monthly).to_csv(OUT/"panel_c_ship_month.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/"panel_d_break_even_sensitivity.csv", index=False)
    pd.DataFrame(mix).to_csv(OUT/"panel_e_route_composition.csv", index=False)
    pd.DataFrame(scenarios).to_csv(OUT/"closure_summary.csv", index=False)
    feasibility = pd.read_csv(ROOT/"data/cmip6_monthly_repair/model_year_month_feasibility.csv")
    core = feasibility.groupby(["route", "month"]).agg(n_model_years=("model", "size"),
        common_path_passable=("common_path_passable", "sum"), any_path_exists=("network_path_exists", "sum")).reset_index()
    core["support_pct"] = 100*core.common_path_passable/core.n_model_years
    core.to_csv(OUT/"panel_b_monthly_core_support.csv", index=False)
    ww = exposed.activity_weight
    report = dict(version=VERSION, status="computed_pending_map_geometry_validation", od_type_month_rows=len(op),
        physical_ports=int(pd.read_csv(NET/"canonical_port_registry.csv").shape[0]),
        represented_ports=len(set(baseline.origin_port_id)|set(baseline.dest_port_id)),
        closure_scenarios=24, exposed_od_scenario_rows=len(exposed),
        selected_exposed_activity_pct=float(100*ww[exposed.selected_route.ne("none")].sum()/ww.sum()),
        incremental_exposed_activity_pct=float(100*ww[exposed.incremental_selection].sum()/ww.sum()),
        no_sea_bypass_pct=float(100*ww[~np.isfinite(exposed.with_km)].sum()/ww.sum()),
        model="matched physical shortest paths; mandatory monthly core; common 30/60 model-year screen; complete geometric simplicity",
        primary_weight="capacity-normalized call activity index, not trade or tonnage",
        trade_sensitivity="2024 BACI times 2021 sea share; uncovered shares median-imputed as in current Fig1; eligible subset only",
        cost="extra unit-distance generalized cost; no empirical dollar claim, bootstrap or Monte Carlo claim",
        limitations=["PC6-like hypothetical criterion, not actual ice-class certification by vessel type",
            "Single scenario SSP585, 2091-2100; not a multi-SSP forecast", "24 independent 100 km point-buffer closures, not official polygons",
            "Inland/lock capacity and vessel draft are not certified", "Shortest admissible candidate search can miss alternate ice-feasible access paths"])
    (OUT/"calculation_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    (OUT/"routing").mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        source_files = [NET/"physical_edges.parquet", NET/"edge_geometry.npz", NET/"canonical_endpoint_access.csv",
            NET/"gateway_mapping.csv", OLD/"baseline_all.parquet", OLD/"input_manifest.json",
            OLD/"climate/edge_monthly_ice.npz", ROOT/"data/cmip6_monthly_repair/monthly_four_route_paths.csv",
            ROOT/"data/cmip6_monthly_repair/model_year_month_feasibility.csv", ROOT/"data/chokepoints24/registry.csv"]
        manifest = dict(version=VERSION, inputs={str(p.relative_to(ROOT)): sha(p) for p in source_files},
                        script_sha256=sha(__file__))
        (OUT/"input_manifest.json").write_text(json.dumps(manifest, indent=2))
        tasks = [(s,m) for s in SHIPS for m in range(1,13)]
        with get_context("spawn").Pool(args.workers) as pool:
            list(pool.imap_unordered(run_group, tasks))
    summarize()


if __name__ == "__main__":
    main()
