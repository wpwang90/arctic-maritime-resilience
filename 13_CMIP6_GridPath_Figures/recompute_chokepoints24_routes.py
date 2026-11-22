"""Re-solve AIS-supported routes after 1/2/3 of 24 chokepoints close.

No published rerouting distances enter this solver. A shortest path from a
less restricted parent remains optimal if it avoids the added closures;
otherwise Dijkstra is run again on the newly restricted graph. This exact
reuse rule makes the exhaustive experiment tractable without fixed detours.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from chokepoints24 import EARTH_KM, HERE, OUT, KEYS, Exposure, build_registry, xyz
from analyze_ethan_trajectory_arctic_embedding import (
    SHIP_ICE_FACTOR, ROUTE_RISK_PREMIUM, ROUTE_ESCORT_TOLL, ROUTES,
    haversine_vec, load_graph, margin_lookup, node_lonlat,
)
from repair_cmip6_monthly_routes import repaired_route_metadata as route_month_metadata

ROUTING = OUT / "routing"
ENGINE_VERSION = 3
COMBOS = [()] + [c for k in (1, 2, 3) for c in itertools.combinations(range(24), k)]
MASKS = np.asarray([sum(1 << i for i in c) for c in COMBOS], dtype=np.uint32)
SCENARIOS = ["open" if not c else "+".join(KEYS[i] for i in c) for c in COMBOS]
COMBO_INDEX = {c: i for i, c in enumerate(COMBOS)}
PARENTS = [[COMBO_INDEX[c[:j] + c[j + 1:]] for j in range(len(c))] for c in COMBOS]


def edge_exposure_masks(p1, p2, classifier, radius):
    """Sample every <=5 km along minor great-circle graph edges."""
    u, v = xyz(p1), xyz(p2)
    chord = np.linalg.norm(v - u, axis=1)
    lengths = 2 * EARTH_KM * np.arcsin(np.minimum(chord / 2, 1))
    steps = np.maximum(1, np.ceil(lengths / 5).astype(int))
    output = np.zeros(len(u), dtype=np.uint32)
    for k in range(int(steps.max()) + 1):
        ids = np.flatnonzero(steps >= k)
        f = k / steps[ids, None]
        points = (1 - f) * u[ids] + f * v[ids]
        points /= np.linalg.norm(points, axis=1)[:, None]
        c, gate = classifier.tree.query(points)
        distance = 2 * EARTH_KM * np.arcsin(np.minimum(c / 2, 1))
        hit = distance <= radius
        output[ids[hit]] |= np.left_shift(np.uint32(1), gate[hit].astype(np.uint32))
    return output


class ClosedRouter:
    def __init__(self, graph, classifier, radius):
        self.active = graph.active
        self.to_compact = np.full(64800, -1, dtype=np.int32)
        self.to_compact[self.active] = np.arange(len(self.active))
        self.xy = np.asarray([node_lonlat(n) for n in self.active])
        coo = graph.graph.tocoo()
        undirected = coo.row < coo.col
        self.row = self.to_compact[coo.row[undirected]]
        self.col = self.to_compact[coo.col[undirected]]
        self.impedance = coo.data[undirected]
        p1, p2 = self.xy[self.row], self.xy[self.col]
        self.km = haversine_vec(p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1])
        self.edge_mask = edge_exposure_masks(p1, p2, classifier, radius)
        self.edge_lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.row, self.col))}
        self.paths = [np.empty(0, dtype=np.int32)]
        self.cost = [np.inf]
        self.length = [np.inf]
        self.path_mask = [0]
        self.path_lookup = {}
        self.dijkstra_trees = 0
        self.new_paths = 0

    def restricted_graph(self, closed):
        keep = (self.edge_mask & closed) == 0
        return coo_matrix((np.r_[self.impedance[keep], self.impedance[keep]],
            (np.r_[self.row[keep], self.col[keep]], np.r_[self.col[keep], self.row[keep]])),
            shape=(len(self.active), len(self.active))).tocsr()

    def store_path(self, nodes, cost):
        key = tuple(nodes)
        if key in self.path_lookup:
            return self.path_lookup[key]
        indices = [self.edge_lookup[(min(a, b), max(a, b))] for a, b in zip(nodes[:-1], nodes[1:])]
        mask = int(np.bitwise_or.reduce(self.edge_mask[indices], initial=np.uint32(0)))
        pid = len(self.paths)
        self.paths.append(np.asarray(nodes, dtype=np.int32))
        self.cost.append(float(cost))
        self.length.append(float(self.km[indices].sum()))
        self.path_mask.append(mask)
        self.path_lookup[key] = pid
        self.new_paths += 1
        return pid

    def solve(self, sources, targets, closed, parent_ids=None):
        if parent_ids is None:
            answer = np.zeros(len(sources), dtype=np.int32)
            unresolved = np.ones(len(sources), dtype=bool)
        else:
            masks = np.asarray(self.path_mask, dtype=np.uint32)[parent_ids]
            usable = (parent_ids > 0) & ((masks & closed) == 0)
            first = np.argmax(usable, axis=0)
            answer = parent_ids[first, np.arange(len(sources))].copy()
            valid = usable.any(axis=0)
            answer[~valid] = 0
            # Deleting more edges cannot reconnect a disconnected parent.
            unresolved = ~valid & ~(parent_ids == 0).any(axis=0)
        if unresolved.any():
            restricted = self.restricted_graph(closed)
            for source in np.unique(sources[unresolved]):
                ids = np.flatnonzero(unresolved & (sources == source))
                distances, pred = dijkstra(restricted, directed=False, indices=int(source), return_predecessors=True)
                self.dijkstra_trees += 1
                for i in ids:
                    target = int(targets[i])
                    if not np.isfinite(distances[target]):
                        continue
                    path = [target]
                    while path[-1] != source:
                        predecessor = int(pred[path[-1]])
                        if predecessor < 0:
                            raise AssertionError("Broken predecessor chain")
                        path.append(predecessor)
                    path.reverse()
                    answer[i] = self.store_path(path, distances[target])
        assert not np.any(np.asarray(self.path_mask, dtype=np.uint32)[answer] & closed)
        return answer

    def save_catalogue(self, prefix):
        offsets = np.r_[0, np.cumsum([len(p) for p in self.paths])]
        nodes = np.concatenate(self.paths)
        np.savez_compressed(str(prefix) + "_paths.npz", offsets=offsets,
            grid_nodes=self.active[nodes], modal_impedance_km=np.asarray(self.cost),
            physical_km=np.asarray(self.length), gate_mask=np.asarray(self.path_mask, dtype=np.uint32))


def prepare_od():
    raw = pd.read_parquet(HERE / "data/ethan_representative_trajectory_summary.parquet")
    # Country display preferences must not remove the physical transit network.
    raw["origin_port_id"] = raw.origin_iso3.fillna("UNK") + ":" + raw.origin_label_port_key.fillna(raw.origin_key)
    raw["dest_port_id"] = raw.dest_iso3.fillna("UNK") + ":" + raw.dest_label_port_key.fillna(raw.dest_key)
    raw = raw[raw.origin_port_id.ne(raw.dest_port_id)].copy()
    port_samples = []
    for side in ("origin", "dest"):
        sample = pd.DataFrame({"port_id": raw[f"{side}_port_id"],
            "lon": raw[f"{side}_label_lon"].fillna(raw[f"{side}_port_lon"]),
            "lat": raw[f"{side}_label_lat"].fillna(raw[f"{side}_port_lat"]),
            "label_coordinate": raw[f"{side}_label_lon"].notna() & raw[f"{side}_label_lat"].notna()})
        port_samples.append(sample)
    port_samples = pd.concat(port_samples, ignore_index=True).dropna(subset=["lon", "lat"])
    locations = []
    for port, group in port_samples.groupby("port_id"):
        labelled = group[group.label_coordinate]
        use = labelled if not labelled.empty else group
        coords = use[["lon", "lat"]].to_numpy(float)
        vectors = xyz(coords)
        centre = np.median(vectors, axis=0)
        centre /= np.linalg.norm(centre)
        chosen = coords[np.argmin(np.linalg.norm(vectors - centre, axis=1))]
        distance = haversine_vec(group.lon.to_numpy(), group.lat.to_numpy(), chosen[0], chosen[1])
        locations.append(dict(port_id=port, lon=chosen[0], lat=chosen[1], coordinate_samples=len(group),
            source="matched_port_label" if len(labelled) else "robust_representative_endpoint_not_independently_geocoded",
            p95_coordinate_deviation_km=float(np.quantile(distance, .95)),
            fraction_over_100km=float(np.mean(distance > 100))))
    locations = pd.DataFrame(locations).set_index("port_id")
    locations.to_csv(OUT / "canonical_port_locations.csv")
    for side in ("origin", "dest"):
        raw[f"{side}_port_lon"] = raw[f"{side}_port_id"].map(locations.lon)
        raw[f"{side}_port_lat"] = raw[f"{side}_port_id"].map(locations.lat)
    columns = ["origin_port_lon", "origin_port_lat", "dest_port_lon", "dest_port_lat"]
    valid = np.isfinite(raw[columns]).all(axis=1)
    excluded = raw.loc[~valid, ["member", "shiptype", "start_month", *columns]]
    excluded.to_csv(OUT / "excluded_missing_port_coordinates.csv", index=False)
    od = raw.loc[valid].groupby(["shiptype", "start_month", "origin_port_id", "dest_port_id"], as_index=False).agg(
        activity_weight=("activity_weight", "sum"), call_count=("call_count", "sum"), member_count=("member", "size"),
        origin_name=("origin_label_port_name", "first"), dest_name=("dest_label_port_name", "first"),
        origin_iso3=("origin_iso3", "first"), dest_iso3=("dest_iso3", "first"),
        origin_lon=("origin_port_lon", "first"), origin_lat=("origin_port_lat", "first"),
        dest_lon=("dest_port_lon", "first"), dest_lat=("dest_port_lat", "first"),
        observed_loop_erased_km=("loop_erased_distance_km", "median"),
    )
    od.to_parquet(OUT / "monthly_od.parquet", index=False)
    return od


def network_metrics(node_a, node_b, mask, countries):
    n = len(countries)
    graph = coo_matrix((np.ones(int(mask.sum()), dtype=np.int8), (node_a[mask], node_b[mask])), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False)
    size = np.bincount(labels)
    largest = int(np.argmax(size))
    return (int(size[largest]), len(set(countries[labels == largest]) - {"UNK"}),
        float(np.sum(size * (size - 1))) / max(n * (n - 1), 1))


def metric_row(od, weight, affected, without, with_available, selected, base_valid, src, dst, countries):
    total = float(weight.sum())
    a = network_metrics(src, dst, without, countries)
    b = network_metrics(src, dst, with_available, countries)
    restored = ~without & with_available & base_valid
    return dict(total_activity_weight=total, baseline_routable_weight=float(weight[base_valid].sum()),
        affected_activity_weight=float(weight[affected].sum()), affected_activity_pct=100 * weight[affected].sum() / total,
        without_routable_weight=float(weight[without].sum()), with_routable_weight=float(weight[with_available].sum()),
        arctic_selected_weight=float(weight[selected].sum()), restored_activity_weight=float(weight[restored].sum()),
        unresolved_without_weight=float(weight[base_valid & ~without].sum()),
        unresolved_with_weight=float(weight[base_valid & ~with_available].sum()),
        arctic_selected_pct=100 * weight[selected].sum() / total,
        A_lcc_ports=a[0], A_lcc_countries=a[1], A_reachability=a[2],
        AB_lcc_ports=b[0], AB_lcc_countries=b[1], AB_reachability=b[2],
        arctic_lcc_gain_ports=b[0] - a[0], arctic_reachability_gain=b[2] - a[2])


def run_group(args):
    shiptype, month, radius, max_closures = args
    prefix = ROUTING / f"{shiptype}_{month:02d}_r{radius}"
    marker = Path(str(prefix) + "_audit.json")
    if marker.exists():
        result = json.loads(marker.read_text())
        if result.get("max_closures") == max_closures and result.get("engine_version") == ENGINE_VERSION:
            return result
    od = pd.read_parquet(OUT / "monthly_od.parquet")
    od = od[(od.shiptype == shiptype) & (od.start_month == month)].reset_index(drop=True)
    graph = load_graph(shiptype, month)
    classifier = Exposure(pd.read_csv(OUT / "registry.csv"))
    router = ClosedRouter(graph, classifier, radius)
    n = len(od)
    margin_map, default_margin = margin_lookup()
    margins = np.asarray([margin_map.get((r.dest_iso3, r.origin_iso3), margin_map.get((r.origin_iso3, r.dest_iso3), default_margin)) for r in od.itertuples()])
    multiplier = 1 + margins / 100
    endpoint_cache = {}
    for side in ("origin", "dest"):
        for r in od.itertuples():
            port = getattr(r, f"{side}_port_id")
            if port not in endpoint_cache:
                lon, lat = getattr(r, f"{side}_lon"), getattr(r, f"{side}_lat")
                node, snap = graph.snap(lon, lat)
                cp_mask = classifier.masks(np.asarray([[lon, lat], node_lonlat(node)]))[radius]
                endpoint_cache[port] = (int(router.to_compact[node]), snap, cp_mask)
    a = np.asarray([endpoint_cache[x] for x in od.origin_port_id])
    b = np.asarray([endpoint_cache[x] for x in od.dest_port_id])
    main_sources, main_targets = a[:, 0].astype(int), b[:, 0].astype(int)
    terminal_km = a[:, 1] + b[:, 1]
    terminal_mask = a[:, 2].astype(np.uint32) | b[:, 2].astype(np.uint32)
    terminal_valid = (a[:, 1] <= 150) & (b[:, 1] <= 150)
    od["origin_snap_km"], od["dest_snap_km"] = a[:, 1], b[:, 1]
    od["origin_grid_node"] = router.active[main_sources]
    od["dest_grid_node"] = router.active[main_targets]
    od["terminal_gate_mask"] = terminal_mask
    paths, route_meta = route_month_metadata()
    available_meta = {route: route_meta[(route, month)] for route in ROUTES if (route, month) in route_meta and route_meta[(route, month)]["cmip6_continuous"]}
    unique_nodes = np.unique(np.r_[main_sources, main_targets])
    node_to_slot = {node: j for j, node in enumerate(unique_nodes)}
    gate_cache, route_gates = {}, {}
    for route, meta in available_meta.items():
        route_gates[route] = []
        for side in ("a", "b"):
            point = (meta[f"gateway_{side}_lon"], meta[f"gateway_{side}_lat"])
            if point not in gate_cache:
                node, snap = graph.snap(*point)
                gate_cache[point] = dict(node=int(router.to_compact[node]), snap=snap,
                    mask=classifier.masks(np.asarray([point, node_lonlat(node)]))[radius],
                    offset=n + len(gate_cache) * len(unique_nodes))
            route_gates[route].append(gate_cache[point])
        middle = paths[(paths.route == route) & (paths.month == month)].sort_values("order")
        meta["middle_mask"] = classifier.masks(middle[["lon", "lat"]].to_numpy(float))[radius]
    sources = np.concatenate([main_sources] + [np.full(len(unique_nodes), g["node"], dtype=int) for g in gate_cache.values()])
    targets = np.concatenate([main_targets] + [unique_nodes for _ in gate_cache])
    scenario_count = sum(len(c) <= max_closures for c in COMBOS)
    without_ids = np.zeros((scenario_count, n), dtype=np.int32)
    route_codes = np.zeros((scenario_count, n), dtype=np.int8)
    entry_ids = np.zeros((scenario_count, n), dtype=np.int32)
    exit_ids = np.zeros((scenario_count, n), dtype=np.int32)
    orientation_codes = np.zeros((scenario_count, n), dtype=np.int8)
    parent_cache = {}
    rows, singleton_details = [], []
    ports = pd.Index(sorted(set(od.origin_port_id) | set(od.dest_port_id)))
    source_port, target_port = ports.get_indexer(od.origin_port_id), ports.get_indexer(od.dest_port_id)
    countries = np.asarray([x.split(":")[0] for x in ports])
    weight = od.activity_weight.to_numpy(float)
    source_slots = np.asarray([node_to_slot[x] for x in main_sources])
    target_slots = np.asarray([node_to_slot[x] for x in main_targets])
    started = time.monotonic()
    for i in range(scenario_count):
        closed = MASKS[i]
        parents = np.stack([parent_cache[j] for j in PARENTS[i]]) if i else None
        answer = router.solve(sources, targets, closed, parents)
        if len(COMBOS[i]) <= 2:
            parent_cache[i] = answer
        cost = np.asarray(router.cost)
        length = np.asarray(router.length)
        masks = np.asarray(router.path_mask, dtype=np.uint32)
        valid_terminal = terminal_valid & ((terminal_mask & closed) == 0)
        without = answer[:n].copy()
        without[~valid_terminal] = 0
        conventional_cost = (cost[without] + terminal_km) * multiplier
        conventional_km = length[without] + terminal_km
        best_cost = conventional_cost.copy()
        best_km = conventional_km.copy()
        chosen = np.zeros(n, dtype=np.int8)
        for route, meta in available_meta.items():
            g1, g2 = route_gates[route]
            if max(g1["snap"], g2["snap"]) > 150 or (g1["mask"] | g2["mask"] | meta["middle_mask"]) & int(closed):
                continue
            ice = SHIP_ICE_FACTOR[shiptype] * (0.04 + 0.18 * (12 - meta["path_mean_months"]) / 12)
            factor = multiplier + ice + ROUTE_RISK_PREMIUM[route] + ROUTE_ESCORT_TOLL[route]
            for orientation, (left, right) in enumerate(((g1, g2), (g2, g1)), 1):
                left_ids = answer[left["offset"] + source_slots]
                right_ids = answer[right["offset"] + target_slots]
                connectors = terminal_km + left["snap"] + right["snap"]
                candidate_cost = (cost[left_ids] + cost[right_ids] + connectors + meta["arctic_middle_km"]) * factor
                improve = valid_terminal & (candidate_cost < best_cost - 1e-7)
                best_cost[improve] = candidate_cost[improve]
                best_km[improve] = (length[left_ids] + length[right_ids] + connectors + meta["arctic_middle_km"])[improve]
                chosen[improve] = ROUTES.index(route) + 1
                entry_ids[i, improve] = left_ids[improve]
                exit_ids[i, improve] = right_ids[improve]
                orientation_codes[i, improve] = orientation
        without_ids[i], route_codes[i] = without, chosen
        if not i:
            base_ids, base_valid = without.copy(), without > 0
            base_km, base_cost = conventional_km.copy(), conventional_cost.copy()
            base_mask = masks[base_ids] | terminal_mask
        affected = base_valid & ((base_mask & closed) != 0)
        selected = chosen > 0
        with_available = np.isfinite(best_cost)
        row = metric_row(od, weight, affected, without > 0, with_available, selected, base_valid, source_port, target_port, countries)
        row.update(shiptype=shiptype, month=month, buffer_km=radius, scenario=SCENARIOS[i], scenario_index=i, closure_count=len(COMBOS[i]))
        finite_pair = affected & (without > 0)
        row["distance_comparison_weight"] = float(weight[finite_pair].sum())
        row["weighted_added_distance_km"] = float(np.sum(weight[finite_pair] * (conventional_km[finite_pair] - base_km[finite_pair])))
        paired_arctic = selected & (without > 0)
        row["arctic_paired_comparison_weight"] = float(weight[paired_arctic].sum())
        row["weighted_arctic_distance_saving_km"] = float(np.sum(weight[paired_arctic] * (conventional_km[paired_arctic] - best_km[paired_arctic])))
        rows.append(row)
        if len(COMBOS[i]) == 1:
            detail = od.loc[affected].copy()
            detail["chokepoint"] = KEYS[COMBOS[i][0]]
            detail["baseline_km"] = base_km[affected]
            detail["closed_without_arctic_km"] = conventional_km[affected]
            detail["closed_with_arctic_km"] = best_km[affected]
            detail["added_without_arctic_km"] = conventional_km[affected] - base_km[affected]
            detail["with_route"] = np.asarray(["conventional", *ROUTES])[chosen[affected]]
            detail["without_path_id"] = without[affected]
            detail["arctic_entry_path_id"] = entry_ids[i, affected]
            detail["arctic_exit_path_id"] = exit_ids[i, affected]
            detail["without_observed_network_path"] = ~np.isfinite(conventional_km[affected])
            detail["with_observed_network_path"] = np.isfinite(best_km[affected])
            singleton_details.append(detail)
        if i and i % 300 == 0:
            print(f"{shiptype}-{month:02d}: {i}/{scenario_count - 1} closures, {router.dijkstra_trees} trees, {time.monotonic() - started:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table["strict_incremental"] = False
    for i in range(1, scenario_count):
        if len(COMBOS[i]) == 1:
            keep = table.at[i, "affected_activity_weight"] > 0
        else:
            threshold = max(0.0005 * weight.sum(), 0.005 * table.at[i, "affected_activity_weight"])
            keep = True
            for parent in PARENTS[i]:
                extra = table.at[i, "affected_activity_weight"] - table.at[parent, "affected_activity_weight"]
                changed = (without_ids[i] != without_ids[parent]) | (route_codes[i] != route_codes[parent]) | (entry_ids[i] != entry_ids[parent]) | (exit_ids[i] != exit_ids[parent])
                if extra <= threshold and weight[changed].sum() <= threshold:
                    keep = False
                    break
        table.at[i, "strict_incremental"] = keep
    table.to_csv(str(prefix) + "_scenarios.csv", index=False)
    pd.concat(singleton_details, ignore_index=True).to_parquet(str(prefix) + "_single_od.parquet", index=False)
    od.to_parquet(str(prefix) + "_od.parquet", index=False)
    np.savez_compressed(str(prefix) + "_choices.npz", without_path_ids=without_ids, with_route_codes=route_codes,
        arctic_entry_path_ids=entry_ids, arctic_exit_path_ids=exit_ids,
        arctic_orientation=orientation_codes, scenario_masks=MASKS[:scenario_count])
    router.save_catalogue(prefix)
    audit = dict(engine_version=ENGINE_VERSION, shiptype=shiptype, month=month, buffer_km=radius, od_count=n,
        scenarios=scenario_count - 1, max_closures=max_closures, dijkstra_trees=router.dijkstra_trees,
        distinct_paths=len(router.paths) - 1, elapsed_seconds=time.monotonic() - started,
        active_grid_nodes=len(router.active), strict_retained=int(table.strict_incremental.sum()),
        baseline_unroutable_od=int((~base_valid).sum()),
        baseline_arctic_selected_od=int((route_codes[0] > 0).sum()),
        cmip6_continuous_routes=list(available_meta),
        published_detour_used_by_solver=False, closure_reintersection_failures=0)
    marker.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"FINISHED {shiptype}-{month:02d}: {audit}", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--radius", type=int, choices=[75, 100, 150], default=100)
    parser.add_argument("--max-closures", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--shiptype", choices=["bulk", "container", "lng", "tanker"])
    parser.add_argument("--month", type=int)
    args = parser.parse_args()
    build_registry()
    ROUTING.mkdir(parents=True, exist_ok=True)
    od = prepare_od()
    groups = [(s, int(m), args.radius, args.max_closures) for (s, m), _ in od.groupby(["shiptype", "start_month"])
        if (not args.shiptype or s == args.shiptype) and (not args.month or m == args.month)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run_group, groups)
    print(json.dumps(dict(completed_groups=len(results), scenarios=sum(r["scenarios"] for r in results),
        distinct_paths=sum(r["distinct_paths"] for r in results), buffer_km=args.radius), indent=2))


if __name__ == "__main__":
    main()
