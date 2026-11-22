"""Physical, explicit maritime routes with a mandatory monthly Arctic core.

The same water-checked public/AIS backbone is used in both counterfactuals.
Distances from the literature and economic penalties do not enter Dijkstra.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from physical_ocean_routing import HERE, OUT, length
from prepare_physical_monthly_ice import ALL_YEARS, bit_count
from repair_cmip6_monthly_routes import MODELS, repaired_route_metadata
from recompute_chokepoints24_routes import ClosedRouter, COMBOS, MASKS, SCENARIOS, PARENTS, metric_row
from chokepoints24 import Exposure

ROUTES = ["NWP", "NEP", "NSR", "TSR"]
ROUTING = OUT / "routing"
VERSION = 5


class PhysicalRouter(ClosedRouter):
    def __init__(self, edges, coordinates, masks, radius):
        self.active = np.unique(edges[["a", "b"]].to_numpy())
        self.to_compact = np.full(len(coordinates), -1, dtype=np.int32)
        self.to_compact[self.active] = np.arange(len(self.active))
        self.xy = coordinates[self.active]
        self.row = self.to_compact[edges.a.to_numpy()]
        self.col = self.to_compact[edges.b.to_numpy()]
        self.impedance = self.km = edges.km.to_numpy()
        self.edge_mask = edges[f"mask_{radius}"].to_numpy(np.uint32)
        self.edge_ice = masks[edges.geometry_id.to_numpy()]
        self.edge_geometry = edges.geometry_id.to_numpy()
        self.edge_lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.row, self.col))}
        self.paths = [np.empty(0, dtype=np.int32)]
        self.cost = [np.inf]
        self.length = [np.inf]
        self.path_mask = [0]
        self.path_ice = [np.uint64(0)]
        self.path_lookup = {}
        self.dijkstra_trees = self.new_paths = 0

    def store_path(self, nodes, cost):
        previous = len(self.paths)
        pid = super().store_path(nodes, cost)
        if len(self.paths) > previous:
            ids = [self.edge_lookup[(min(a,b), max(a,b))] for a,b in zip(nodes[:-1], nodes[1:])]
            self.path_ice.append(np.bitwise_and.reduce(self.edge_ice[ids], initial=ALL_YEARS))
            assert len(set(nodes)) == len(nodes)
            assert abs(self.length[pid] - cost) < 1e-5
        return pid

    def save_catalogue(self, prefix):
        np.savez_compressed(str(prefix) + "_paths.npz",
            offsets=np.r_[0, np.cumsum([len(p) for p in self.paths])],
            nodes=self.active[np.concatenate(self.paths)], physical_km=np.asarray(self.length),
            gate_mask=np.asarray(self.path_mask, dtype=np.uint32),
            model_year_mask=np.asarray(self.path_ice, dtype=np.uint64))


def core_ice_masks():
    frame = pd.read_csv(HERE / "data/cmip6_monthly_repair/model_year_month_feasibility.csv")
    out = {}
    for (route, month), g in frame.groupby(["route", "month"]):
        value = 0
        for r in g[g.common_path_passable].itertuples():
            value |= 1 << (MODELS.index(r.model) * 10 + int(r.year) - 2091)
        out[(route, int(month))] = np.uint64(value)
    return out


def run_group(args):
    shiptype, month, radius, max_closures = args[:4]
    network_dir=Path(args[4]) if len(args)>4 else OUT
    return run_network_group(shiptype,month,radius,max_closures,network_dir)


def run_network_group(shiptype,month,radius,max_closures,OUT):
    ROUTING=OUT/"routing"
    ROUTING.mkdir(exist_ok=True)
    prefix = ROUTING / f"{shiptype}_{month:02d}_r{radius}"
    marker = Path(str(prefix) + "_audit.json")
    fingerprint = hashlib.sha256((OUT/"physical_edges.parquet").read_bytes()).hexdigest()
    geometry_fingerprint=hashlib.sha256((OUT/"edge_geometry.npz").read_bytes()).hexdigest()
    od_file = OUT / "monthly_od_repaired.parquet"
    if not od_file.exists():
        od_file = HERE / "data/chokepoints24/monthly_od.parquet"
    endpoint_file = OUT / "canonical_endpoint_access.csv"
    if not endpoint_file.exists():
        endpoint_file = OUT / "endpoint_access.csv"
    demand_fingerprint=hashlib.sha256(od_file.read_bytes()).hexdigest()
    endpoint_fingerprint=hashlib.sha256(endpoint_file.read_bytes()).hexdigest()
    climate_provenance=json.loads((OUT/"monthly_ice_provenance.json").read_text())
    if climate_provenance.get("geometry_sha256")!=geometry_fingerprint:
        raise RuntimeError("Sea-ice masks do not match the current geometry. Rerun prepare_physical_monthly_ice.py.")
    if marker.exists():
        audit = json.loads(marker.read_text())
        if (audit.get("version") == VERSION and audit.get("max_closures") == max_closures
            and audit.get("graph_sha256")==fingerprint and audit.get("geometry_sha256")==geometry_fingerprint
            and audit.get("demand_sha256")==demand_fingerprint and audit.get("endpoint_sha256")==endpoint_fingerprint):
            return audit
    started = time.monotonic()
    od = pd.read_parquet(od_file)
    od = od[(od.shiptype == shiptype) & (od.start_month == month)].reset_index(drop=True)
    edges = pd.read_parquet(OUT / "physical_edges.parquet")
    observed = pd.read_parquet(HERE / "data/ethan_representative_ais_grid_edges_1deg.parquet",
        filters=[("shiptype", "==", shiptype), ("month", "==", month)])
    mapping = pd.read_parquet(OUT / "ais_water_edges.parquet")
    active = observed.merge(mapping, on=["x1", "y1", "x2", "y2"])
    keys = set(zip(np.minimum(active.a,active.b), np.maximum(active.a,active.b)))
    keep = np.asarray([r.source != "AIS_supported_water_grid" or (r.a,r.b) in keys for r in edges.itertuples()])
    ice = np.load(OUT / "edge_monthly_ice.npz")["model_year_mask"][month-1]
    if len(ice) <= int(edges.geometry_id.max()):
        raise RuntimeError("New waterway geometry lacks CMIP6 sampling: rerun prepare_physical_monthly_ice.py first")
    keep &= bit_count(ice[edges.geometry_id.to_numpy()]) >= 30
    router = PhysicalRouter(edges.loc[keep].reset_index(drop=True), np.load(OUT / "node_coordinates.npy"), ice, radius)
    endpoints = pd.read_csv(endpoint_file).set_index("key")
    src = router.to_compact[endpoints.loc[od.origin_port_id, "node"].to_numpy()]
    dst = router.to_compact[endpoints.loc[od.dest_port_id, "node"].to_numpy()]
    valid_end = (src >= 0) & (dst >= 0)
    # Isolated climate-filtered endpoints use a dummy index only for array
    # bookkeeping. Their results are explicitly set to unavailable below.
    src_safe, dst_safe = np.maximum(src,0), np.maximum(dst,0)
    registry_ports=endpoints[endpoints.kind=="port"].copy()
    port_compact=router.to_compact[registry_ports.node.to_numpy(int)]
    # Registry-wide physical access does not imply observed demand at every port.
    unique = np.unique(np.r_[src_safe,dst_safe,port_compact[port_compact>=0]])
    slot = {int(x):j for j,x in enumerate(unique)}
    port_slot=np.asarray([slot.get(int(x),0) for x in port_compact])
    src_slot = np.asarray([slot[int(x)] for x in src_safe])
    dst_slot = np.asarray([slot[int(x)] for x in dst_safe])
    points, metadata = repaired_route_metadata()
    gateway_map = pd.read_csv(OUT / "gateway_mapping.csv")
    gates, meta = {}, {}
    classifier = Exposure(pd.read_csv(HERE / "data/chokepoints24/registry.csv"))
    cmasks = core_ice_masks()
    for route in ROUTES:
        if (route,month) not in metadata:
            continue
        rows = gateway_map[(gateway_map.route==route)&(gateway_map.month==month)].set_index("side")
        pair = []
        for side in ("a","b"):
            key = rows.loc[side,"key"]
            node = int(router.to_compact[int(endpoints.loc[key,"node"])])
            if node < 0:
                break
            if key not in gates:
                gates[key] = dict(node=node, offset=len(od)+len(gates)*len(unique))
            pair.append(gates[key])
        if len(pair) != 2:
            continue
        core = points[(points.route==route)&(points.month==month)].sort_values("order")[["lon","lat"]].to_numpy(float)
        meta[route] = dict(pair=pair, km=length(core), ice=cmasks[(route,month)], mask=classifier.masks(core)[radius])
    sources = np.concatenate([src_safe]+[np.full(len(unique),g["node"],dtype=int) for g in gates.values()])
    targets = np.concatenate([dst_safe]+[unique for _ in gates])
    n, count = len(od), sum(len(c)<=max_closures for c in COMBOS)
    without_ids = np.zeros((count,n),dtype=np.int32)
    codes = np.zeros((count,n),dtype=np.int8)
    entry_ids = np.zeros((count,n),dtype=np.int32)
    exit_ids = np.zeros((count,n),dtype=np.int32)
    orientations = np.zeros((count,n),dtype=np.int8)
    with_km = np.full((count,n),np.inf,dtype=np.float32)
    parents_cache, simple_cache = {}, {}
    rows, detail = [], []
    weight = od.activity_weight.to_numpy(float)
    ports = pd.Index(sorted(set(od.origin_port_id)|set(od.dest_port_id)))
    pa,pb = ports.get_indexer(od.origin_port_id),ports.get_indexer(od.dest_port_id)
    countries = np.asarray([x.split(":")[0] for x in ports])
    rejected_joint = rejected_loops = 0
    for si in range(count):
        closed = MASKS[si]
        parent = np.stack([parents_cache[j] for j in PARENTS[si]]) if si else None
        answer = router.solve(sources,targets,closed,parent)
        if len(COMBOS[si])<=2:
            parents_cache[si] = answer
        lengths = np.asarray(router.length)
        path_ice = np.asarray(router.path_ice,dtype=np.uint64)
        if si==0:
            access_tables=[]
            for route in ROUTES:
                for side in (0,1):
                    if route in meta:
                        m=meta[route]
                        gate=m["pair"][side]
                        pid=answer[gate["offset"]+port_slot].copy()
                        pid[port_compact<0]=0
                        support=bit_count(path_ice[pid]&m["ice"])/60
                        access=lengths[pid]
                        core_km=m["km"]
                    else:
                        pid=np.zeros(len(registry_ports),dtype=int)
                        access=np.full(len(registry_ports),np.inf)
                        support=np.zeros(len(registry_ports))
                        core_km=np.nan
                    access_tables.append(pd.DataFrame(dict(port_id=registry_ports.index,
                        shiptype=shiptype,month=month,route=route,entry_side=side,
                        access_path_id=pid,access_km=np.where(np.isfinite(access),access,np.nan),
                        core_km=core_km,access_plus_core_model_year_fraction=support,
                        entry_and_core_screen_pass=(pid>0)&(support>=.5),
                        coverage_note="one port plus core; not a complete port-pair voyage or observed monthly demand")))
            pd.concat(access_tables,ignore_index=True).to_parquet(str(prefix)+"_registry_arctic_access.parquet",index=False)
        without = answer[:n].copy()
        without[~valid_end] = 0
        # Pointwise ensemble feasibility is necessary but not sufficient for
        # a continuous voyage: intersect the same 60 model-year bit masks.
        joint_without = bit_count(path_ice[without])>=30
        without[~joint_without] = 0
        best = lengths[without].copy()
        for route,m in meta.items():
            if int(m["mask"]) & int(closed):
                continue
            ga,gb = m["pair"]
            for orientation,(left,right) in enumerate(((ga,gb),(gb,ga)),1):
                p = answer[left["offset"]+src_slot]
                q = answer[right["offset"]+dst_slot]
                candidate = lengths[p]+m["km"]+lengths[q]
                joint = bit_count(path_ice[p]&path_ice[q]&m["ice"])>=30
                potential = valid_end & np.isfinite(candidate) & (candidate<best-1e-7)
                rejected_joint += int((potential&~joint).sum())
                improve = potential & joint
                # Never erase a forced-core loop into an unrelated route.
                for j in np.flatnonzero(improve):
                    key = (int(p[j]),int(q[j]))
                    if key not in simple_cache:
                        simple_cache[key] = not bool(set(router.paths[key[0]])&set(router.paths[key[1]]))
                    if not simple_cache[key]:
                        improve[j] = False
                        rejected_loops += 1
                best[improve] = candidate[improve]
                codes[si,improve] = ROUTES.index(route)+1
                entry_ids[si,improve],exit_ids[si,improve] = p[improve],q[improve]
                orientations[si,improve] = orientation
                if si==0:
                    detail.append(pd.DataFrame(dict(od_index=np.flatnonzero(improve),route=route,orientation=orientation,
                        candidate_km=candidate[improve],whole_path_model_year_fraction=bit_count((path_ice[p]&path_ice[q]&m["ice"])[improve])/60)))
        without_ids[si],with_km[si] = without,best
        if si==0:
            base_valid = without>0
            base_masks = np.asarray(router.path_mask,dtype=np.uint32)[without]
        affected = base_valid & ((base_masks&closed)!=0)
        selected = codes[si]>0
        row = metric_row(od,weight,affected,without>0,np.isfinite(best),selected,base_valid,pa,pb,countries)
        finite = np.isfinite(best)&(without>0)
        savings = lengths[without][finite]-best[finite]
        row.update(scenario=SCENARIOS[si],closed_count=len(COMBOS[si]),scenario_index=si,shiptype=shiptype,month=month,
            selected_od=int(selected.sum()),routable_without_od=int((without>0).sum()),routable_with_od=int(np.isfinite(best).sum()),
            summed_distance_saved_km=float(savings.sum()),activity_weighted_saved_km=float((savings*weight[finite]).sum()))
        for ri,route in enumerate(ROUTES,1):
            row[f"selected_{route}_od"] = int((codes[si]==ri).sum())
        rows.append(row)
    od.to_parquet(str(prefix)+"_od.parquet",index=False)
    pd.DataFrame(rows).to_parquet(str(prefix)+"_scenarios.parquet",index=False)
    if detail:
        pd.concat(detail,ignore_index=True).to_parquet(str(prefix)+"_open_improving_candidates.parquet",index=False)
    np.savez_compressed(str(prefix)+"_solutions.npz",without_path_id=without_ids,route_code=codes,
        entry_path_id=entry_ids,exit_path_id=exit_ids,orientation=orientations,with_physical_km=with_km)
    router.save_catalogue(prefix)
    audit = dict(version=VERSION,graph_sha256=fingerprint,geometry_sha256=geometry_fingerprint,
        demand_sha256=demand_fingerprint,endpoint_sha256=endpoint_fingerprint,
        shiptype=shiptype,month=month,radius_km=radius,max_closures=max_closures,
        od_count=n,registry_port_count=len(registry_ports),scenarios=count,paths=len(router.paths)-1,dijkstra_trees=router.dijkstra_trees,
        available_cores=list(meta),access_distance_cap_km=None,objective="physical_distance_km_on_explicit_water_graph",
        common_ice_scenario_count=60,common_path_minimum_support=30,
        candidate_checks_rejected_joint_ice=rejected_joint,candidate_checks_rejected_repeated_nodes=rejected_loops,
        limitations="Shortest paths use an ensemble-screened graph; if the shortest access fails joint ice support, a longer member-specific alternative is not searched. Ice capability is a shared hypothetical PC6 assumption, not observed fleet certification.",
        elapsed_seconds=time.monotonic()-started)
    marker.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print("PHYSICAL COMPLETE",shiptype,month,"OD",n,"scenarios",count,"seconds",round(audit["elapsed_seconds"]),flush=True)
    return audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ships",nargs="+",default=["bulk","container","lng","tanker"])
    p.add_argument("--months",nargs="+",type=int,default=list(range(1,13)))
    p.add_argument("--max-closures",type=int,choices=[0,1,2,3],default=3)
    p.add_argument("--radius",type=int,choices=[75,100,150],default=100)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--network-dir",type=Path,default=OUT)
    args=p.parse_args()
    jobs=[(s,m,args.radius,args.max_closures,str(args.network_dir)) for s in args.ships for m in args.months]
    if args.workers==1:
        audits=[run_group(job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            audits=list(pool.imap_unordered(run_group,jobs))
    (args.network_dir/"routing_run_audits.json").write_text(json.dumps(audits,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
