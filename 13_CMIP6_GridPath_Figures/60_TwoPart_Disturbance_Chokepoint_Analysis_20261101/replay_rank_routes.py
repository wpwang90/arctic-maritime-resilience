"""Re-solve complete water routes for the dependency-constrained five-gate sets."""
import json
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT))
from rebuild_mainfig2_latest_network import Geometry, NET, OLD, ROUTES, simple
from recompute_physical_arctic_embedding import PhysicalRouter, core_ice_masks
from repair_cmip6_monthly_routes import repaired_route_metadata
from prepare_physical_monthly_ice import bit_count
from physical_ocean_routing import length
from chokepoints24 import Exposure


def run_month(month, requests):
    out = TASK / 'results/rank_route_replay' / ('month_%02d' % month)
    out.mkdir(parents=True, exist_ok=True)
    requested = [r for r in requests if r['month'] == month]
    if (out / 'audit.json').exists():
        previous = json.loads((out / 'audit.json').read_text())
        if previous['requests'] == requested and previous['status'] == 'PASS':
            print('REPLAY MATCHED MONTH', month, flush=True)
            return
    base = pd.read_parquet(ROOT / '58_FourClass_Threshold_Analysis_20261101/results' / ('month_%02d' % month) / 'demand.parquet')
    geo = Geometry()
    ice = np.load(OLD / 'climate/edge_monthly_ice.npz')['model_year_mask'][month - 1]
    conventional = PhysicalRouter(geo.edges, geo.nodes, ice, 100)
    arouter = PhysicalRouter(geo.edges[bit_count(ice[geo.edges.geometry_id.to_numpy()]) >= 30].reset_index(drop=True), geo.nodes, ice, 100)
    endpoints = pd.read_csv(NET / 'canonical_endpoint_access.csv').set_index('key')
    on = endpoints.loc[base.origin_port_id, 'node'].to_numpy(int)
    dn = endpoints.loc[base.dest_port_id, 'node'].to_numpy(int)
    cs, ct = conventional.to_compact[on], conventional.to_compact[dn]
    ps, pt = arouter.to_compact[on], arouter.to_compact[dn]
    valid = (ps >= 0) & (pt >= 0)
    unique = np.unique(np.r_[ps[ps >= 0], pt[pt >= 0]])
    slots = {x: j for j, x in enumerate(unique)}
    ss, tt = np.array([slots.get(x, 0) for x in ps]), np.array([slots.get(x, 0) for x in pt])
    points, metadata = repaired_route_metadata()
    mapping = pd.read_csv(NET / 'gateway_mapping.csv')
    classifier = Exposure(pd.read_csv(ROOT / 'data/chokepoints24/registry.csv'))
    core_ice = core_ice_masks()
    gates, cores = {}, {}
    for route in ROUTES:
        if (route, month) not in metadata or int(bit_count(np.array([core_ice[(route, month)]]))[0]) < 30:
            continue
        group = mapping[mapping.route.eq(route) & mapping.month.eq(month)].set_index('side')
        pair = []
        for side in ['a', 'b']:
            key = group.loc[side, 'key']
            node = int(arouter.to_compact[int(endpoints.loc[key, 'node'])])
            if node < 0:
                break
            if key not in gates:
                gates[key] = dict(node=node, offset=len(gates) * len(unique))
            pair.append(gates[key])
        if len(pair) == 2:
            xy = points[points.route.eq(route) & points.month.eq(month)].sort_values('order')[['lon', 'lat']].to_numpy(float)
            cores[route] = dict(pair=pair, points=xy, km=length(xy), ice=core_ice[(route, month)], mask=classifier.masks(xy)[100])
    sources = np.concatenate([np.full(len(unique), g['node']) for g in gates.values()]).astype(int) if cores else np.array([], int)
    targets = np.tile(unique, len(gates)) if cores else np.array([], int)

    @lru_cache(maxsize=2048)
    def path_xy(pid):
        return geo.path(arouter.active[arouter.paths[pid]])

    @lru_cache(maxsize=65536)
    def check(p, q, route, orientation):
        if set(arouter.paths[p]) & set(arouter.paths[q]):
            return False
        left, right = path_xy(p)[::-1], path_xy(q)
        core = cores[route]['points'] if orientation == 1 else cores[route]['points'][::-1]
        if not len(left) or not len(right):
            return False
        assert max(length([left[-1], core[0]]), length([core[-1], right[0]])) < .001
        return simple(np.vstack([left, core[1:], right[1:]]))

    n = len(base)
    results = []
    saved = {}
    errors = []
    ref = pd.read_parquet(ROOT / '28_MainFig3_FiveClosures' / ('month_%02d' % month) / 'scenario_metrics.parquet')
    ref = ref[ref.scope.eq('trade_covered')].set_index(['scenario_index', 'shiptype'])
    for request in sorted(requested, key=lambda r: (len(r['gates']), r['scenario_index'])):
        mask = int(request['closed_mask'])
        parents = [m for m in saved if m & mask == m]
        cp = conventional.solve(cs, ct, np.uint32(mask), np.stack([saved[m][0] for m in parents]) if parents else None)
        ap = arouter.solve(sources, targets, np.uint32(mask), np.stack([saved[m][1] for m in parents]) if parents else None) if cores else np.array([], int)
        saved[mask] = (cp, ap)
        without = np.asarray(conventional.length)[cp]
        best = np.full(n, np.inf)
        entry, exit_, code, ori = [np.zeros(n, np.int32) for _ in range(4)]
        joint_best = np.zeros(n, np.uint64)
        if cores:
            lengths = np.asarray(arouter.length)
            icepaths = np.asarray(arouter.path_ice, np.uint64)
            for route_code, route in enumerate(ROUTES, 1):
                if route not in cores or cores[route]['mask'] & mask:
                    continue
                core = cores[route]
                for orientation, (left, right) in enumerate([core['pair'], core['pair'][::-1]], 1):
                    p, q = ap[left['offset'] + ss], ap[right['offset'] + tt]
                    candidate = lengths[p] + core['km'] + lengths[q]
                    joint = icepaths[p] & icepaths[q] & core['ice']
                    take = valid & np.isfinite(candidate) & (candidate < best - 1e-6) & (1.05 * candidate < without - 1e-6) & (bit_count(joint) >= 30)
                    for j in np.flatnonzero(take):
                        if not check(int(p[j]), int(q[j]), route, orientation):
                            take[j] = False
                    best[take] = candidate[take]
                    entry[take], exit_[take], code[take], ori[take] = p[take], q[take], route_code, orientation
                    joint_best[take] = joint[take]
        b = np.isfinite(best)
        a = np.isfinite(without) & ~b
        u = ~(a | b)
        exposure = (np.asarray(conventional.path_mask, np.uint32)[saved[0][0]] & np.uint32(mask)) != 0
        for ship in base.shiptype.unique():
            w = base.trade_weight_usd.to_numpy() * base.shiptype.eq(ship).to_numpy()
            pct = lambda z: 100 * np.sum(w * z) / np.sum(w)
            r = ref.loc[(request['scenario_index'], ship)]
            errors += [abs(float(pct(z)) - float(r[col])) for z, col in [(a, 'A_weight_pct'), (b, 'B_weight_pct'), (u, 'unresolved_pct'), (exposure, 'p_exposed_pct')]]
        results.append(dict(scenario_index=request['scenario_index'], closed_mask=mask, without_km=without,
            arctic_km=best, A=a, B=b, U=u, conventional_id=cp, entry_id=entry, exit_id=exit_, route_code=code,
            orientation=ori, joint_ice=joint_best))
    assert max(errors) < 1e-6, (month, max(errors))
    arrays = {key: np.array([r[key] for r in results]) for key in results[0]}
    np.savez_compressed(out / 'routes.npz', **arrays)
    base.to_parquet(out / 'demand.parquet', index=False)
    conventional.save_catalogue(out / 'conventional')
    arouter.save_catalogue(out / 'arctic')
    audit = dict(status='PASS', month=month, requests=requested, maximum_cached_share_error_pp=float(max(errors)),
        complete_core_access_exit_geometry_checked=True, selected_common_support_at_least_30_of_60=True,
        loops_rejected=True, premium_pct=5, no_port_service_denial=True)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    print('RANK ROUTES MONTH %02d PASS; max error %.3g pp' % (month, max(errors)), flush=True)


if __name__ == '__main__':
    requests = json.loads((TASK / 'results/rank_replay_requests.json').read_text())
    for month in range(1, 13):
        run_month(month, requests)
