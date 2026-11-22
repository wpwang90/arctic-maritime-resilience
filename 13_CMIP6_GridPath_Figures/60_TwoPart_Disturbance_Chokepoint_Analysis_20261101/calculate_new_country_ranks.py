"""Same original proximity-betweenness estimator, new OD-derived closure sets."""
import json
from functools import lru_cache
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pycountry
import shapely
from scipy.spatial import cKDTree

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
OUT = TASK / 'results'
OLD = ROOT / '51_TradeWeighted_CountryBetweenness/results'
sys.path.insert(0, str(ROOT / '51_TradeWeighted_CountryBetweenness'))
sys.path.insert(0, str(ROOT))
from calculate_coastal_betweenness import sample_line, coastal_parts, EARTH
from rebuild_mainfig2_latest_network import Geometry, ROUTES
from repair_cmip6_monthly_routes import repaired_route_metadata

RADII = [25, 50, 98, 100, 102, 200]
SHIPS = ['container', 'bulk', 'tanker', 'lng']


def main():
    geo = Geometry()
    points, _ = repaired_route_metadata()
    chosen = pd.read_csv(OUT / 'partII_mainfig4_fixed_month_optimal_closures.csv')
    chosen = chosen[chosen.n_closed == 5].set_index('shiptype')
    world = gpd.read_parquet(OLD / 'coastal_countries.parquet')
    mapping = pd.read_csv(OLD / 'country_code_mapping.csv').drop_duplicates('admin_iso3').set_index('admin_iso3').iso3.to_dict()
    old_catalogue = json.loads((OLD / 'physical_route_catalogue.json').read_text())
    old_lookup = {key: j for j, key in enumerate(old_catalogue['feature_keys'])}
    old_distances = np.load(OLD / 'feature_country_distances_v2.npz')
    old_cloud = np.load(OLD / 'sampled_route_features.npz')
    old_xyz, old_offsets = old_cloud['xyz'], old_cloud['offsets']
    countries = old_distances['countries'].tolist()
    assert countries == world.loc[world.coastal, 'iso3'].tolist()
    features, feature_xy, cache_match = {}, [], []
    route_ids, routes, od_rows = {}, [], []

    def feature(key, xy):
        if key in features:
            return features[key]
        fi = len(features)
        features[key] = fi
        samples = sample_line(xy)
        oj = old_lookup.get(key)
        same = False
        if oj is not None:
            old = old_xyz[old_offsets[oj]:old_offsets[oj + 1]]
            same = old.shape == samples.shape and np.allclose(old, samples, atol=2e-7, rtol=0)
        cache_match.append(oj if same else -1)
        feature_xy.append(samples)
        return fi

    def edge_features(nodes):
        ids = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            edge = geo.lookup[(min(a, b), max(a, b))]
            gid = int(edge.geometry_id)
            xy = geo.xy[geo.offsets[gid]:geo.offsets[gid + 1]]
            ids.append(feature('edge_' + str(gid), xy))
        if not ids and len(nodes):
            ids = [feature('node_' + str(nodes[0]), geo.nodes[nodes[:1]])]
        return tuple(ids)

    def intern(ids):
        key = tuple(sorted(set(ids)))
        if not key:
            return -1
        if key not in route_ids:
            route_ids[key] = len(routes)
            routes.append(key)
        return route_ids[key]

    for month in range(1, 13):
        folder = OUT / 'rank_route_replay' / ('month_%02d' % month)
        assert json.loads((folder / 'audit.json').read_text())['status'] == 'PASS'
        base = pd.read_parquet(folder / 'demand.parquet')
        state = dict(np.load(folder / 'routes.npz'))
        cats = {kind: dict(np.load(folder / (kind + '_paths.npz'))) for kind in ['conventional', 'arctic']}

        @lru_cache(maxsize=32768)
        def path(kind, pid):
            cat = cats[kind]
            nodes = cat['nodes'][cat['offsets'][pid]:cat['offsets'][pid + 1]]
            return edge_features(nodes)

        for ship in SHIPS:
            for step, index in [(0, 0), (5, int(chosen.loc[ship].scenario_index))]:
                si = int(np.flatnonzero(state['scenario_index'] == index)[0])
                for j in np.flatnonzero(base.shiptype.eq(ship).to_numpy()):
                    row = base.iloc[j]
                    aid = -1
                    if np.isfinite(state['without_km'][si, j]):
                        aid = intern(path('conventional', int(state['conventional_id'][si, j])))
                    bid = aid
                    if state['B'][si, j]:
                        route = ROUTES[int(state['route_code'][si, j]) - 1]
                        core = points[points.route.eq(route) & points.month.eq(month)].sort_values('order')[['lon', 'lat']].to_numpy(float)
                        core_id = feature('core_%s_%d' % (route, month), core)
                        bid = intern(path('arctic', int(state['entry_id'][si, j])) + path('arctic', int(state['exit_id'][si, j])) + (core_id,))
                    for network, rid in [('traditional', aid), ('embedded_premium05', bid)]:
                        od_rows.append(dict(shiptype=ship, month=month, step=step, network=network, od_id=int(row.od_id),
                            origin_iso3=mapping.get(row.origin_iso3, row.origin_iso3),
                            dest_iso3=mapping.get(row.dest_iso3, row.dest_iso3), trade_weight_usd=float(row.trade_weight_usd),
                            route_id=rid, scenario_index=index, is_arctic=bool(network != 'traditional' and state['B'][si, j])))
        print('COUNTRY ROUTE FEATURES MONTH', month, len(features), flush=True)
    rows = pd.DataFrame(od_rows)
    rows.to_parquet(OUT / 'country_rank_OD_route_states.parquet', index=False)
    distance = np.full((len(features), len(countries)), np.inf, np.float32)
    known = np.array(cache_match) >= 0
    distance[known] = old_distances['distance_km'][np.asarray(cache_match)[known]]
    missing = np.flatnonzero(~known)
    if len(missing):
        samples = [feature_xy[j] for j in missing]
        offsets = np.r_[0, np.cumsum([len(p) for p in samples])]
        cloud = np.vstack(samples)
        cap = 2 * np.sin(205. / EARTH / 2)
        for col, row in enumerate(world[world.coastal].itertuples()):
            coast = shapely.from_wkb(row.coast_wkb)
            shore = np.vstack([sample_line(g) for g in coastal_parts(coast)])
            tree = cKDTree(shore)
            chord, _ = tree.query(cloud, k=1, distance_upper_bound=cap, workers=1)
            km = 2 * EARTH * np.arcsin(np.minimum(chord / 2, 1))
            km[~np.isfinite(chord)] = np.inf
            distance[missing, col] = np.minimum.reduceat(km, offsets[:-1])
            if col % 20 == 0:
                print('NEW COAST DISTANCES', col + 1, '/', len(countries), flush=True)
    route_distance = np.stack([distance[np.asarray(ids)].min(axis=0) for ids in routes])
    np.savez_compressed(OUT / 'new_route_country_distances.npz', distance_km=route_distance, countries=np.asarray(countries))
    np.savez_compressed(OUT / 'new_feature_country_distances.npz', distance_km=distance, countries=np.asarray(countries))
    (OUT / 'new_route_feature_catalogue.json').write_text(json.dumps(dict(feature_keys=list(features), route_features=routes)))
    codes = {c.alpha_3 for c in pycountry.countries}
    keep = np.array([c in codes for c in countries])
    ids = np.array(countries)[keep]
    d = route_distance[:, keep]
    names = world.set_index('iso3')['name'].to_dict()
    aggregate = []
    groupings = list(rows.groupby(['shiptype', 'month', 'step', 'network'], sort=False))
    groupings += [((ship, 0, step, network), group) for (ship, step, network), group in rows.groupby(['shiptype', 'step', 'network'], sort=False)]
    for (ship, month, step, network), group in groupings:
        w, rid = group.trade_weight_usd.to_numpy(), group.route_id.to_numpy(int)
        distances = np.full((len(group), len(ids)), np.inf)
        good = rid >= 0
        distances[good] = d[rid[good]]
        endpoints = (group.origin_iso3.to_numpy()[:, None] == ids) | (group.dest_iso3.to_numpy()[:, None] == ids)
        distances[endpoints] = np.inf
        for radius in RADII:
            passed = distances <= radius
            score = 100 * np.sum(w[:, None] * passed, axis=0) / np.sum(w)
            rank = pd.Series(score).round(10).rank(method='min', ascending=False).astype(int).to_numpy()
            for j, iso in enumerate(ids):
                aggregate.append(dict(shiptype=ship, month=month, aggregation='annual' if month == 0 else 'monthly', step=step,
                    network=network, radius_km=radius, iso3=iso, name=names[iso], betweenness_pct=float(score[j]),
                    rank=int(rank[j]), transit_od_count=int(passed[:, j].sum()), transit_trade_usd=float(np.sum(w[passed[:, j]])),
                    total_trade_usd=float(w.sum()), od_count=len(group), reachable_trade_pct=float(100 * w[good].sum() / w.sum())))
    result = pd.DataFrame(aggregate)
    result.to_csv(OUT / 'country_betweenness_new_closures.csv', index=False)
    (OUT / 'country_rank_calculation_audit.json').write_text(json.dumps(dict(status='PASS',
        main_radius_km=100, sensitivity_radii_km=RADII, rank_country_universe=len(ids),
        feature_count=len(features), geometry_checked_cached_features=int(known.sum()), newly_sampled_features=len(missing),
        cached_feature_geometry_agreement_tolerance_unit_sphere=2e-7, route_count=len(routes), OD_network_state_records=len(rows),
        estimator='Fixed-demand weighted single-shortest-path coastal group betweenness; endpoint countries excluded',
        coastal_sampling_km=2, not_territorial_waters=True, all_twelve_months_retained=True,
        new_Arctic_choice_uses_full_voyage_5pct_premium=True, same_closures_in_both_compared_networks=True,
        population_not_used_in_route_or_closure_optimization=True), indent=2))
    print('COUNTRY RANKS COMPLETE', len(result), flush=True)


if __name__ == '__main__':
    main()
