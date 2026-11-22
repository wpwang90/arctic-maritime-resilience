"""Economics of the exact vessel-specific routes and weights used in new Figure 4."""
import calendar
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREV = ROOT / '60_TwoPart_Disturbance_Chokepoint_Analysis_20261101'
OLD_IO = ROOT / '32_NCC2023_IndirectExposure/results'
OUT = HERE / 'results'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from rebuild_mainfig4_regional_monthly import loss_kernel

SHIPS = ['container', 'bulk', 'tanker', 'lng']
BUFFERS = [0, 3, 7, 14]
CHANNELS = ['transport_insurance_usd', 'delay_usd', 'inventory_usd']
USD_PER_MEUR = 1e6 * 1.0824


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def routes_and_losses():
    sets = pd.read_csv(PREV / 'results/partII_mainfig4_fixed_month_optimal_closures.csv')
    sets = sets[sets.n_closed.eq(5)].set_index('shiptype')
    margin_table = pd.read_csv(ROOT / '30_MainFig4_CountryRelief_Audit/oecd_2024_primary_country_pair_margins.csv')
    margin_lookup = margin_table.set_index(['importer_iso3', 'exporter_iso3']).margin_cif_pct
    fallback = float(margin_table.margin_cif_pct.median())
    rows, audits = [], []
    for month in range(1, 13):
        folder = PREV / 'results/rank_route_replay' / ('month_%02d' % month)
        base = pd.read_parquet(folder / 'demand.parquet')
        a = np.load(folder / 'routes.npz')
        baseline_state = int(np.flatnonzero(a['scenario_index'] == 0)[0])
        for ship in SHIPS:
            selected = sets.loc[ship]
            state = int(np.flatnonzero(a['scenario_index'] == int(selected.scenario_index))[0])
            take = base.shiptype.eq(ship).to_numpy()
            d = base.loc[take].copy().reset_index(drop=True)
            d0, da, db = a['without_km'][baseline_state, take], a['without_km'][state, take], a['arctic_km'][state, take]
            choice = a['B'][state, take]
            picked = np.where(choice, db, da)
            np.testing.assert_array_equal(choice, np.isfinite(db) & (1.05 * db < da - 1e-6))
            assert np.isfinite(d0).all() and (d0 > 0).all()
            keys = pd.MultiIndex.from_arrays([d.dest_iso3, d.origin_iso3])
            margins = margin_lookup.reindex(keys).to_numpy(float)
            d['margin_imputed'] = ~np.isfinite(margins)
            margins = np.where(np.isfinite(margins), margins, fallback)
            freight = margins / (100 - margins)
            d['margin_cif_pct'] = margins
            d['month'] = month
            d['scenario_index'] = int(selected.scenario_index)
            d['closed_gates'] = selected.closed_gates
            d['primary_case'] = month == int(selected.month)
            d['days_in_month'] = calendar.monthrange(2024, month)[1]
            d['open_km'], d['conventional_km'], d['arctic_km'] = d0, da, db
            d['selected_B'] = choice
            d['selected_U'] = a['U'][state, take]
            d['selected_A'] = a['A'][state, take]
            d['new_B'] = choice & ~a['B'][baseline_state, take]
            paired = {}
            for case, distance in [('without_arctic', da), ('with_arctic', picked)]:
                values = loss_kernel(d.trade_weight_usd.to_numpy() / d.days_in_month.to_numpy(), d0,
                                     distance, freight, premium=0.)
                if case == 'with_arctic' and choice.any():
                    values[choice] = loss_kernel(
                        d.trade_weight_usd.to_numpy()[choice] / d.days_in_month.to_numpy()[choice],
                        d0[choice], distance[choice], freight[choice], premium=.05)
                frame = d.copy()
                frame['case'] = case
                frame['selected_distance_km'] = distance
                delay = np.full(len(d), 30.)
                finite = np.isfinite(distance)
                delay[finite] = np.minimum(30., np.maximum(distance[finite] - d0[finite], 0.) / (16 * 1.852 * 24))
                frame['effective_delay_days'] = delay
                frame[CHANNELS] = values
                frame['total_loss_usd'] = values.sum(1)
                rows.append(frame)
                paired[case] = frame
            assert np.all(paired['with_arctic'].effective_delay_days <= paired['without_arctic'].effective_delay_days + 1e-9)
            assert np.all(paired['with_arctic'].total_loss_usd <= paired['without_arctic'].total_loss_usd + .01)
            pct = lambda mask: float(100 * d.trade_weight_usd[mask].sum() / d.trade_weight_usd.sum())
            audits.append(dict(shiptype=ship, month=month, primary_case=month == int(selected.month),
                od_count=len(d), trade_weight_usd=d.trade_weight_usd.sum(), A_pct=pct(d.selected_A), B_pct=pct(d.selected_B),
                U_pct=pct(d.selected_U), new_B_pct=pct(d.new_B), scenario_index=int(selected.scenario_index),
                closed_gates=selected.closed_gates, margin_imputed_weight_pct=pct(d.margin_imputed),
                source_demand_sha256=sha(folder / 'demand.parquet'), source_routes_sha256=sha(folder / 'routes.npz')))
            if month == int(selected.month):
                np.testing.assert_allclose([pct(d.selected_A), pct(d.selected_B), pct(d.selected_U)],
                    [selected.A_weight_pct, selected.B_weight_pct, selected.unresolved_pct], atol=1e-8)
        print('ROUTE-ALIGNED ECONOMICS', month, flush=True)
    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(OUT / 'paired_od_losses.parquet', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'route_to_Figure4_alignment.csv', index=False)
    losscols = CHANNELS + ['total_loss_usd']
    result.groupby(['shiptype', 'month', 'primary_case', 'case'])[losscols].sum().reset_index().to_csv(
        OUT / 'monthly_loss_channels.csv', index=False)
    endpoint = []
    for side in ['origin', 'dest']:
        frame = result[['shiptype', 'month', 'primary_case', 'case', side + '_iso3'] + losscols].copy()
        frame = frame.rename(columns={side + '_iso3': 'iso3'})
        frame[losscols] *= .5
        endpoint.append(frame)
    country = pd.concat(endpoint).groupby(['shiptype', 'month', 'primary_case', 'case', 'iso3'])[losscols].sum().reset_index()
    country.to_csv(OUT / 'country_endpoint_losses.csv', index=False)
    return result, sets


def sector_allocation(routes):
    baci = pd.read_parquet(OLD_IO / 'baci2024_country_industry_vessel_eligibility.parquet')
    baseline = routes[routes.case.eq('without_arctic')].copy()
    assert baseline.od_id.is_unique and len(baseline) == 16233
    allocated = []
    for i, ship in enumerate(SHIPS):
        eligible = baci[(baci.vessel_mask.astype(int) & (1 << i)) != 0].copy()
        eligible = eligible.groupby(['origin_iso3', 'dest_iso3', 'origin_region', 'dest_region', 'group'], as_index=False).baci_usd.sum()
        eligible['pair_total'] = eligible.groupby(['origin_iso3', 'dest_iso3']).baci_usd.transform('sum')
        eligible['commodity_fraction'] = eligible.baci_usd / eligible.pair_total
        merged = baseline[baseline.shiptype.eq(ship)][['od_id', 'shiptype', 'month', 'origin_iso3', 'dest_iso3', 'trade_weight_usd']].merge(
            eligible, on=['origin_iso3', 'dest_iso3'], how='inner', validate='many_to_many')
        merged['sector_weight_usd'] = merged.trade_weight_usd * merged.commodity_fraction
        allocated.append(merged)
    allocation = pd.concat(allocated, ignore_index=True)
    check = allocation.groupby('od_id').sector_weight_usd.sum()
    original = baseline.set_index('od_id').trade_weight_usd
    np.testing.assert_allclose(check, original.reindex(check.index), rtol=1e-10)
    # No sector may be assigned more annual represented value than its observed
    # bilateral merchandise pool; unsupported portions remain an explicit gap.
    annual_assigned = allocation.groupby(['origin_iso3', 'dest_iso3', 'group']).sector_weight_usd.sum()
    capacity = baci.groupby(['origin_iso3', 'dest_iso3', 'group']).baci_usd.sum()
    fraction = (capacity.reindex(annual_assigned.index) / annual_assigned).clip(upper=1)
    index = pd.MultiIndex.from_frame(allocation[['origin_iso3', 'dest_iso3', 'group']])
    allocation['capacity_factor'] = fraction.reindex(index).to_numpy()
    allocation['supported_sector_weight_usd'] = allocation.sector_weight_usd * allocation.capacity_factor
    allocation.to_parquet(OUT / 'OD_industry_weight_bridge.parquet', index=False)
    covered = allocation.groupby('od_id').supported_sector_weight_usd.sum()
    coverage = baseline[['od_id', 'shiptype', 'month', 'primary_case', 'trade_weight_usd']].copy()
    coverage['mapped_usd'] = coverage.od_id.map(check).fillna(0)
    coverage['supported_usd'] = coverage.od_id.map(covered).fillna(0)
    coverage.to_csv(OUT / 'SI_commodity_bridge_coverage.csv', index=False)
    totals = coverage.groupby(['shiptype', 'primary_case'])[['trade_weight_usd', 'mapped_usd', 'supported_usd']].sum()
    totals['supported_pct'] = 100 * totals.supported_usd / totals.trade_weight_usd
    totals.to_csv(OUT / 'commodity_bridge_coverage_summary.csv')
    denominator = baci.groupby(['origin_region', 'group', 'dest_region']).baci_usd.sum()
    return allocation, denominator


def compute_io(routes, allocation, denominator):
    mrio = np.load(OLD_IO / 'mrio2024_aggregated.npz')
    z, x, v = mrio['z'], mrio['output'], mrio['value_added']
    regions = np.unique(mrio['region'])
    region_index = {r: i for i, r in enumerate(regions)}
    nodes = pd.DataFrame({'region': mrio['region'], 'group': mrio['group']})
    ng = 34
    full_index = pd.MultiIndex.from_product([regions, range(ng), regions], names=['origin_region', 'group', 'dest_region'])
    denom = denominator.reindex(full_index, fill_value=0).to_numpy().reshape(len(regions) * ng, len(regions))
    node_rows = np.array([region_index[r] * ng + int(g) for r, g in zip(mrio['region'], mrio['group'])])
    node_buyers = np.array([region_index[r] for r in mrio['region']])
    source_indices = [np.flatnonzero(mrio['region'] == r) for r in regions]
    industry_results, country_sources, checks = [], [], []
    for month in range(1, 13):
        for ship in SHIPS:
            route = routes[routes.month.eq(month) & routes.shiptype.eq(ship)]
            is_primary = bool(route.primary_case.iloc[0])
            weights = allocation[allocation.shiptype.eq(ship) & allocation.month.eq(month)].copy()
            international = weights.origin_region.ne(weights.dest_region)
            weights = weights[international].copy()
            rowid = weights.origin_region.map(region_index).to_numpy() * ng + weights.group.to_numpy()
            colid = weights.dest_region.map(region_index).to_numpy()
            for buffer in BUFFERS:
                paired = {}
                for case in ['without_arctic', 'with_arctic']:
                    selected = route[route.case.eq(case)].set_index('od_id')
                    delay = np.maximum(selected.effective_delay_days - buffer, 0)
                    event_fraction = weights.od_id.map(delay).to_numpy() / calendar.monthrange(2024, month)[1]
                    # q is an event-total equivalent fraction of annual supply.
                    # The monthly allocated trade weights are not annualized again.
                    numerator = np.zeros_like(denom)
                    np.add.at(numerator, (rowid, colid), weights.supported_sector_weight_usd.to_numpy() * event_fraction)
                    qsmall = np.divide(numerator, denom, out=np.zeros_like(denom), where=denom > 0)
                    assert np.isfinite(qsmall).all() and qsmall.max() <= 1 + 1e-10, (ship, month, buffer, qsmall.max())
                    q = qsmall[node_rows][:, node_buyers]
                    extracted = z * q
                    matrix = np.eye(len(x)) - ((z - extracted) / x[:, None]).T
                    lu = lu_factor(matrix, check_finite=False)
                    first = extracted.sum(0)
                    total = lu_solve(lu, first, check_finite=False)
                    residual = np.max(np.abs(np.einsum('ij,j->i', matrix, total, optimize=False) - first)) / max(1, float(first.max()))
                    equilibrium = np.max(np.abs(np.einsum('ij,j->i', matrix, x - total, optimize=False) - v)) / float(x.max())
                    assert residual < 1e-8 and equilibrium < 1e-7
                    assert total.min() > -1e-7 and (total - first).min() > -1e-7
                    frame = nodes.copy()
                    frame['shiptype'], frame['month'], frame['primary_case'] = ship, month, is_primary
                    frame['case'], frame['buffer_days'] = case, buffer
                    frame['first_hit_usd'] = np.maximum(first, 0) * USD_PER_MEUR
                    frame['higher_order_usd'] = np.maximum(total - first, 0) * USD_PER_MEUR
                    frame['output_exposure_usd'] = np.maximum(total, 0) * USD_PER_MEUR
                    industry_results.append(frame)
                    paired[case] = dict(q=q, lu=lu, total=total)
                    if buffer == 0 and is_primary:
                        rhs = np.stack([extracted[index].sum(0) for index in source_indices], axis=1)
                        source_values = lu_solve(lu, rhs, check_finite=False)
                        np.testing.assert_allclose(source_values.sum(1), total, rtol=1e-9, atol=1e-7)
                        source_frame = pd.DataFrame(dict(
                            source_region=np.repeat(regions, len(x)), receiver_region=np.tile(mrio['region'], len(regions)),
                            group=np.tile(mrio['group'], len(regions)), output_exposure_usd=source_values.T.ravel() * USD_PER_MEUR,
                            first_hit_usd=rhs.T.ravel() * USD_PER_MEUR))
                        source_frame['shiptype'], source_frame['month'], source_frame['case'] = ship, month, case
                        country_sources.append(source_frame)
                    checks.append(dict(shiptype=ship, month=month, primary_case=is_primary, case=case, buffer_days=buffer,
                        linear_residual=residual, equilibrium_residual=equilibrium, max_q=float(qsmall.max()),
                        output_exposure_usd=float(total.sum() * USD_PER_MEUR),
                        first_hit_usd=float(first.sum() * USD_PER_MEUR)))
                np.testing.assert_array_less(paired['with_arctic']['q'], paired['without_arctic']['q'] + 1e-10)
                assert np.all(paired['with_arctic']['total'] <= paired['without_arctic']['total'] + 1e-7)
                if buffer == 0 and is_primary:
                    wo, wi = paired['without_arctic'], paired['with_arctic']
                    restored = z * (wo['q'] - wi['q']) * ((x - wo['total']) / x)[:, None]
                    rhs = np.stack([restored[index].sum(0) for index in source_indices], axis=1)
                    relief = lu_solve(wi['lu'], rhs, check_finite=False)
                    np.testing.assert_allclose(relief.sum(1), wo['total'] - wi['total'], rtol=1e-6, atol=1e-7)
                    frame = pd.DataFrame(dict(source_region=np.repeat(regions, len(x)),
                        receiver_region=np.tile(mrio['region'], len(regions)), group=np.tile(mrio['group'], len(regions)),
                        output_exposure_usd=relief.T.ravel() * USD_PER_MEUR,
                        first_hit_usd=rhs.T.ravel() * USD_PER_MEUR))
                    frame['shiptype'], frame['month'], frame['case'] = ship, month, 'avoided'
                    country_sources.append(frame)
            print('ALIGNED IO', month, ship, 'primary' if is_primary else 'SI', flush=True)
    data = pd.concat(industry_results, ignore_index=True)
    data.to_parquet(OUT / 'mrio_industry_month_vessel_exposure.parquet', index=False)
    pd.concat(country_sources).to_parquet(OUT / 'primary_country_source_receiver_exposure.parquet', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'hem_equilibrium_checks.csv', index=False)
    data.groupby(['shiptype', 'month', 'primary_case', 'case', 'buffer_days', 'region'])[
        ['first_hit_usd', 'higher_order_usd', 'output_exposure_usd']].sum().reset_index().to_csv(OUT / 'country_output_exposure.csv', index=False)
    primary = data[data.primary_case & data.buffer_days.eq(0)].groupby('case').output_exposure_usd.sum() / 4
    audit = dict(status='PASS', routing_and_weights_identical_to_Figure4=True, duration_days=30,
        ship_cases_independent_not_joint=True, primary_months=dict(container=7, bulk=7, tanker=9, lng=8),
        primary_aggregate='Equal-weight mean of four separate vessel-specific 30-day stress tests at Figure4 months',
        annual_diagnostic='All twelve months with each Figure4-selected closure set held fixed',
        q_definition='sum(supported monthly OD industry weight * residual delay / days_in_month) / annual bilateral industry merchandise value',
        output_units='2024 USD event-total-equivalent structural gross-output exposure',
        output_conversion=USD_PER_MEUR, no_second_duration_year_scaling=True,
        fixed_route_choice='Exact 5%-distance-premium Figure4 selection; no economic reoptimization',
        commodity_assignment='Published concordances plus ship compatibility, not observed cargo manifests',
        unsupported_sector_capacity_not_imputed=True, within_RoW_extraction_excluded=True,
        maximum_q=float(pd.DataFrame(checks).max_q.max()), checks=len(checks),
        primary_mean_without_usd=float(primary.without_arctic), primary_mean_with_usd=float(primary.with_arctic),
        primary_mean_avoided_pct=float(100 * (primary.without_arctic - primary.with_arctic) / primary.without_arctic),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [
            OLD_IO / 'mrio2024_aggregated.npz', OLD_IO / 'baci2024_country_industry_vessel_eligibility.parquet',
            PREV / 'results/partII_mainfig4_fixed_month_optimal_closures.csv', PREV / 'results/MainFig4_country_rank_source.csv']})
    (OUT / 'economic_recalculation_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        routes, sets = routes_and_losses()
        allocation, denominator = sector_allocation(routes)
        compute_io(routes, allocation, denominator)
