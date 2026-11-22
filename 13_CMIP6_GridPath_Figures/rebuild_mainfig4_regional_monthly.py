"""Conditional monthly disruption losses on the latest physical route network.

No event probabilities or future trade growth are inferred. Country attribution
uses the receiving port, not the incidence of welfare or final consumption.
"""
from __future__ import annotations

import argparse
import calendar
import itertools
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rebuild_mainfig3_lcc_transition import ROOT, OLD, NET, run, cost_choice, sha

OUT = ROOT / '29_MainFig4_RegionalMonthlyLoss'
SHIPS = ['container', 'bulk', 'tanker', 'lng']
CASES = {'container': (0, 15), 'bulk': (2, 16),
         'tanker': (0, 11, 18), 'lng': (0, 10, 11, 15, 18)}
COMMON = CASES['lng']
CHANNELS = ['transport_insurance_usd', 'delay_usd', 'inventory_usd']
RAW_OECD = Path((__import__("os").environ.get("ARCTIC_LOCAL_DATA", "external") + '/[OECD保险成本]OECD.SDD.TPS,DSD_ITIC@DF_ITIC,1.1+all.csv'))


def combinations():
    combos = {()}
    for gates in CASES.values():
        combos.update(c for k in range(1, len(gates)+1)
                      for c in itertools.combinations(gates, k))
    return sorted(combos, key=lambda c: (len(c), c))


def prepare_routes():
    OUT.mkdir(exist_ok=True)
    combos = combinations()
    for month in range(1, 13):
        path = OUT / 'routes' / f'month_{month:02d}'
        if (path / 'calculation_audit.json').exists():
            saved = json.loads((path / 'scenarios.json').read_text())
            assert saved == [list(c) for c in combos]
            print(f'Reusing month {month:02d}', flush=True)
            continue
        run(month, 5, combos, destination_root=OUT / 'routes')


def prepare_margins():
    """Extract one year and one HS resolution, without pooling years/products."""
    import duckdb
    OUT.mkdir(exist_ok=True)
    target = OUT / 'oecd_2024_hs4_country_pair_margins.csv'
    if target.exists():
        print('Reusing 2024 OECD pair cache', flush=True)
        return
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    raw = str(RAW_OECD).replace("'", "''")
    print('Reading the local OECD CSV; extracting 2024 HS4 margins', flush=True)
    result = con.execute(f"""
        SELECT REF_AREA AS importer_iso3, COUNTERPART_AREA AS exporter_iso3,
               MEDIAN(TRY_CAST(OBS_VALUE AS DOUBLE)) AS margin_cif_pct,
               COUNT(*) AS product_records,
               COUNT(DISTINCT COMM_HS2017) AS hs4_products,
               COUNT(*) FILTER (WHERE OBS_STATUS = 'I') AS imputed_records,
               COUNT(*) FILTER (WHERE METHODOLOGY_TYPE = 'ME') AS model_records
        FROM read_csv_auto('{raw}', header=true, all_varchar=true,
                           ignore_errors=true, store_rejects=true, rejects_limit=0)
        WHERE TIME_PERIOD = '2024' AND MEASURE = 'C_F'
          AND UNIT_MEASURE = 'PT_IM_CIF' AND PRODUCT_TYPE = 'C'
          AND regexp_full_match(COMM_HS2017, 'HS17_[0-9]{{4}}')
          AND TRY_CAST(OBS_VALUE AS DOUBLE) >= 0
          AND TRY_CAST(OBS_VALUE AS DOUBLE) < 100
        GROUP BY 1, 2
    """).fetchdf()
    assert len(result) and result.product_records.eq(result.hs4_products).all()
    result.to_csv(target, index=False)
    rejects = con.execute('SELECT * FROM reject_errors').fetchdf()
    rejects.to_csv(OUT/'SI_oecd_csv_parse_rejects.csv', index=False)
    (OUT / 'oecd_extraction_provenance.json').write_text(json.dumps(dict(
        raw_path=str(RAW_OECD), raw_bytes=RAW_OECD.stat().st_size,
        raw_mtime=RAW_OECD.stat().st_mtime, year=2024, measure='C_F',
        unit='PT_IM_CIF', product_regex='HS17_[0-9]{4}',
        aggregation='Unweighted median across distinct HS4 products per directed pair',
        note='Analytical OECD estimates, not observed vessel invoices or sea-only prices. Local raw CSV ends in a truncated quoted record; parsing rejects are exported, not silently discarded.',
        rejected_rows=len(rejects), rows=len(result), cache_sha256=sha(target)), indent=2))
    print(f'OECD 2024: {len(result)} country pairs', flush=True)


def public_margins():
    """Use official all-commodity pair aggregates, not HS-unweighted medians."""
    import requests
    OUT.mkdir(exist_ok=True)
    raw = OUT/'oecd_public_2024_total_commodity.csv'
    url = 'https://sdmx.oecd.org/sti-public/rest/v1/data/OECD.SDD.TPS,DSD_ITIC@DF_ITIC,1.1/..C_F.C._T.A.PT_IM_CIF'
    params = dict(startPeriod='2024',endPeriod='2024',format='csvfile')
    if not raw.exists():
        response = requests.get(url, params=params, timeout=180)
        response.raise_for_status()
        table = pd.read_csv(io.StringIO(response.text))
        assert {'REF_AREA','COUNTERPART_AREA','OBS_VALUE'} <= set(table)
        raw.write_bytes(response.content)
        (OUT/'oecd_public_download.json').write_text(json.dumps(dict(
            url=response.url, retrieved_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            bytes=len(response.content), raw_sha256=sha(raw), rows=len(table)), indent=2))
    table = pd.read_csv(raw)
    assert table.TIME_PERIOD.eq(2024).all() and table.UNIT_MEASURE.eq('PT_IM_CIF').all()
    known = set(pd.read_csv(OLD/'panel_c_country_region_mapping.csv').iso3)
    table = table[table.REF_AREA.isin(known) & table.COUNTERPART_AREA.isin(known)].copy()
    assert not table.duplicated(['REF_AREA','COUNTERPART_AREA']).any()
    table = table.rename(columns={'REF_AREA':'importer_iso3','COUNTERPART_AREA':'exporter_iso3','OBS_VALUE':'margin_cif_pct'})
    table['margin_source'] = 'OECD_2024_total_commodity'
    table[['importer_iso3','exporter_iso3','margin_cif_pct','margin_source','OBS_STATUS','METHODOLOGY_TYPE']].to_csv(
        OUT/'oecd_2024_primary_country_pair_margins.csv',index=False)
    print(f'Official 2024 total-commodity margins: {len(table)} pairs, {table.importer_iso3.nunique()} importers',flush=True)


def loss_kernel(value_day, baseline, route, freight_fraction, duration=30.,
                speed_knots=16., premium=0., vot=.01, inventory_scale=1., margin_scale=1.):
    """Return waiting and route-response loss channels for one fixed cohort.

    Baseline freight is inferred from CIF shares converted to a FOB ratio.
    All monetary outputs are conditional event totals, not USD per year.
    """
    inventory = np.array([7., 25., 60.]) * inventory_scale
    fractions = np.array([.15, .50, .35])
    finite = np.isfinite(route)
    extra_days = np.full(len(baseline), duration)
    extra_days[finite] = np.maximum(route[finite]-baseline[finite], 0) / (speed_knots*1.852*24)
    days = np.minimum(extra_days, duration)
    can_reroute = finite & (extra_days < duration)
    charge_distance = np.zeros(len(baseline))
    charge_distance[can_reroute] = np.maximum(
        (1+premium)*route[can_reroute]/baseline[can_reroute]-1, 0)
    transport = value_day * duration * freight_fraction * margin_scale * charge_distance
    capped = np.minimum(days[:, None], inventory[None, :])
    # Continuous triangular extension avoids whole-day rounding artefacts.
    delay = value_day * vot * np.sum(capped*(capped+1)/2*fractions, axis=1)
    stockout = value_day * np.sum(np.maximum(days[:, None]-inventory, 0)*fractions, axis=1)
    return np.column_stack([transport, delay, stockout])


def evaluate(base, arrays, si, margins, duration=30., speed=16., premium=.05,
             vot=.01, inventory_scale=1., margin_scale=1.):
    w = base.trade_weight_usd.to_numpy(float)
    month = int(base.start_month.iloc[0])
    value_day = w/calendar.monthrange(2024, month)[1]
    # Use the same open-graph optimum as the disruption counterfactual.
    # Raw AIS polylines describe the graph/data provenance, not an inefficient
    # observed path against which a shortest detour could look spuriously cheap.
    d0 = arrays['without_km'][0]
    dc = arrays['without_km'][si]
    da = arrays['arctic_km'][si]
    params = dict(duration=duration, speed_knots=speed, vot=vot,
                  inventory_scale=inventory_scale, margin_scale=margin_scale)
    wait = loss_kernel(value_day, d0, np.full(len(base), np.inf), margins, **params)
    conventional = loss_kernel(value_day, d0, dc, margins, **params)
    cw = np.isfinite(dc) & ((dc-d0)/(speed*1.852*24) < duration) & (conventional.sum(axis=1) <= wait.sum(axis=1))
    without = np.where(cw[:, None], conventional, wait)
    eligible, _ = cost_choice(dc, da, premium)
    arctic = loss_kernel(value_day, d0, da, margins, premium=premium, **params)
    use_arctic = eligible & (arctic.sum(axis=1) < without.sum(axis=1)-1e-8)
    with_arctic = np.where(use_arctic[:, None], arctic, without)
    return without, with_arctic, use_arctic, cw, dc, da


def load_month(month, margin_table):
    folder = OUT/'routes'/f'month_{month:02d}'
    raw = pd.read_parquet(folder/'demand.parquet')
    eligible = raw.trade_weight_usd.notna().to_numpy()
    base = raw.loc[eligible].reset_index(drop=True)
    arrays = {k: v[:, eligible] if v.ndim == 2 else v[eligible]
              for k, v in np.load(folder/'scenario_routes.npz').items()}
    assert np.isfinite(arrays['without_km'][0]).all()
    assert (arrays['without_km'][0] > 0).all()
    lookup = margin_table.set_index(['importer_iso3', 'exporter_iso3']).margin_cif_pct
    keys = pd.MultiIndex.from_arrays([base.dest_iso3, base.origin_iso3])
    match = lookup.reindex(keys).to_numpy(float)
    # Never reverse importer/exporter to hide a missing directional estimate.
    fallback = float(margin_table.margin_cif_pct.median())
    base['margin_imputed'] = ~np.isfinite(match)
    base['margin_cif_pct'] = np.where(np.isfinite(match), match, fallback)
    base['freight_fob_fraction'] = base.margin_cif_pct/(100-base.margin_cif_pct)
    return base, arrays


def main_calculation(trade_path=None):
    OUT.mkdir(exist_ok=True)
    margins = pd.read_csv(OUT/'oecd_2024_primary_country_pair_margins.csv')
    combos = combinations()
    registry = pd.read_csv(ROOT/'data/chokepoints24/registry.csv')
    geo = pd.read_csv(OLD/'panel_c_country_region_mapping.csv').drop_duplicates('iso3').set_index('iso3')
    geo_map = geo.region.to_dict()
    records, sensitivity, mechanisms, coverage = [], [], [], []
    # Factorial stress tests, not samples from a fitted probability distribution.
    parameters = list(itertools.product([7., 30., 90.], [12., 16., 20.], [.02, .05, .10]))
    parameters += [(30., 16., .05, v, inv, mf)
                   for v in [.005, .01, .02] for inv in [.5, 1., 2.] for mf in [.75, 1., 1.25]]
    for month in range(1, 13):
        base, arrays = load_month(month, margins)
        freight = base.freight_fob_fraction.to_numpy(float)
        for design in ['fig3_fixed', 'common_five']:
            for ship in SHIPS:
                mask = base.shiptype.eq(ship).to_numpy()
                gates = CASES[ship] if design == 'fig3_fixed' else COMMON
                si = combos.index(gates)
                results = evaluate(base, arrays, si, freight)
                wo, wi, use, reroute, dc, da = results
                # Each row has two paired cases, never four copies of the same
                # bilateral annual trade total allocated separately by ship type.
                columns = ['od_id', 'shiptype', 'start_month', 'origin_port_id', 'dest_port_id',
                           'origin_iso3', 'dest_iso3', 'origin_lon', 'origin_lat', 'dest_lon', 'dest_lat',
                           'origin_name', 'dest_name', 'trade_weight_usd', 'sea_share_imputed',
                           'margin_imputed', 'margin_cif_pct', 'freight_fob_fraction', 'baseline_source',
                           'endpoint_closure_mask']
                for case, values in [('without_arctic', wo), ('with_arctic', wi)]:
                    frame = base.loc[mask, columns].copy()
                    frame['design'] = design
                    frame['case'] = case
                    frame['gates'] = '+'.join(registry.iloc[list(gates)]['key'])
                    frame['scenario_index'] = si
                    frame['endpoint_in_closure_mask'] = (frame.endpoint_closure_mask.to_numpy(dtype=np.uint32) & sum(1<<g for g in gates)) != 0
                    frame['dest_region'] = frame.dest_iso3.map(geo_map)
                    frame['origin_region'] = frame.origin_iso3.map(geo_map)
                    frame[CHANNELS] = values[mask]
                    frame['total_loss_usd'] = values[mask].sum(axis=1)
                    frame['open_km'] = arrays['without_km'][0, mask]
                    frame['conventional_km'] = dc[mask]
                    frame['arctic_km'] = da[mask]
                    frame['arctic_selected'] = use[mask] if case == 'with_arctic' else False
                    frame['waiting_selected'] = (~reroute[mask]) & (~frame.arctic_selected.to_numpy())
                    frame['disconnected_without'] = ~np.isfinite(dc[mask])
                    frame['disconnected_with'] = ~np.isfinite(dc[mask]) & ~np.isfinite(da[mask])
                    records.append(frame)
                if design == 'common_five':
                    for item in parameters:
                        duration, speed, premium = item[:3]
                        vot, inv, mf = item[3:] if len(item) == 6 else (.01, 1., 1.)
                        x, y, *_ = evaluate(base, arrays, si, freight, duration, speed, premium, vot, inv, mf)
                        a, b = x[mask].sum(), y[mask].sum()
                        sensitivity.append(dict(month=month, shiptype=ship, duration_days=duration,
                            speed_knots=speed, arctic_premium=premium, vot_per_day=vot,
                            inventory_scale=inv, margin_scale=mf, without_usd=a, with_usd=b,
                            avoided_pct=100*(a-b)/a if a else np.nan))
                # Record zero-marginal gates explicitly; do not silently portray
                # a fixed stress set as minimal in every vessel-month.
                bchosen, fullcost = cost_choice(dc, da)
                for gate in gates:
                    parent = combos.index(tuple(g for g in gates if g != gate))
                    pb, pcost = cost_choice(arrays['without_km'][parent], arrays['arctic_km'][parent])
                    finite = np.isfinite(fullcost) & np.isfinite(pcost)
                    diff = np.zeros(len(base))
                    diff[finite] = np.abs(fullcost[finite]-pcost[finite])
                    changed = (np.isfinite(fullcost) != np.isfinite(pcost)) | (diff > .01) | (bchosen != pb)
                    mechanisms.append(dict(design=design, shiptype=ship, month=month,
                        gate=registry.iloc[gate]['key'], changed_od=int((changed & mask).sum()),
                        changed_trade_pct=100*base.loc[changed & mask,'trade_weight_usd'].sum()/base.loc[mask,'trade_weight_usd'].sum()))
        coverage.append(dict(month=month, trade_od=len(base), trade_usd=base.trade_weight_usd.sum(),
            margin_fallback_od=int(base.margin_imputed.sum()),
            margin_fallback_trade_pct=100*base.loc[base.margin_imputed,'trade_weight_usd'].sum()/base.trade_weight_usd.sum(),
            sea_share_imputed_trade_pct=100*base.loc[base.sea_share_imputed.astype(bool),'trade_weight_usd'].sum()/base.trade_weight_usd.sum()))
        print(f'Month {month:02d}: {len(base)} covered OD rows evaluated', flush=True)
    od = pd.concat(records, ignore_index=True)
    od.to_parquet(OUT/'paired_od_losses.parquet', index=False)
    losses = CHANNELS+['total_loss_usd']
    monthly = od.groupby(['design','shiptype','start_month','case'])[losses].sum().reset_index()
    monthly.to_csv(OUT/'monthly_loss_channels.csv',index=False)
    country = od.groupby(['design','shiptype','dest_iso3','case'])[losses].sum()/12
    country.reset_index().to_csv(OUT/'country_mean_event_losses.csv',index=False)
    od.groupby(['design','shiptype','start_month','dest_iso3','case'])[losses].sum().reset_index().to_csv(
        OUT/'SI_country_monthly_losses.csv',index=False)
    region = od.groupby(['design','shiptype','dest_region','case'])[losses].sum()/12
    region.reset_index().to_csv(OUT/'region_mean_event_losses.csv',index=False)
    port = od.groupby(['design','shiptype','dest_port_id','dest_name','dest_iso3','dest_lon','dest_lat','case'])[losses].sum()/12
    port.reset_index().to_csv(OUT/'port_mean_event_losses.csv',index=False)
    od.groupby(['design','shiptype','origin_iso3','case'])[losses].sum().div(12).reset_index().to_csv(
        OUT/'SI_origin_attribution_mean_losses.csv', index=False)
    od.groupby(['design','shiptype','start_month','case','endpoint_in_closure_mask'])[
        ['trade_weight_usd']+losses].sum().reset_index().to_csv(OUT/'SI_endpoint_closure_attribution.csv',index=False)
    pd.DataFrame(sensitivity).drop_duplicates().to_csv(OUT/'SI_parameter_sensitivity.csv',index=False)
    pd.DataFrame(mechanisms).to_csv(OUT/'SI_marginal_gate_effects.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage_audit.csv',index=False)
    brief = monthly.groupby(['design','shiptype','case'])[losses].sum().div(12).reset_index()
    pivot = brief.pivot(index=['design','shiptype'],columns='case',values='total_loss_usd').reset_index()
    pivot['avoided_usd'] = pivot.without_arctic-pivot.with_arctic
    pivot['avoided_pct'] = 100*pivot.avoided_usd/pivot.without_arctic
    pivot.to_csv(OUT/'summary.csv',index=False)
    print(pivot.to_string(index=False),flush=True)
    trade_path = trade_path or OLD/'panel_c_trade_weighted_od.parquet'
    source = pd.read_parquet(trade_path)
    covered = source[source.selection_status.eq('included')]
    main = od[od.design.eq('fig3_fixed') & od.case.eq('without_arctic')]
    assert len(main)==len(covered) and main.od_id.is_unique
    assert np.isclose(main.trade_weight_usd.sum(),covered.trade_weight_usd.sum(),rtol=1e-12)
    paired = od.pivot(index=['design','od_id'],columns='case',values='total_loss_usd')
    assert ((paired.with_arctic-paired.without_arctic)<1e-5).all()
    assert np.isfinite(od[losses]).all().all() and od[losses].ge(0).all().all()
    assert np.allclose(od[CHANNELS].sum(axis=1),od.total_loss_usd)
    assert main.dest_region.notna().all()
    pair_totals = main.groupby(['origin_iso3','dest_iso3']).trade_weight_usd.sum()
    targets = covered.groupby(['origin_iso3','dest_iso3']).maritime_proxy_2024_usd.first()
    assert np.allclose(pair_totals.sort_index(),targets.sort_index(),rtol=1e-12)
    inputs = [trade_path, OUT/'oecd_2024_primary_country_pair_margins.csv',
              NET/'physical_edges.parquet', ROOT/'data/cmip6_monthly_repair/monthly_four_route_paths.csv']
    audit = dict(status='PASS', input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        covered_od=len(main), represented_ports=len(set(main.origin_port_id)|set(main.dest_port_id)),
        destination_countries=main.dest_iso3.nunique(), annual_allocated_trade_usd=main.trade_weight_usd.sum(),
        maximum_pair_allocation_error_usd=float(np.abs(pair_totals-targets).max()),
        maximum_pair_benefit_violation_usd=float((paired.with_arctic-paired.without_arctic).max()),
        months=sorted(main.start_month.unique().astype(int).tolist()),
        unit='USD per conditional 30-day event; map summaries are means of 12 monthly cases',
        joint_event_probabilities='Not estimated; stress cases are not additive',
        separate_war_risk_premium='Not identified; not fabricated or added',
        SI_sensitivity='Unweighted deterministic parameter sweep, not confidence intervals')
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['routes','margins','public-margins','compute'])
    args=parser.parse_args()
    {'routes':prepare_routes,'margins':prepare_margins,'public-margins':public_margins,'compute':main_calculation}[args.stage]()
