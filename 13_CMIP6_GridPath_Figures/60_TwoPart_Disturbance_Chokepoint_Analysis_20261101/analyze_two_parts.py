"""Connect heterogeneous disturbances to a separate chokepoint experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
HET = ROOT / '58_FourClass_Threshold_Analysis_20261101'
PURE = ROOT / '28_MainFig3_FiveClosures'
OUT = TASK / 'results'
SHIPS = ['container', 'bulk', 'tanker', 'lng']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def split_gate_weights(masks, weights):
    """Assign each OD once, splitting multi-gate support; retain no-gate ODs."""
    weights = np.asarray(weights, dtype=float)
    assert len(masks) == len(weights) and np.all(np.isfinite(weights)) and np.all(weights >= 0)
    incidence = (np.asarray(masks, np.uint32)[:, None] & (1 << np.arange(24, dtype=np.uint32))) != 0
    counts = incidence.sum(axis=1)
    allocated = np.sum(weights[:, None] * incidence / np.maximum(counts[:, None], 1), axis=0)
    allocated = np.r_[allocated, np.sum(weights[counts == 0])]
    assert np.isclose(allocated.sum(), np.sum(weights), atol=1e-4, rtol=1e-10)
    return allocated


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(ROOT / 'data/chokepoints24/registry.csv')
    names = registry.name.tolist() + ['No listed chokepoint']
    keys = registry.key.tolist() + ['NONE_24']
    combos = json.loads((PURE / 'scenarios.json').read_text())
    assert len(combos) == 55455 and all(len(c) <= 5 for c in combos)
    indices = {tuple(c): j for j, c in enumerate(combos)}
    masks = np.array([sum(1 << j for j in c) for c in combos], np.uint32)
    counts = np.array([len(c) for c in combos])
    parent = np.zeros((len(combos), 5), np.int32)
    slot_valid = np.zeros(parent.shape, bool)
    for si, combo in enumerate(combos):
        for j in range(len(combo)):
            parent[si, j] = indices[tuple(combo[:j] + combo[j + 1:])]
            slot_valid[si, j] = True
    manifests = [json.loads((PURE / ('month_%02d' % m) / 'input_manifest.json').read_text()) for m in range(1, 13)]
    reference = manifests[0]['inputs']
    assert all(m['inputs'] == reference and m['premium'] == .05 and m['max_closures'] == 5 for m in manifests)
    for name, expected in reference.items():
        assert digest(ROOT / name) == expected, 'Changed main-search input: ' + name
    for m in range(1, 13):
        ha = json.loads((HET / 'results' / ('month_%02d' % m) / 'audit.json').read_text())
        assert ha['status'] == 'PASS'
        for name, value in reference.items():
            if name in ha['inputs']:
                assert ha['inputs'][name] == value
    data = pd.read_csv(HET / 'results/four_class_monthly_shares.csv')
    assert len(data) == 1152
    gate_rows, country_rows, service_rows, core_rows = [], [], [], []
    for month in range(1, 13):
        folder = HET / 'results' / ('month_%02d' % month)
        base = pd.read_parquet(folder / 'demand.parquet')
        arrays = np.load(folder / 'od_states.npz')
        catalogue = np.load(folder / 'conventional_paths.npz')
        orig_mask = catalogue['gate_mask'][arrays['conventional_id'][0]] & np.uint32((1 << 24) - 1)
        for row in data[data.month == month].itertuples():
            i = row.state_row
            take = base.shiptype.eq(row.shiptype).to_numpy()
            w = base.trade_weight_usd.to_numpy()
            total = np.sum(w[take])
            start = i - row.stage
            previous = i - 1 if row.stage else i
            new = take & arrays['B'][i] & ~arrays['B'][start]
            lost = take & arrays['B'][start] & ~arrays['B'][i]
            incoming = take & arrays['B'][i] & ~arrays['B'][previous]
            outgoing = take & arrays['B'][previous] & ~arrays['B'][i]
            new_value = float(w[new].sum())
            gate_value = split_gate_weights(orig_mask[new], w[new])
            for gate in range(25):
                gate_rows.append(dict(class_id=row.class_id, shiptype=row.shiptype, month=month, stage=row.stage,
                    gate=keys[gate], gate_name=names[gate], allocated_weight_usd=float(gate_value[gate]),
                    original_weight_usd=float(total), new_Arctic_weight_usd=new_value,
                    share_original_trade_pct=100 * gate_value[gate] / total,
                    share_new_Arctic_pct=100 * gate_value[gate] / new_value if new_value > 0 else np.nan))
            u = take & arrays['U'][i]
            denied = u & ~arrays['service_allowed'][i]
            physical = u & arrays['service_allowed'][i]
            service_rows.append(dict(class_id=row.class_id, shiptype=row.shiptype, month=month, stage=row.stage,
                original_weight_usd=float(total), new_B_pct=100 * new_value / total,
                lost_baseline_B_pct=100 * float(w[lost].sum()) / total,
                marginal_in_B_pct=100 * float(w[incoming].sum()) / total,
                marginal_out_B_pct=100 * float(w[outgoing].sum()) / total,
                denied_port_U_pct=100 * float(w[denied].sum()) / total,
                path_or_ice_U_pct=100 * float(w[physical].sum()) / total))
            for route, code in zip(['NWP', 'NEP', 'NSR', 'TSR'], range(1, 5)):
                use = new & (arrays['route_code'][i] == code)
                core_rows.append(dict(class_id=row.class_id, shiptype=row.shiptype, month=month,
                    stage=row.stage, route=route, new_B_weight_usd=float(w[use].sum()), original_weight_usd=float(total)))
            if row.stage == 5:
                g = base[['origin_iso3', 'dest_iso3']].copy()
                for name, flag in [('new_B', new), ('A', take & arrays['A'][i]), ('B', take & arrays['B'][i]), ('U', u)]:
                    g[name + '_weight_usd'] = w * flag
                g['original_weight_usd'] = w * take
                g = g.groupby(['origin_iso3', 'dest_iso3'], as_index=False).sum(numeric_only=True)
                g = g[g.original_weight_usd > 0]
                g['class_id'], g['shiptype'], g['month'] = row.class_id, row.shiptype, month
                country_rows.append(g)
    gates = pd.DataFrame(gate_rows)
    flow = pd.DataFrame(service_rows)
    data = data.merge(flow.drop(columns='original_weight_usd'), on=['class_id', 'shiptype', 'month', 'stage'], validate='one_to_one')
    assert np.allclose(data.U_pct, data.denied_port_U_pct + data.path_or_ice_U_pct, atol=1e-8)
    assert np.allclose(data.B_pct - data.B0_pct, data.new_B_pct - data.lost_baseline_B_pct, atol=1e-8)
    assert np.allclose(data.new_B_pct, data.new_Arctic_flow_pct, atol=1e-8)
    assert np.allclose(data.marginal_Arctic_share_pp, data.marginal_in_B_pct - data.marginal_out_B_pct, atol=1e-8)
    data.to_csv(OUT / 'partI_monthly_mechanism_decomposition.csv', index=False)
    gates.to_csv(OUT / 'partII_new_Arctic_OD_baseline_gate_dependency.csv', index=False)
    pd.concat(country_rows).to_csv(OUT / 'partI_stage5_country_OD_results.csv', index=False)
    pd.DataFrame(core_rows).to_csv(OUT / 'partI_new_Arctic_four_passage_composition.csv', index=False)
    stage5 = gates[gates.stage == 5]
    gg = stage5.groupby(['class_id', 'shiptype', 'gate', 'gate_name'], as_index=False).agg(
        allocated_weight_usd=('allocated_weight_usd', 'sum'), original_weight_usd=('original_weight_usd', 'sum'),
        new_Arctic_weight_usd=('new_Arctic_weight_usd', 'sum'))
    gg['share_original_trade_pct'] = 100 * gg.allocated_weight_usd / gg.original_weight_usd
    gg['share_new_Arctic_pct'] = 100 * gg.allocated_weight_usd / gg.new_Arctic_weight_usd
    gg.to_csv(OUT / 'partII_stage5_annual_gate_dependency.csv', index=False)
    eligible_counts, selections, increments, source_audits, envelopes = [], [], [], [], []
    heatmap = np.zeros((4, 80, 80), np.int64)
    bins_p = np.linspace(0, 100, 81)
    bins_gap = np.linspace(-100, 100, 81)
    analog_rows = []
    for month in range(1, 13):
        folder = PURE / ('month_%02d' % month)
        audit = json.loads((folder / 'calculation_audit.json').read_text())
        assert audit['status'] == 'PASS' and audit['scenarios'] == len(combos)
        source_audits.append(audit)
        frame = pd.read_parquet(folder / 'scenario_metrics.parquet')
        frame = frame[frame.scope == 'trade_covered']
        covered = pd.read_parquet(folder / 'demand.parquet')
        covered = covered[covered.trade_weight_usd.notna() & (covered.trade_weight_usd > 0)].set_index('od_id')
        current = pd.read_parquet(HET / 'results' / ('month_%02d' % month) / 'demand.parquet').set_index('od_id')
        assert set(covered.index) == set(current.index)
        assert np.allclose(covered.loc[current.index, 'trade_weight_usd'], current.trade_weight_usd)
        for vi, ship in enumerate(SHIPS):
            g = frame[frame.shiptype == ship].sort_values('scenario_index').reset_index(drop=True)
            assert np.array_equal(g.scenario_index, np.arange(len(combos)))
            assert np.array_equal(g.n_closed, counts)
            assert np.allclose(g.A_weight_pct + g.B_weight_pct + g.unresolved_pct, 100, atol=1e-8)
            before = data[(data.shiptype == ship) & (data.month == month) & (data.stage == 0)].iloc[0]
            assert np.allclose([g.iloc[0].A_weight_pct, g.iloc[0].B_weight_pct], [before.A_pct, before.B_pct], atol=1e-7)
            valid = g.nonredundant.to_numpy(bool)
            gap = (g.B_weight_pct - g.A_weight_pct).to_numpy()
            heatmap[vi] += np.histogram2d(g.p_exposed_pct[valid], gap[valid], bins=[bins_p, bins_gap])[0].astype(np.int64)
            for k in range(1, 6):
                raw = g[g.n_closed == k]
                candidates = raw[raw.nonredundant]
                cross = candidates[candidates.crossover]
                eligible_counts.append(dict(month=month, shiptype=ship, n_closed=k,
                    raw_combinations=len(raw), nonredundant_combinations=len(candidates), crossover_combinations=len(cross)))
                chosen = cross if len(cross) else candidates
                if len(chosen):
                    chosen = chosen.assign(gap_pp=chosen.B_weight_pct - chosen.A_weight_pct)
                    sort = ['p_exposed_pct', 'unresolved_pct', 'gap_pp', 'scenario_index'] if len(cross) else ['gap_pp', 'unresolved_pct', 'p_exposed_pct', 'scenario_index']
                    asc = [True, True, False, True] if len(cross) else [False, True, True, True]
                    r = chosen.sort_values(sort, ascending=asc).iloc[0].to_dict()
                    r['selection_rule'] = 'minimum_exposure_among_dominant' if len(cross) else 'maximum_gap_no_dominance'
                    selections.append(r)
                rows = np.flatnonzero((counts == k) & valid)
                if len(rows):
                    delta = g.B_weight_pct.to_numpy()[rows, None] - g.B_weight_pct.to_numpy()[parent[rows]]
                    ok = slot_valid[rows] & valid[parent[rows]]
                    delta[~ok] = -np.inf
                    maximum = np.max(delta)
                    if np.isfinite(maximum):
                        ii, jj = np.where(np.abs(delta - maximum) < 1e-10)
                        order = np.lexsort((jj, rows[ii], g.p_exposed_pct.to_numpy()[rows[ii]]))
                        ci, slot = int(rows[ii[order[0]]]), int(jj[order[0]])
                        pi = int(parent[ci, slot])
                        increments.append(dict(shiptype=ship, month=month, n_closed=k, child_index=ci, parent_index=pi,
                            added_gate=registry.iloc[combos[ci][slot]].key, delta_B_pp=float(delta[ii[order[0]], slot]),
                            child_p_pct=float(g.iloc[ci].p_exposed_pct), parent_p_pct=float(g.iloc[pi].p_exposed_pct),
                            child_A_pct=float(g.iloc[ci].A_weight_pct), child_B_pct=float(g.iloc[ci].B_weight_pct),
                            child_U_pct=float(g.iloc[ci].unresolved_pct), parent_B_pct=float(g.iloc[pi].B_weight_pct)))
            # Compare responses at similar exposure without calling them equivalent interventions.
            for r in data[(data.shiptype == ship) & (data.month == month) & (data.stage > 0)].itertuples():
                close = g[(g.p_exposed_pct - r.p_exposed_pct).abs() <= 2.0]
                close = close[close.nonredundant]
                entry = dict(class_id=r.class_id, shiptype=ship, month=month, stage=r.stage,
                    heterogeneous_p_pct=r.p_exposed_pct, heterogeneous_gap_pp=r.B_pct - r.A_pct,
                    heterogeneous_U_pct=r.U_pct, exposure_window_pp=2.0, nearby_closure_sets=len(close))
                if len(close):
                    cg = close.B_weight_pct - close.A_weight_pct
                    entry.update(closure_gap_min_pp=float(cg.min()), closure_gap_max_pp=float(cg.max()),
                        closure_U_min_pct=float(close.unresolved_pct.min()), closure_U_max_pct=float(close.unresolved_pct.max()),
                        gap_within_descriptive_range=bool(cg.min() - 1e-8 <= r.B_pct - r.A_pct <= cg.max() + 1e-8))
                analog_rows.append(entry)
        print('AGGREGATE MONTH %02d COMPLETE' % month, flush=True)
    pd.DataFrame(eligible_counts).to_csv(OUT / 'partII_closure_search_counts.csv', index=False)
    candidates = pd.DataFrame(selections)
    selected = []
    for (_, _), g in candidates.groupby(['shiptype', 'n_closed']):
        cross = g[g.crossover]
        take = cross if len(cross) else g
        if len(cross):
            r = take.sort_values(['p_exposed_pct', 'unresolved_pct', 'gap_pp', 'month', 'scenario_index'], ascending=[True, True, False, True, True]).iloc[0].copy()
        else:
            r = take.sort_values(['gap_pp', 'unresolved_pct', 'p_exposed_pct', 'month', 'scenario_index'], ascending=[False, True, True, True, True]).iloc[0].copy()
        r['closed_gates'] = '|'.join(registry.iloc[j].key for j in combos[int(r.scenario_index)])
        r['closed_names'] = ' + '.join(registry.iloc[j]['name'] for j in combos[int(r.scenario_index)])
        selected.append(r)
    selected = pd.DataFrame(selected)
    selected.to_csv(OUT / 'partII_independent_dominance_selection.csv', index=False)
    inc = pd.DataFrame(increments).sort_values(['shiptype', 'n_closed', 'delta_B_pp', 'child_p_pct', 'month', 'child_index', 'parent_index'], ascending=[True, True, False, True, True, True, True])
    inc.to_csv(OUT / 'partII_monthly_maximum_nested_increments.csv', index=False)
    inc = inc.drop_duplicates(['shiptype', 'n_closed']).copy()
    for col in ['child', 'parent']:
        inc[col + '_closed_gates'] = ['|'.join(registry.iloc[j].key for j in combos[int(i)]) for i in inc[col + '_index']]
    inc.to_csv(OUT / 'partII_maximum_nested_increment_selection.csv', index=False)
    pd.DataFrame(analog_rows).to_csv(OUT / 'SI_similar_exposure_response_ranges_NOT_equivalent_shocks.csv', index=False)
    np.savez_compressed(OUT / 'partII_closure_response_histogram.npz', counts=heatmap, p_edges=bins_p, gap_edges=bins_gap)
    replay = set()
    for r in selected.itertuples():
        replay.add((int(r.month), int(r.scenario_index)))
    for r in inc.itertuples():
        replay.add((int(r.month), int(r.child_index)))
        replay.add((int(r.month), int(r.parent_index)))
    replay.update((m, 0) for m, _ in list(replay))
    requests = [dict(month=m, scenario_index=i, closed_mask=int(masks[i]), gates=[registry.iloc[j].key for j in combos[i]]) for m, i in sorted(replay)]
    (OUT / 'selected_replay_requests.json').write_text(json.dumps(requests, indent=2))
    audit = dict(status='AGGREGATE_AND_OD_DEPENDENCY_CHECKS_PASS', design='two distinct stages; heterogeneous operators retained',
        main_input_sha256=reference, heterogeneous_source_sha256=digest(HET / 'scenarios.json'),
        original_trade_OD_records=16233, heterogeneous_ship_month_stage_rows=len(data),
        raw_nonempty_closure_sets=55454, months=12, shiptypes=4,
        no_new_global_route_search_claim=True, cached_main_search_verified_against_current_inputs=True,
        baseline_dependency_attribution='Equal weight split across baseline conventional path chokepoints; no listed gate retained',
        annual_dependency_scope='Stage5 per class pooled across 12 months; no pooling or duplication across classes',
        selected_replay_states=len(requests), selection_uncertainty='Outcome-selected cases; subsequent bootstrap is fixed-case sensitivity, not search-adjusted inference',
        controls='SSP585 2091-2100; PC6-core; common >=50% support; full-voyage 5% distance-equivalent premium')
    (OUT / 'aggregate_audit.json').write_text(json.dumps(audit, indent=2))
    print(selected[['shiptype', 'n_closed', 'month', 'p_exposed_pct', 'A_weight_pct', 'B_weight_pct', 'unresolved_pct', 'switched_to_B_pct', 'closed_gates']].to_string(index=False), flush=True)
    print('REPLAY STATES', len(requests), flush=True)


if __name__ == '__main__':
    main()
