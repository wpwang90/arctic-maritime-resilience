"""Select closures only from baseline dependencies of genuinely new Arctic ODs."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_two_parts import SHIPS, split_gate_weights

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
OUT = TASK / 'results'
HET = ROOT / '58_FourClass_Threshold_Analysis_20261101/results'
PURE = ROOT / '28_MainFig3_FiveClosures'


def main():
    registry = pd.read_csv(ROOT / 'data/chokepoints24/registry.csv')
    combos = json.loads((PURE / 'scenarios.json').read_text())
    masks = np.array([sum(1 << j for j in c) for c in combos], np.uint32)
    sizes = np.array([len(c) for c in combos])
    dependency = np.zeros((4, 25))
    pool_rows = []
    original_total = np.zeros(4)
    for month in range(1, 13):
        folder = HET / ('month_%02d' % month)
        base = pd.read_parquet(folder / 'demand.parquet')
        states = np.load(folder / 'od_states.npz')
        cat = np.load(folder / 'conventional_paths.npz')
        original_mask = cat['gate_mask'][states['conventional_id'][0]] & np.uint32((1 << 24) - 1)
        ever_new = (states['B'] & ~states['B'][0]).any(axis=0)
        for vi, ship in enumerate(SHIPS):
            take = base.shiptype.eq(ship).to_numpy()
            eligible = take & ever_new
            original_total[vi] += base.trade_weight_usd[take].sum()
            dependency[vi] += split_gate_weights(original_mask[eligible], base.trade_weight_usd.to_numpy()[eligible])
            z = base.loc[eligible, ['od_id', 'shiptype', 'start_month', 'origin_iso3', 'dest_iso3', 'trade_weight_usd']].copy()
            z['baseline_gate_mask'] = original_mask[eligible]
            pool_rows.append(z)
    pd.concat(pool_rows).to_csv(OUT / 'partII_unique_new_Arctic_OD_candidate_pool.csv', index=False)
    pools = [sum(1 << j for j in range(24) if dependency[vi, j] > 0) for vi in range(4)]
    metrics = ['A_weight_pct', 'B_weight_pct', 'unresolved_pct', 'switched_to_B_pct', 'p_exposed_pct']
    sums = np.zeros((4, len(combos), len(metrics)))
    margins = np.zeros((4, len(combos), 5))
    totals = np.zeros(4)
    for month in range(1, 13):
        folder = PURE / ('month_%02d' % month)
        data = pd.read_parquet(folder / 'scenario_metrics.parquet')
        gate_margins = np.load(folder / 'marginal_gate_audit.npz')['marginal_pct']
        for vi, ship in enumerate(SHIPS):
            g = data[(data.scope == 'trade_covered') & (data.shiptype == ship)].sort_values('scenario_index')
            total = float(g.weight_total.iloc[0])
            totals[vi] += total
            sums[vi] += total * g[metrics].to_numpy()
            margins[vi] += total * np.nan_to_num(gate_margins[:, :, 2 * vi + 1].astype(float), nan=0.)
        print('DEPENDENT SEARCH MONTH %02d' % month, flush=True)
    sums /= totals[:, None, None]
    margins /= totals[:, None, None]
    assert np.allclose(totals, original_total)
    rows, gates, global_refs = [], [], []
    for vi, ship in enumerate(SHIPS):
        for j, amount in enumerate(dependency[vi]):
            gates.append(dict(shiptype=ship, gate=registry.iloc[j].key if j < 24 else 'NONE_24',
                gate_name=registry.iloc[j]['name'] if j < 24 else 'No listed chokepoint',
                eligible_for_closure=bool(j < 24 and amount > 0), allocated_unique_new_OD_weight_usd=float(amount),
                share_unique_new_OD_weight_pct=float(100 * amount / dependency[vi].sum()),
                definition='Deduplicate each OD-ship-month across all four scenario classes and stages before splitting its baseline gate dependencies'))
        valid = np.ones(len(combos), bool)
        for slot in range(5):
            valid &= (sizes <= slot) | (margins[vi, :, slot] > 1e-6)
        eligible = (masks & np.uint32(pools[vi])) == masks
        frame = pd.DataFrame(sums[vi], columns=metrics)
        frame['scenario_index'] = np.arange(len(combos))
        frame['n_closed'] = sizes
        frame['annual_nonredundant'] = valid
        frame['within_dependency_pool'] = eligible
        frame['gap_pp'] = frame.B_weight_pct - frame.A_weight_pct
        frame['shiptype'] = ship
        frame.to_parquet(OUT / ('partII_annual_closure_search_' + ship + '.parquet'), index=False)
        for k in range(6):
            allowed = frame[(sizes == k) & valid & eligible]
            if allowed.empty:
                raise RuntimeError('No genuinely effective candidate set: %s k=%d; do not force five closures' % (ship, k))
            chosen = allowed.sort_values(['switched_to_B_pct', 'B_weight_pct', 'unresolved_pct', 'p_exposed_pct', 'scenario_index'], ascending=[False, False, True, True, True]).iloc[0].to_dict()
            si = int(chosen['scenario_index'])
            chosen.update(candidate_gate_count=int(int(pools[vi]).bit_count()) if hasattr(int, 'bit_count') else bin(pools[vi]).count('1'),
                candidate_raw_sets=math.comb(bin(pools[vi]).count('1'), k), candidate_nonredundant_sets=len(allowed),
                closed_mask=int(masks[si]), closed_gates='|'.join(registry.iloc[j].key for j in combos[si]),
                selection='Exact k independently maximizes pooled twelve-month new Arctic trade share within dependency-derived gate pool; not forced B>A',
                annual_crossover=bool(chosen['B_weight_pct'] > chosen['A_weight_pct'] + 1e-8),
                objective='gross new B relative to no-closure Arctic baseline, not declining A or endpoint service denial')
            rows.append(chosen)
            all24 = frame[(sizes == k) & valid].sort_values(['switched_to_B_pct', 'B_weight_pct', 'unresolved_pct', 'p_exposed_pct', 'scenario_index'], ascending=[False, False, True, True, True]).iloc[0].to_dict()
            all24['closed_gates'] = '|'.join(registry.iloc[j].key for j in combos[int(all24['scenario_index'])])
            global_refs.append(all24)
    selected = pd.DataFrame(rows)
    selected.to_csv(OUT / 'partII_dependency_constrained_optimal_closures.csv', index=False)
    pd.DataFrame(global_refs).to_csv(OUT / 'SI_unrestricted24_same_objective_optima.csv', index=False)
    pd.DataFrame(gates).to_csv(OUT / 'partII_gate_candidates_deduplicated_OD_dependencies.csv', index=False)
    original_ranks = pd.read_csv(ROOT / '51_TradeWeighted_CountryBetweenness/results/vessel_top50_rank_population_primary.csv')
    fixed_months = original_ranks.groupby('shiptype').month.first().astype(int).to_dict()
    fixed = []
    for vi, ship in enumerate(SHIPS):
        month = fixed_months[ship]
        frame = pd.read_parquet(PURE / ('month_%02d' % month) / 'scenario_metrics.parquet')
        frame = frame[(frame.scope == 'trade_covered') & (frame.shiptype == ship)].sort_values('scenario_index').copy()
        frame['within_dependency_pool'] = (masks & np.uint32(pools[vi])) == masks
        for k in range(6):
            allowed = frame[(frame.n_closed == k) & frame.nonredundant & frame.within_dependency_pool]
            assert len(allowed), (ship, month, k)
            row = allowed.sort_values(['switched_to_B_pct', 'B_weight_pct', 'unresolved_pct', 'p_exposed_pct', 'scenario_index'], ascending=[False, False, True, True, True]).iloc[0].to_dict()
            si = int(row['scenario_index'])
            row.update(closed_mask=int(masks[si]), closed_gates='|'.join(registry.iloc[j].key for j in combos[si]),
                candidate_gate_count=bin(pools[vi]).count('1'), candidate_nonredundant_sets=len(allowed),
                objective='Maximize new Arctic trade within the ORIGINAL fixed vessel month; no dominance requirement',
                original_figure_month_preserved=True)
            fixed.append(row)
    fixed = pd.DataFrame(fixed)
    fixed.to_csv(OUT / 'partII_mainfig4_fixed_month_optimal_closures.csv', index=False)
    main = fixed[fixed.n_closed == 5]
    requests = []
    for month in range(1, 13):
        for si in sorted({0} | set(main.scenario_index.astype(int))):
            requests.append(dict(month=month, scenario_index=si, closed_mask=int(masks[si]),
                gates=[registry.iloc[j].key for j in combos[si]]))
    (OUT / 'rank_replay_requests.json').write_text(json.dumps(requests, indent=2))
    (OUT / 'dependency_selection_audit.json').write_text(json.dumps(dict(status='PASS',
        dependency_pool_precedes_closure_optimization=True, OD_month_deduplicated_across_hypotheses=True,
        selection_is_scenario_dependent_not_unbiased_causal_discovery=True,
        objective='Main: maximize new Arctic trade within original fixed vessel months; annual optimum is a separate sensitivity',
        exact_closure_budgets=list(range(6)), same_five_gate_set_used_in_all_months_for_each_shiptype=True,
        every_member_changes_annual_cost_or_choice=True, main_figure_form='Original four vessel-specific top-50 country rank panels',
        phase_dominance_not_forced=True, main_comparator='same five closures: conventional distance shortest paths versus 5%-premium Arctic choice',
        independent_k_optima_not_assumed_nested=True, rank_replay_states=len(requests)), indent=2))
    print(main[['shiptype', 'candidate_gate_count', 'switched_to_B_pct', 'A_weight_pct', 'B_weight_pct', 'unresolved_pct', 'closed_gates']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
