"""Select one maximum-dominance response per disturbance class, without altering inputs."""
import calendar
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import shapely

import plot_partI_maps as maps

TASK, ROOT, OUT, FIG, HET = maps.TASK, maps.ROOT, maps.OUT, maps.FIG, maps.HET
CLASSES = ['T1', 'T2', 'T3', 'T4']
CLASS_NAMES = {
    'T1': 'Security / conflict', 'T2': 'Weather / hydrology',
    'T3': 'Coastal access exclusion', 'T4': 'Infrastructure / logistics',
}
METRICS = ['baseline_km', 'traditional_km', 'embedded_km']
RULE = ('Among stages 1-5, four vessel types and twelve months with A0>B0, maximize '
        'B-A using the fixed original trade denominator; ties: lower U, larger gross '
        'new B, earlier stage, earlier month, alphabetical shiptype. This is a '
        'descriptive extreme-response selection, not a representative or inferential estimate.')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def choose_cases(data):
    candidates = data[data.stage.between(1, 5) & data.A0_pct.gt(data.B0_pct)].copy()
    candidates['gap_pp'] = candidates.B_pct - candidates.A_pct
    candidates = candidates.sort_values(
        ['class_id', 'gap_pp', 'U_pct', 'new_B_pct', 'stage', 'month', 'shiptype'],
        ascending=[True, False, True, False, True, True, True], kind='mergesort')
    cases = candidates.groupby('class_id', sort=False).head(1).copy()
    cases['selection_rule'] = RULE
    cases['selection_status'] = np.where(cases.gap_pp > 0, 'Arctic dominance', 'Closest; no dominance')
    cases['eligible_cases_in_class'] = cases.class_id.map(candidates.groupby('class_id').size())
    candidates.to_csv(OUT / 'SI_partI_all_extreme_selection_candidates.csv', index=False)
    return cases.sort_values('class_id').reset_index(drop=True)


def country_distances(case):
    folder = HET / 'results' / ('month_%02d' % int(case.month))
    base = pd.read_parquet(folder / 'demand.parquet')
    with np.load(folder / 'od_states.npz') as arrays:
        si = int(case.state_row)
        d0 = arrays['without_km'][0]
        da = arrays['without_km'][si]
        db = np.where(arrays['B'][si], arrays['arctic_km'][si], da)
        take = base.shiptype.eq(case.shiptype).to_numpy()
        weights = base.trade_weight_usd.to_numpy()
        observed = {}
        for key in ['A', 'B', 'U']:
            observed[key + '_pct'] = float(100 * weights[take & arrays[key][si]].sum() / weights[take].sum())
            assert np.isclose(observed[key + '_pct'], case[key + '_pct'], atol=1e-8)
        assert not np.any(arrays['B'][si] & arrays['U'][si])
        assert np.all(1.05 * arrays['arctic_km'][si, arrays['B'][si]] < arrays['without_km'][si, arrays['B'][si]] - 1e-6)
        support = np.array([int(value).bit_count() for value in arrays['joint_ice'][si, arrays['B'][si]]])
        assert len(support) == 0 or support.min() >= 30
        # A common finite-distance cohort avoids fabricating distances for unserved ODs.
        matched = take & arrays['service_allowed'][si] & np.isfinite(d0) & np.isfinite(da) & np.isfinite(db)
        assert not np.any(matched & arrays['U'][si])
        frame = base.loc[matched, ['origin_iso3', 'trade_weight_usd']].copy()
        for key, value in zip(METRICS, [d0, da, db]):
            frame[key] = value[matched] * frame.trade_weight_usd
        sums = frame.groupby('origin_iso3').sum(numeric_only=True)
        totals = sums.trade_weight_usd.copy()
        for key in METRICS:
            sums[key] /= totals
        sums['matched_trade_usd'] = totals
        sums['matched_od_count'] = frame.groupby('origin_iso3').size()
        sums = sums.reset_index().rename(columns={'origin_iso3': 'iso3'})
        for key in ['class_id', 'shiptype', 'month', 'stage']:
            sums[key] = case[key]
        diagnostic = dict(
            class_id=case.class_id, shiptype=case.shiptype, month=int(case.month), stage=int(case.stage),
            matched_trade_pct=float(100 * weights[matched].sum() / weights[take].sum()),
            full_od_count=int(take.sum()), matched_od_count=int(matched.sum()),
            minimum_joint_ice_support=int(support.min()) if len(support) else None,
            original_weight_usd=float(weights[take].sum()),
            input_sha256={name: sha256(folder / name) for name in ['demand.parquet', 'od_states.npz']},
            source_shares_verified=observed, finite_distance_not_imputed_for_unserved=True)
    return sums, diagnostic


def draw_share_panel(ax, data, case, letter, bottom_row):
    values = data[(data.class_id == case.class_id) & (data.shiptype == case.shiptype) &
                  (data.month == case.month)].sort_values('stage')
    assert values.stage.tolist() == list(range(6))
    bottom = np.zeros(6)
    for key in ['A', 'B', 'U']:
        ax.bar(range(6), values[key + '_pct'], bottom=bottom, width=.64,
               color=maps.COLORS[key], edgecolor='white', linewidth=.45)
        bottom += values[key + '_pct'].to_numpy()
    # Outline the state used for the maps; retain every stage, including reversals.
    ax.add_patch(Rectangle((int(case.stage) - .36, -.4), .72, 100.8,
                           fill=False, edgecolor='#303f48', linewidth=1.35, clip_on=False))
    dominant = values[values.B_pct.gt(values.A_pct)]
    if len(dominant):
        ax.scatter(dominant.stage, np.full(len(dominant), 104.), marker='*', s=35,
                   c=maps.COLORS['B'], edgecolors='white', linewidth=.3, clip_on=False)
    ax.set(xlim=(-.6, 5.6), ylim=(0, 110), xticks=range(6), yticks=[0, 50, 100], ylabel='Covered trade (%)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#7d8b90')
    ax.tick_params(labelsize=8, width=.5, length=3)
    ax.yaxis.label.set_fontsize(8.5)
    for label, stage in zip(ax.get_xticklabels(), range(6)):
        if stage == int(case.stage):
            label.set_weight('bold')
    if bottom_row:
        ax.set_xlabel('Cumulative scenario stage', fontsize=8.5)
    ax.text(-.16, 1.04, letter, transform=ax.transAxes, fontsize=11, weight='bold')


def plot_selection_landscape(data, cases):
    """Expose all 960 tested candidates so the main-figure selection is auditable."""
    cmap = LinearSegmentedColormap.from_list('dominance_gap', ['#345e7d', '#8db7c8', '#f7f7f2', '#edbe85', '#c76b36'])
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    fig, axes = plt.subplots(4, 4, figsize=(13.8, 12.7), sharex=True, sharey=True)
    for ri, cls in enumerate(CLASSES):
        for ci, ship in enumerate(maps.SHIPS):
            ax = axes[ri, ci]
            sub = data[(data.class_id == cls) & (data.shiptype == ship) & data.stage.between(1, 5)].copy()
            sub['gap'] = sub.B_pct - sub.A_pct
            grid = sub.pivot(index='month', columns='stage', values='gap').reindex(index=range(1, 13), columns=range(1, 6))
            im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')
            for m in range(12):
                for s in range(5):
                    value = grid.iloc[m, s]
                    ax.text(s, m, '%.1f' % value, ha='center', va='center', fontsize=6.3,
                            color='white' if abs(value) >= 60 else '#273c45')
            chosen = cases[cases.class_id.eq(cls)].iloc[0]
            if ship == chosen.shiptype:
                ax.add_patch(Rectangle((chosen.stage - 1.5, chosen.month - 1.5), 1, 1,
                                       fill=False, edgecolor='#161c20', linewidth=2))
            ax.set_xticks(range(5), range(1, 6))
            ax.set_yticks(range(12), [calendar.month_abbr[m] for m in range(1, 13)])
            ax.tick_params(length=0, labelsize=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ri == 0:
                ax.set_title(maps.styling.NAMES[ship], fontsize=10.5, pad=8)
            if ci == 0:
                ax.set_ylabel(cls + '  ' + CLASS_NAMES[cls], fontsize=9, labelpad=10)
            if ri == 3:
                ax.set_xlabel('Scenario stage', fontsize=8)
    fig.subplots_adjust(left=.095, right=.98, bottom=.10, top=.965, wspace=.13, hspace=.16)
    cax = fig.add_axes([.30, .045, .5, .012])
    bar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-100, -75, -50, -25, 0, 25, 50, 75, 100])
    bar.set_label('Arctic B minus traditional A (percentage points); outline = main-figure extreme case', fontsize=8)
    bar.ax.tick_params(labelsize=7)
    for ext in ['png', 'pdf']:
        fig.savefig(FIG / ('SI_PartI_All960_DominanceCases.' + ext), dpi=220, facecolor='white')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    main4_paths = sorted(FIG.glob('MainFig4_ODDependent_CountryRank.*'))
    main4_hashes = {p.name: sha256(p) for p in main4_paths}
    assert len(main4_hashes) >= 2
    data = pd.read_csv(OUT / 'partI_monthly_mechanism_decomposition.csv')
    assert len(data) == 1152 and np.allclose(data.A_pct + data.B_pct + data.U_pct, 100)
    cases = choose_cases(data)
    config = json.loads((HET / 'scenarios.json').read_text())
    groups = {g['id']: g for g in config['classes']}
    world = maps.styling.cart.base_world()
    registry = pd.read_csv(ROOT / 'data/chokepoints24/registry.csv').set_index('key')
    full_base = pd.concat([pd.read_parquet(HET / 'results' / ('month_%02d' % m) / 'demand.parquet') for m in range(1, 13)])
    all_frames, anchor_info, country_frames, numeric, route_audits = {}, {}, [], [], []
    for i, case in cases.iterrows():
        # Only the interventions already active at the selected stage are mapped.
        group = dict(groups[case.class_id])
        group['steps'] = group['steps'][:int(case.stage)]
        xy, markers, labels = maps.anchor_points(world, group, full_base, registry)
        anchor_info[case.class_id] = (markers, labels)
        sums, diagnostic = country_distances(case)
        country_frames.append(sums)
        route_audits.append(diagnostic)
        cases.loc[i, 'matched_trade_pct'] = diagnostic['matched_trade_pct']
        cases.loc[i, 'active_intervention_locators'] = '|'.join(labels)
        fingerprint = hashlib.sha256(sums.to_csv(index=False).encode() + xy.tobytes()).hexdigest()[:12]
        for col in METRICS:
            prefix = 'extreme_%s_%s_m%02d_s%d_%s_%s' % (case.class_id, case.shiptype, case.month, case.stage, fingerprint, col)
            frame, anchors, audit = maps.deformation(world, sums, col, xy, prefix)
            expected = frame.iso3.map(sums.set_index('iso3')[col])
            assert np.allclose(frame.distance_km, expected, equal_nan=True)
            all_frames[(case.class_id, col)] = (frame, anchors)
            numeric.append(dict(class_id=case.class_id, shiptype=case.shiptype, month=int(case.month),
                                stage=int(case.stage), metric=col, cache_prefix=prefix,
                                **{k: v for k, v in audit.items() if k != 'anchor_xy'}))
        print('EXTREME MAPS READY', case.class_id, case.shiptype, int(case.month), int(case.stage),
              'A/B/U', case.A_pct, case.B_pct, case.U_pct, flush=True)
    cases.to_csv(OUT / 'partI_extreme_case_selection.csv', index=False)
    pd.concat(country_frames).to_csv(OUT / 'partI_extreme_country_distance_source.csv', index=False)
    (OUT / 'partI_extreme_cartogram_audit.json').write_text(json.dumps(numeric, indent=2))
    styles = maps.styling.build_colour_scales()
    bounds = maps.styling.cart.shared_bounds([value[0] for value in all_frames.values()])
    fig = plt.figure(figsize=(22, 12.6), facecolor='white')
    for ri, case in cases.iterrows():
        top = .948 - ri * .208
        label = '%s  %s  |  %s  |  %s, stage %d' % (case.class_id, CLASS_NAMES[case.class_id],
                 maps.styling.NAMES[case.shiptype], calendar.month_abbr[int(case.month)], int(case.stage))
        fig.text(.018, top, label, fontsize=10, weight='bold', color='#293f49')
        for ci, col in enumerate(METRICS):
            ax = fig.add_axes([.008 + .253 * ci, top - .199, .246, .183])
            frame, anchors = all_frames[(case.class_id, col)]
            cmap, norm = styles['original' if ci == 0 else 'original_relative']
            values = frame.distance_km.to_numpy() if ci == 0 else frame.change_pct.to_numpy()
            values = np.where(frame.has_data, values, np.nan)
            frame.plot(ax=ax, color=cmap(norm(np.ma.masked_invalid(values))),
                       edgecolor=to_rgba(maps.styling.EDGE, .2), linewidth=.3)
            if ci:
                world.boundary.plot(ax=ax, color='#aab4b9', lw=.25, ls=':', alpha=.2)
                for point, marker in zip(anchors, anchor_info[case.class_id][0]):
                    ax.scatter(*point, marker=marker, s=18, c='#c9553e', edgecolors='white', linewidths=.4, zorder=6)
            ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect='equal')
            ax.set_axis_off()
            ax.text(.005, 1.015, chr(97 + ri * 4 + ci), transform=ax.transAxes,
                    va='top', fontsize=11, weight='bold')
        ax = fig.add_axes([.805, top - .174, .18, .153])
        draw_share_panel(ax, data, case, chr(100 + ri * 4), ri == 3)
    for ci, label in enumerate(['Open reference', 'Disturbed traditional detour', 'Disturbed Arctic option', 'Five-stage response']):
        fig.text([.018, .271, .524, .805][ci], .983, label, fontsize=10.5, weight='bold', color='#293f49')
    for key, rect in [('original', [.055, .081, .20, .012]), ('original_relative', [.32, .081, .43, .012])]:
        cmap, norm = styles[key]
        cax = fig.add_axes(rect)
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both')
        spec = maps.styling.colour_scale_definitions()[key]
        bar.set_ticks(spec['ticks'], labels=spec['ticklabels'])
        bar.set_label(spec['label'], fontsize=8)
        bar.ax.tick_params(labelsize=7.5)
        bar.minorticks_off()
    handles = [Patch(color=maps.COLORS[k], label=v) for k, v in [
        ('A', 'Traditional A'), ('B', 'Arctic B'), ('U', 'Unserved U')]] + [
        Patch(color=maps.styling.NO_DATA, label='No comparable mapped OD'),
        Line2D([], [], marker='o', color='#c9553e', lw=0, label='Restricted corridor'),
        Line2D([], [], marker='^', color='#c9553e', lw=0, label='Excluded coast: country locator'),
        Line2D([], [], marker='s', color='#c9553e', lw=0, label='Closed port service'),
        Line2D([], [], marker='*', color=maps.COLORS['B'], lw=0, label='B > A'),
        Patch(facecolor='none', edgecolor='#303f48', label='Mapped stage')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .009),
               ncol=5, frameon=False, fontsize=8, columnspacing=2.2, handlelength=1.7)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(FIG / ('MainFig3_FourDisturbance_ExtremeCases.' + ext), dpi=220, facecolor='white')
    plt.close(fig)
    plot_selection_landscape(data, cases)
    assert all(sha256(FIG / name) == value for name, value in main4_hashes.items())
    audit = dict(
        status='PASS', material_passport=dict(material_type='existing computed scenarios and route arrays',
        raw_inputs_changed=False, data_access_level='derived plus local route states',
        verification_status='source shares recomputed; geometry audited; descriptive selection'),
        selection_rule=RULE, selected_classes=cases.class_id.tolist(), selected_rows=4,
        candidate_count=int(cases.eligible_cases_in_class.sum()),
        selected_vessel_types=sorted(cases.shiptype.unique()),
        selection_is_post_hoc_extreme_not_representative=True,
        confidence_intervals_are_pointwise_not_selection_adjusted=True,
        map_metric_reference='Both disturbed maps are relative to the same open conventional reference',
        map_finite_cohort_excludes_unserved_but_shares_keep_original_denominator=True,
        map_count=len(numeric), map_audits='partI_extreme_cartogram_audit.json',
        source_csv_sha256=sha256(OUT / 'partI_monthly_mechanism_decomposition.csv'),
        protected_mainfig4_sha256=main4_hashes, mainfig4_unchanged=True,
        selected_route_audits=route_audits,
        inference='Conditional represented monthly trade dominance; not global annual or statistically robust phase transition')
    (OUT / 'partI_extreme_selection_audit.json').write_text(json.dumps(audit, indent=2))
    print(cases[['class_id', 'shiptype', 'month', 'stage', 'A_pct', 'B_pct', 'U_pct', 'gap_pp',
                 'gap_lo_pp', 'gap_hi_pp', 'new_B_pct', 'matched_trade_pct']].to_string(index=False), flush=True)
    print('NEW FOUR-ROW MAIN FIGURE AND ALL-CASE SI READY', flush=True)


if __name__ == '__main__':
    main()
