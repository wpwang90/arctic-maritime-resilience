"""Four disturbance figures in the original three-maps-plus-shares composition."""
import argparse
import calendar
import json
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import shapely

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / '49_Transparent_StageJump_Maps'))
import plot_transparent as styling

OUT = TASK / 'results'
FIG = TASK / 'figures'
HET = ROOT / '58_FourClass_Threshold_Analysis_20261101'
SHIPS = ['container', 'bulk', 'tanker', 'lng']
COLORS = {'A': '#0077b6', 'B': '#e87519', 'U': '#a5abb0'}


class ConservativeAnchor(styling.AnchorCartogram):
    @property
    def _reduction_factor(self):
        return .25 / (self.average_error + 1)


def deformation(world, records, column, anchors, prefix):
    cache = OUT / 'partI_cartograms' / (prefix + '.parquet')
    if cache.exists():
        audit = json.loads(cache.with_suffix('.json').read_text())
        return gpd.read_parquet(cache), np.asarray(audit['anchor_xy']), audit
    cache.parent.mkdir(exist_ok=True)
    frame = world.copy()
    reference = records.set_index('iso3')
    frame['distance_km'] = frame.iso3.map(reference[column])
    frame['reference_km'] = frame.iso3.map(reference['baseline_km'])
    frame['distance_ratio'] = frame.distance_km / frame.reference_km
    frame['change_pct'] = 100 * (frame.distance_ratio - 1)
    frame['has_data'] = frame.distance_ratio.notna()
    frame['matched_trade_usd'] = frame.iso3.map(reference['matched_trade_usd'])
    frame['target_area'] = frame.reference_area * frame.distance_ratio.fillna(1)
    result = frame.copy()
    xy = anchors.copy()
    iterations = 0
    transformed_nontrivially = not np.allclose(frame.distance_ratio.fillna(1), 1)
    if transformed_nontrivially:
        parts = styling.cart.component_targets(frame)
        transform = styling.AnchorCartogram(parts, 'target_area', max_iterations=150, max_average_error=.0005, anchor_xy=xy)
        transformed = gpd.GeoDataFrame(transform, geometry='geometry', crs=None)
        for index, group in transformed.groupby('parent_index'):
            result.loc[index, 'geometry'] = shapely.union_all(group.geometry.array)
        scale = np.sqrt(frame.target_area.sum() / shapely.area(result.geometry.array).sum())
        result.geometry = shapely.transform(result.geometry.array, lambda p: p * scale)
        xy = transform.__dict__['anchor_xy'] * scale
        iterations = int(transform.iteration)
    area_error = np.abs(shapely.area(result.geometry.array) / result.target_area - 1)
    observed = result.has_data & result.matched_trade_usd.gt(0)
    error = float(100 * np.average(area_error[observed], weights=result.matched_trade_usd[observed]))
    total = float(shapely.area(result.geometry.array).sum())
    union = float(shapely.area(shapely.union_all(result.geometry.array)))
    overlap = 100 * max(0, total - union) / union
    conservative_retry = False
    if transformed_nontrivially and (overlap >= .05 or error >= .5):
        print('CONSERVATIVE MAP STEP', prefix, 'overlap', overlap, 'area error', error, flush=True)
        parts = styling.cart.component_targets(frame)
        transform = ConservativeAnchor(parts, 'target_area', max_iterations=600, max_average_error=.0005, anchor_xy=anchors)
        transformed = gpd.GeoDataFrame(transform, geometry='geometry', crs=None)
        result = frame.copy()
        for index, group in transformed.groupby('parent_index'):
            result.loc[index, 'geometry'] = shapely.union_all(group.geometry.array)
        scale = np.sqrt(frame.target_area.sum() / shapely.area(result.geometry.array).sum())
        result.geometry = shapely.transform(result.geometry.array, lambda p: p * scale)
        xy = transform.__dict__['anchor_xy'] * scale
        iterations = int(transform.iteration)
        area_error = np.abs(shapely.area(result.geometry.array) / result.target_area - 1)
        error = float(100 * np.average(area_error[observed], weights=result.matched_trade_usd[observed]))
        total = float(shapely.area(result.geometry.array).sum())
        union = float(shapely.area(shapely.union_all(result.geometry.array)))
        overlap = 100 * max(0, total - union) / union
        conservative_retry = True
    assert shapely.is_valid(result.geometry.array).all()
    assert overlap < .05 and error < .5, (prefix, overlap, error)
    audit = dict(status='PASS', weighted_area_error_pct=error, overlap_pct=overlap, iterations=iterations,
        conservative_quarter_step=conservative_retry,
        anchor_xy=xy.tolist(), area_encoding='country area times ratio of mean physical route distance to open conventional distance',
        numerical_only_not_geographic_displacement=True)
    result.to_parquet(cache, index=False)
    cache.with_suffix('.json').write_text(json.dumps(audit, indent=2))
    return result, xy, audit


def anchor_points(world, group, base, registry):
    xy, markers, labels = [], [], []
    windex = world.set_index('iso3')
    for step in group['steps']:
        for gate in step.get('gates', []):
            r = registry.loc[gate]
            p = gpd.GeoSeries([shapely.Point(float(r.lon), float(r.lat))], crs=4326).to_crs(8857).iloc[0]
            xy.append([p.x / 1e6, p.y / 1e6]); markers.append('o'); labels.append(gate)
        for country in step.get('coasts', []):
            p = windex.loc[country].geometry.representative_point()
            xy.append([p.x, p.y]); markers.append('^'); labels.append(country)
        for port in step.get('ports', []):
            possible = base[base.origin_port_id.eq(port)]
            if len(possible):
                lon, lat = possible.iloc[0][['origin_lon', 'origin_lat']]
            else:
                r = base[base.dest_port_id.eq(port)].iloc[0]
                lon, lat = r.dest_lon, r.dest_lat
            p = gpd.GeoSeries([shapely.Point(float(lon), float(lat))], crs=4326).to_crs(8857).iloc[0]
            xy.append([p.x / 1e6, p.y / 1e6]); markers.append('s'); labels.append(port)
    return np.asarray(xy), markers, labels


def main(class_ids=None):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    styles = styling.build_colour_scales()
    world = styling.cart.base_world()
    data = pd.read_csv(OUT / 'partI_monthly_mechanism_decomposition.csv')
    config = json.loads((HET / 'scenarios.json').read_text())
    if class_ids:
        config['classes'] = [g for g in config['classes'] if g['id'] in class_ids]
    registry = pd.read_csv(ROOT / 'data/chokepoints24/registry.csv').set_index('key')
    full_base = pd.concat([pd.read_parquet(HET / 'results' / ('month_%02d' % m) / 'demand.parquet') for m in range(1, 13)])
    selected_rows, country_frames, numerical = [], [], []
    all_frames = {}
    anchor_info = {}
    for group in config['classes']:
        cls = group['id']
        xy, markers, labels = anchor_points(world, group, full_base, registry)
        anchor_info[cls] = (markers, labels)
        for ship in SHIPS:
            r = data[(data.class_id == cls) & (data.shiptype == ship) & (data.stage == 5)].sort_values(
                ['new_B_pct', 'U_pct', 'month'], ascending=[False, True, True]).iloc[0]
            month = int(r.month)
            folder = HET / 'results' / ('month_%02d' % month)
            base = pd.read_parquet(folder / 'demand.parquet')
            arrays = np.load(folder / 'od_states.npz')
            si = int(r.state_row)
            d0 = arrays['without_km'][0]
            da = arrays['without_km'][si]
            db = np.where(arrays['B'][si], arrays['arctic_km'][si], da)
            matched = base.shiptype.eq(ship).to_numpy() & arrays['service_allowed'][si] & np.isfinite(d0) & np.isfinite(da) & np.isfinite(db)
            frame = base.loc[matched, ['origin_iso3', 'trade_weight_usd']].copy()
            for key, value in [('baseline_km', d0), ('traditional_km', da), ('embedded_km', db)]:
                frame[key] = value[matched] * frame.trade_weight_usd
            sums = frame.groupby('origin_iso3').sum(numeric_only=True)
            totals = sums.trade_weight_usd.copy()
            for key in ['baseline_km', 'traditional_km', 'embedded_km']:
                sums[key] /= totals
            sums['matched_trade_usd'] = totals
            sums['matched_od_count'] = frame.groupby('origin_iso3').size()
            sums = sums.reset_index().rename(columns={'origin_iso3': 'iso3'})
            sums['class_id'], sums['shiptype'], sums['month'] = cls, ship, month
            country_frames.append(sums)
            selected_rows.append(dict(class_id=cls, shiptype=ship, month=month, stage=5,
                rule='Stage five held fixed; choose month with largest gross new Arctic trade, then least U, then month',
                matched_trade_pct=float(100 * base.trade_weight_usd[matched].sum() / r.original_weight_usd),
                full_B_pct=r.B_pct, full_A_pct=r.A_pct, full_U_pct=r.U_pct))
            for col in ['baseline_km', 'traditional_km', 'embedded_km']:
                current, anchors, audit = deformation(world, sums, col, xy, cls + '_' + ship + '_' + col)
                all_frames[(cls, ship, col)] = (current, anchors)
                numerical.append(dict(class_id=cls, shiptype=ship, metric=col, **{k: v for k, v in audit.items() if k != 'anchor_xy'}))
            print('PART I MAPS', cls, ship, month, flush=True)
    suffix = '_' + '-'.join(class_ids) if class_ids else ''
    pd.DataFrame(selected_rows).to_csv(OUT / ('partI_map_case_selection' + suffix + '.csv'), index=False)
    pd.concat(country_frames).to_csv(OUT / ('partI_map_country_distance_source' + suffix + '.csv'), index=False)
    pd.DataFrame(numerical).to_json(OUT / ('partI_cartogram_audit' + suffix + '.json'), orient='records', indent=2)
    for group in config['classes']:
        cls = group['id']
        fig = plt.figure(figsize=(22, 13.2), facecolor='white')
        frames = [all_frames[(cls, ship, col)][0] for ship in SHIPS for col in ['baseline_km', 'traditional_km', 'embedded_km']]
        bounds = styling.cart.shared_bounds(frames)
        for ri, ship in enumerate(SHIPS):
            case = next(z for z in selected_rows if z['class_id'] == cls and z['shiptype'] == ship)
            top = .952 - ri * .211
            fig.text(.018, top, styling.NAMES[ship] + ' | ' + calendar.month_abbr[case['month']], fontsize=10, weight='bold')
            for ci, col in enumerate(['baseline_km', 'traditional_km', 'embedded_km']):
                ax = fig.add_axes([.008 + .253 * ci, top - .201, .246, .185])
                current, anchors = all_frames[(cls, ship, col)]
                cmap, norm = styles['original' if ci == 0 else 'original_relative']
                values = current.distance_km.to_numpy() if ci == 0 else current.change_pct.to_numpy()
                values = np.where(current.has_data, values, np.nan)
                current.plot(ax=ax, color=cmap(norm(np.ma.masked_invalid(values))), edgecolor=to_rgba(styling.EDGE, .2), linewidth=.3)
                if ci:
                    world.boundary.plot(ax=ax, color='#aab4b9', lw=.25, ls=':', alpha=.2)
                    for j, (point, marker) in enumerate(zip(anchors, anchor_info[cls][0]), 1):
                        ax.scatter(*point, marker=marker, s=18, c='#c9553e', edgecolors='white', linewidths=.4, zorder=6)
                ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect='equal')
                ax.set_axis_off()
                ax.text(.005, 1.015, chr(97 + ri * 4 + ci), transform=ax.transAxes, va='top', fontsize=11, weight='bold')
            ax = fig.add_axes([.805, top - .174, .18, .153])
            values = data[(data.class_id == cls) & (data.shiptype == ship) & (data.month == case['month'])].sort_values('stage')
            bottom = np.zeros(6)
            for key in ['A', 'B', 'U']:
                ax.bar(range(6), values[key + '_pct'], bottom=bottom, width=.65, color=COLORS[key], edgecolor='white', lw=.4)
                bottom += values[key + '_pct'].to_numpy()
            ax.set(xlim=(-.6, 5.6), ylim=(0, 106), xticks=range(6), yticks=[0, 50, 100], ylabel='Covered trade (%)')
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.text(-.16, 1.06, chr(100 + ri * 4), transform=ax.transAxes, fontsize=11, weight='bold')
            if ri == 3:
                ax.set_xlabel('Cumulative scenario stage')
        for ci, label in enumerate(['Open conventional reference', 'Disturbed conventional detour', 'Disturbed Arctic option', 'Five-step scenario response']):
            fig.text([.018, .271, .524, .805][ci], .983, label, fontsize=10.5, weight='bold')
        for key, rect in [('original', [.05, .078, .2, .012]), ('original_relative', [.32, .078, .43, .012])]:
            cmap, norm = styles[key]
            ax = fig.add_axes(rect)
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal', extend='both')
            spec = styling.colour_scale_definitions()[key]
            bar.set_ticks(spec['ticks'], labels=spec['ticklabels'])
            bar.set_label(spec['label'], fontsize=8)
            bar.ax.tick_params(labelsize=7.5)
            bar.minorticks_off()
        fig.legend(handles=[Patch(color=COLORS[k], label=v) for k, v in [('A', 'Traditional A'), ('B', 'Arctic B'), ('U', 'Unserved U')]] +
            [Patch(color=styling.NO_DATA, label='No comparable mapped OD'),
             Line2D([], [], marker='o', color='#c9553e', lw=0, label='Restricted corridor'),
             Line2D([], [], marker='^', color='#c9553e', lw=0, label='Excluded coast: country locator'),
             Line2D([], [], marker='s', color='#c9553e', lw=0, label='Closed port service')],
            loc='lower center', bbox_to_anchor=(.5, .018), ncol=7, frameon=False, fontsize=8)
        for ext in ['png', 'pdf']:
            fig.savefig(FIG / ('PartI_' + cls + '_Maps_and_TradeShares.' + ext), dpi=200, facecolor='white')
        plt.close(fig)
        print('PART I FIGURE SAVED', cls, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', choices=['T1', 'T2', 'T3', 'T4'])
    main(parser.parse_args().classes)
