"""Retain the original four-panel rank-change design, replacing its scenarios."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
OUT = TASK / 'results'
FIG = TASK / 'figures'
OLD = ROOT / '51_TradeWeighted_CountryBetweenness'
sys.path.insert(0, str(OLD))
import plot_vessel_top50_rank_population as original_plot

SHIPS = ['container', 'bulk', 'tanker', 'lng']
AXIS_MAXIMUM = 75
AXIS_TICKS = [1, 25, 50, 75]


def main(main_only=False):
    FIG.mkdir(exist_ok=True)
    protected = [OUT / 'country_betweenness_new_closures.csv', OUT / 'MainFig4_country_rank_source.csv',
                 OUT / 'vessel_top50_selection.csv'] + list(FIG.glob('SI_MainFig4*')) + list(FIG.glob('MainFig3*'))
    protected_hashes = {str(p.relative_to(TASK)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.exists()}
    previous_hashes = {}
    if main_only:
        backup = OUT / ('MainFig4_before_axis%d' % AXIS_MAXIMUM)
        backup.mkdir(exist_ok=True)
        for path in FIG.glob('MainFig4_ODDependent_CountryRank.*'):
            if not (backup / path.name).exists():
                shutil.copy2(path, backup / path.name)
            previous_hashes[path.name] = hashlib.sha256((backup / path.name).read_bytes()).hexdigest()
        prior_audit = OUT / 'MainFig4_axis_update_audit.json'
        if prior_audit.exists() and not (backup / 'axis_update_audit.json').exists():
            shutil.copy2(prior_audit, backup / 'axis_update_audit.json')
    all_data = pd.read_csv(OUT / 'country_betweenness_new_closures.csv')
    annual = all_data[all_data.month == 0].copy()
    fixed = pd.read_csv(OUT / 'partII_mainfig4_fixed_month_optimal_closures.csv')
    month_by_ship = fixed.groupby('shiptype').month.first().astype(int).to_dict()
    main_data = pd.concat([all_data[(all_data.shiptype == ship) & (all_data.month == month_by_ship[ship])] for ship in SHIPS])
    metadata = pd.read_csv(OLD / 'results/all_country_rank_display_metadata.csv').set_index('iso3')
    population_raw = json.loads((OLD / 'data/population_2024/worldbank_SP.POP.TOTL_2024.json').read_text())
    population = {r['countryiso3code']: r for r in population_raw[1] if r.get('value') is not None}
    extras = pd.read_csv(OLD / 'results/vessel_top50_population_2024.csv').set_index('iso3')
    rows = []
    for ship in SHIPS:
        top = main_data[(main_data.shiptype == ship) & (main_data.step == 5) & (main_data.radius_km == 100) &
                     (main_data.network == 'embedded_premium05') & main_data.iso3.ne('TWN')].sort_values(['rank', 'name']).head(50)
        assert len(top) == 50 and top.betweenness_pct.round(10).gt(0).all()
        for position, r in enumerate(top.itertuples(), 1):
            if r.iso3 in population:
                raw = population[r.iso3]
                assert raw['date'] == '2024'
                pop = float(raw['value'])
                source = 'World Bank WDI SP.POP.TOTL'
                url = 'https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?date=2024&format=json&per_page=400'
                status = 'Cached published value for 2024'
            else:
                extra = extras.loc[r.iso3]
                pop, source, url, status = extra.population, extra.population_source, extra.population_source_url, extra.population_status
            assert pop > 0
            rows.append(dict(shiptype=ship, iso3=r.iso3, selection_position=position, selection_arctic_rank=r.rank,
                display_name=metadata.loc[r.iso3, 'display_name'], region=metadata.loc[r.iso3, 'region'], population=pop,
                population_year=2024, population_source=source, population_source_url=url, population_status=status))
    pd.DataFrame(rows).to_csv(OUT / 'vessel_top50_selection.csv', index=False)
    # This alias is confined to the legacy renderer; stored analytical results
    # explicitly retain the true 5%-premium route-selection label.
    plotting_view = main_data.replace({'network': {'embedded_premium05': 'embedded_distance'}})
    annual_view = annual.replace({'network': {'embedded_premium05': 'embedded_distance'}})
    original_plot.OUT = OUT
    original_plot.FIG = FIG
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'text.color': '#31444f', 'xtick.color': '#637782', 'axes.labelcolor': '#516873'})
    meta, layout, height = original_plot.layouts()

    def draw(radius=100, reference='same_closure', view=None):
        fig, source = original_plot.figure(plotting_view if view is None else view, meta, layout, height, radius=radius, reference=reference)
        for text in fig.findobj(match=matplotlib.text.Text):
            if text.get_text().endswith(' | '):
                text.set_text(text.get_text() + 'annual')
        return fig, source

    fig, source = draw()
    assert len(fig.axes) == 4
    assert source[['conventional_rank', 'arctic_rank']].max().max() <= AXIS_MAXIMUM
    for ax in fig.axes:
        ax.set_xlim(ax.get_xlim()[0], AXIS_MAXIMUM)
        ax.set_xticks(AXIS_TICKS, labels=[str(value) for value in AXIS_TICKS])
    axis_limits = [list(ax.get_xlim()) for ax in fig.axes]
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(FIG / ('MainFig4_ODDependent_CountryRank.' + ext), dpi=230, facecolor='white')
    plt.close(fig)
    source.to_csv(OUT / 'MainFig4_country_rank_source.csv', index=False)
    if main_only:
        assert all(hashlib.sha256((TASK / name).read_bytes()).hexdigest() == value for name, value in protected_hashes.items())
        design_path = OUT / 'MainFig4_design_audit.json'
        design = json.loads(design_path.read_text())
        design.update(x_axis_limits=axis_limits, x_axis_ticks=AXIS_TICKS, x_axis_maximum=AXIS_MAXIMUM)
        design_path.write_text(json.dumps(design, indent=2))
        audit = dict(status='PASS', requested_change='Main figure four x-axis maximum %d' % AXIS_MAXIMUM,
            x_axis_limits=axis_limits, x_axis_ticks=AXIS_TICKS, panels=4,
            numerical_results_unchanged=True, supplementary_figures_unchanged=True,
            previous_figure_sha256=previous_hashes, protected_sha256=protected_hashes,
            updated_figure_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in FIG.glob('MainFig4_ODDependent_CountryRank.*')})
        if (backup / 'axis_update_audit.json').exists():
            audit['previous_axis_audit'] = str((backup / 'axis_update_audit.json').relative_to(TASK))
        (OUT / 'MainFig4_axis_update_audit.json').write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit, indent=2), flush=True)
        return
    fig, annual_source = draw(view=annual_view)
    for ext in ['png', 'pdf']:
        fig.savefig(FIG / ('SI_MainFig4_AnnualSameClosures.' + ext), dpi=200, facecolor='white')
    plt.close(fig)
    annual_source.to_csv(OUT / 'SI_MainFig4_annual_source.csv', index=False)
    fig, reference = draw(reference='original_open')
    for ext in ['png', 'pdf']:
        fig.savefig(FIG / ('SI_MainFig4_OriginalOpenReference.' + ext), dpi=200, facecolor='white')
    plt.close(fig)
    reference.to_csv(OUT / 'SI_MainFig4_original_reference.csv', index=False)
    sources = []
    with PdfPages(FIG / 'SI_MainFig4_CoastalRadius_Sensitivity.pdf') as pdf:
        for radius in [25, 50, 100, 200]:
            fig, data = draw(radius=radius)
            pdf.savefig(fig, facecolor='white')
            plt.close(fig)
            sources.append(data)
    pd.concat(sources).to_csv(OUT / 'SI_MainFig4_radius_sensitivity.csv', index=False)
    sensitivities = []
    for ship in SHIPS:
        a = main_data[(main_data.shiptype == ship) & (main_data.step == 5) & (main_data.network == 'embedded_premium05')]
        ref = a[a.radius_km == 100].set_index('iso3')
        for radius in [25, 50, 98, 102, 200]:
            g = a[a.radius_km == radius].set_index('iso3')
            sensitivities.append(dict(shiptype=ship, radius_km=radius,
                spearman_vs_100km=float(g['rank'].corr(ref['rank'], method='spearman')),
                maximum_rank_difference=float(np.max(np.abs(g['rank'] - ref['rank'])))))
    pd.DataFrame(sensitivities).to_csv(OUT / 'SI_rank_radius_stability.csv', index=False)
    (OUT / 'MainFig4_design_audit.json').write_text(json.dumps(dict(status='PASS', panels=4,
        original_visual_form_retained=True, original_mainfig4_not_overwritten=True,
        countries_per_ship=50, displayed_TWN=False, independent_lists_per_ship=True,
        region_grouped_and_sorted_ascending_arctic_rank=True, country_universe=int(annual.iso3.nunique()),
        x_axis_increasing=True, x_axis_limits=axis_limits, x_axis_ticks=AXIS_TICKS, x_axis_maximum=AXIS_MAXIMUM,
        start_circle_area='60 * population / 1e9', end_diamond_area='1 * weighted betweenness pct',
        colours_unchanged=True, main_reference='Without Arctic under same optimized five closures',
        original_months_preserved=month_by_ship, annual_all_months_in_SI=True, renderer_only_legacy_label_alias=True, true_choice_premium_pct=5,
        uncertainty_note='Scenario-conditioned ranks, not causal forecasts; radius sensitivity supplied'), indent=2))
    print('MAINFIG4 COMPLETE', flush=True)
    print(source[source.iso3.isin(['USA', 'RUS', 'CHN', 'GBR', 'FRA', 'DEU', 'NLD'])][['shiptype', 'iso3', 'conventional_rank', 'arctic_rank', 'rank_gain']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-only', action='store_true', help='Update only the main figure; leave SI and analytical tables unchanged.')
    main(main_only=parser.parse_args().main_only)
