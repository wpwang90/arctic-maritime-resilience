"""Monthly coverage and dependency diagnostics accompanying the two parts."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, LinearSegmentedColormap
import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
OUT, FIG = TASK / 'results', TASK / 'figures'


def main():
    source = ROOT / '58_FourClass_Threshold_Analysis_20261101'
    spec = importlib.util.spec_from_file_location('existing_disturbance_plot', source / 'summarize_and_plot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FIG, module.OUT = FIG, OUT
    data = pd.read_csv(source / 'results/four_class_monthly_shares.csv')
    summary = pd.read_csv(source / 'results/threshold_summary_by_shiptype.csv')
    module.plot(data, summary)
    for stem in ['FourClass_Shiptype_ShareCurves', 'FourClass_Crossover_Calendar']:
        for ext in ['png', 'pdf']:
            (FIG / (stem + '.' + ext)).replace(FIG / ('SI_PartI_' + stem + '.' + ext))
    dependency = pd.read_csv(OUT / 'partII_gate_candidates_deduplicated_OD_dependencies.csv')
    shiptypes = ['container', 'bulk', 'tanker', 'lng']
    order = dependency.groupby(['gate', 'gate_name']).allocated_unique_new_OD_weight_usd.sum().sort_values(ascending=False)
    labels = [x[1] for x in order.index]
    values = dependency.pivot(index='gate', columns='shiptype', values='share_unique_new_OD_weight_pct').reindex(
        index=[x[0] for x in order.index], columns=shiptypes).to_numpy()
    cmap = LinearSegmentedColormap.from_list('dep', ['#eff4ea', '#b2d3c5', '#62a6b1', '#3e6084'], N=512)
    cmap.set_bad('#e1e3e4')
    norm = FuncNorm((lambda x: np.log10(1 + x), lambda x: 10 ** x - 1), vmin=0, vmax=60)
    fig, ax = plt.subplots(figsize=(7.6, 9))
    image = ax.imshow(np.ma.masked_where(values == 0, values), aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(4), ['Container', 'Dry bulk', 'Tanker', 'LNG'])
    ax.set_yticks(range(25), labels)
    ax.tick_params(length=0, labelsize=9)
    for i in range(25):
        for j in range(4):
            ax.text(j, i, '0' if values[i, j] == 0 else '%.1f' % values[i, j], ha='center', va='center',
                fontsize=8, color='white' if values[i, j] > 20 else '#334752')
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=.39, right=.95, bottom=.12, top=.97)
    cax = fig.add_axes([.44, .045, .46, .017])
    bar = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[0, 1, 5, 10, 30, 60], extend='max')
    bar.ax.set_xticklabels(['0', '1', '5', '10', '30', '60'])
    bar.minorticks_off()
    bar.set_label('Split share of unique newly Arctic-routed OD trade (%)', fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(FIG / ('SI_PartII_OD_to_Chokepoint_Dependencies.' + ext), dpi=230, facecolor='white')
    plt.close(fig)
    print('AUXILIARY FIGURES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
