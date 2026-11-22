"""Add a common-scale Conventional-minus-Arctic axis to Fig. 3 d/h/l/p.

The source PDF is clipped outside the four response-panel rectangles, preserving
every other vector element. The replacement panels use unchanged source shares.
"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageChops, ImageDraw
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '64_NetworkTerminology_20261101'
FIG = HERE / 'figures'
QA = HERE / 'qa'
DATA = HERE / 'source_data'
COLORS = {'A': '#0077b6', 'B': '#e87519', 'U': '#a5abb0'}
GAP_COLOR = '#244e49'
LETTERS = ['d', 'h', 'l', 'p']
RIGHT_LIMITS = (-25, 25)
REPLACEMENT_REGIONS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readrows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def source_series():
    rows = readrows(SOURCE / 'source_data/partI_monthly_mechanism_decomposition.csv')
    cases = readrows(SOURCE / 'source_data/partI_extreme_case_selection.csv')
    assert len(rows) == 1152 and len(cases) == 4
    all_values = []
    for case, letter in zip(cases, LETTERS):
        selected = sorted([r for r in rows if all(r[k] == case[k] for k in
                          ['class_id', 'shiptype', 'month'])], key=lambda r: int(r['stage']))
        assert [int(r['stage']) for r in selected] == list(range(6))
        total = float(case['original_weight_usd'])
        for r in selected:
            a, b, u = [float(r[k + '_pct']) for k in 'ABU']
            assert math.isclose(a + b + u, 100, abs_tol=1e-8)
            assert math.isclose(float(r['original_weight_usd']), total, rel_tol=1e-12)
            difference = a - b
            assert RIGHT_LIMITS[0] < difference < RIGHT_LIMITS[1]
            all_values.append(dict(panel=letter, class_id=r['class_id'],
                shiptype=r['shiptype'], month=int(r['month']), stage=int(r['stage']),
                conventional_share_pct=a, arctic_share_pct=b, unserved_share_pct=u,
                conventional_minus_arctic_pp=difference,
                original_covered_trade_usd=total,
                conventional_trade_usd=total * a / 100,
                arctic_trade_usd=total * b / 100,
                conventional_minus_arctic_usd=total * difference / 100,
                selected_map_stage=int(case['stage']),
                arctic_dominant=b > a))
        check = next(r for r in all_values if r['panel'] == letter and
                     r['stage'] == int(case['stage']))
        assert math.isclose(check['conventional_minus_arctic_pp'],
                            float(case['A_pct']) - float(case['B_pct']), abs_tol=1e-12)
    return cases, all_values


def draw_panel(fig, case, values, ri):
    top = .948 - ri * .208
    # Reserve the original right-hand margin for the added axis labels.
    ax = fig.add_axes([.805, top - .174, .142, .153])
    x = np.arange(6)
    bottom = np.zeros(6)
    for key, field in [('A', 'conventional_share_pct'), ('B', 'arctic_share_pct'),
                       ('U', 'unserved_share_pct')]:
        ys = np.array([r[field] for r in values])
        ax.bar(x, ys, bottom=bottom, width=.64, color=COLORS[key],
               edgecolor='white', linewidth=.45, zorder=1)
        bottom += ys
    stage = int(case['stage'])
    ax.add_patch(Rectangle((stage - .36, -.4), .72, 100.8, fill=False,
                           edgecolor='#303f48', linewidth=1.35, clip_on=False))
    dominant = [r['stage'] for r in values if r['arctic_dominant']]
    if dominant:
        ax.scatter(dominant, [104.] * len(dominant), marker='*', s=35,
                   c=COLORS['B'], edgecolors='white', linewidth=.3, clip_on=False)
    ax.set(xlim=(-.6, 5.6), ylim=(0, 110), xticks=x, yticks=[0, 50, 100],
           ylabel='Covered trade (%)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#7d8b90')
    ax.tick_params(labelsize=8, width=.5, length=3)
    ax.yaxis.label.set_fontsize(8.5)
    for tick, s in zip(ax.get_xticklabels(), x):
        if s == stage:
            tick.set_weight('bold')
    if ri == 3:
        ax.set_xlabel('Cumulative scenario stage', fontsize=8.5)
    ax.text(-.16 / (.142 / .18), 1.04, LETTERS[ri], transform=ax.transAxes,
            fontsize=11, weight='bold')
    right = ax.twinx()
    right.patch.set_visible(False)
    right.set_ylim(*RIGHT_LIMITS)
    right.set_yticks([-20, -10, 0, 10, 20])
    right.set_ylabel('Conventional - Arctic (pp)', fontsize=8, color=GAP_COLOR,
                     labelpad=5)
    right.tick_params(axis='y', colors=GAP_COLOR, labelsize=7.5, width=.55, length=3)
    right.spines[['top', 'left', 'bottom']].set_visible(False)
    right.spines['right'].set_color(GAP_COLOR)
    right.spines['right'].set_linewidth(.65)
    right.axhline(0, color=GAP_COLOR, linestyle=(0, (2.5, 2.5)), linewidth=.8,
                  alpha=.75, zorder=4)
    gap = np.array([r['conventional_minus_arctic_pp'] for r in values])
    line, = right.plot(x, gap, color=GAP_COLOR, linewidth=1.6, marker='o',
                       markersize=3.1, markeredgecolor='white', markeredgewidth=.55,
                       zorder=5)
    line.set_path_effects([pe.Stroke(linewidth=2.8, foreground='white', alpha=.9),
                           pe.Normal()])
    assert np.array_equal(line.get_ydata(), gap)
    return ax, right


def clip_original(page, writer, regions):
    stream = ContentStream(page.get_contents(), writer)
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    ops = [([], b'q'), ([FloatObject(v) for v in (0, 0, w, h)], b're')]
    for x0, y0, x1, y1 in regions:
        ops.append(([FloatObject(v) for v in (x0, y0, x1-x0, y1-y0)], b're'))
    # Even-odd clipping leaves four holes in the otherwise untouched source page.
    ops.extend([([], b'W*'), ([], b'n')])
    stream.operations = ops + stream.operations + [([], b'Q')]
    page.replace_contents(stream)


def render(pdf, target, scale):
    doc = pdfium.PdfDocument(str(pdf))
    bitmap = doc[0].render(scale=scale)
    image = bitmap.to_pil().convert('RGB')
    image.save(target)
    bitmap.close()
    doc.close()
    return image


def verify_pixels(source, dest, regions):
    scale = 2
    before = render(source, QA / 'before.png', scale)
    after = render(dest, QA / 'after.png', scale)
    delta = ImageChops.difference(before, after)
    draw = ImageDraw.Draw(delta)
    for x0, y0, x1, y1 in regions:
        box = (math.floor(x0*scale)-3, math.floor(before.height-y1*scale)-3,
               math.ceil(x1*scale)+3, math.ceil(before.height-y0*scale)+3)
        draw.rectangle(box, fill=(0, 0, 0))
    delta.save(QA / 'outside_edit_regions_difference.png')
    maximum = max(max(pair) for pair in delta.getextrema())
    assert maximum <= 1, ('Pixels outside the four panel regions changed', maximum)
    return {'outside_edit_regions_max_pixel_difference': maximum,
            'all_other_maps_labels_and_legends_unchanged': True}




def main():
    for folder in [FIG, QA, DATA]:
        folder.mkdir(parents=True, exist_ok=True)
    inputs = [SOURCE / 'figures/MainFig3.pdf',
              SOURCE / 'source_data/partI_monthly_mechanism_decomposition.csv',
              SOURCE / 'source_data/partI_extreme_case_selection.csv',
]
    source_hashes = {str(p): sha(p) for p in inputs}
    cases, values = source_series()
    with (DATA / 'MainFig3_dual_axis_values.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(22, 12.6), facecolor='none')
    pairs = []
    for ri, case in enumerate(cases):
        panel_values = [r for r in values if r['panel'] == LETTERS[ri]]
        pairs.append(draw_panel(fig, case, panel_values, ri))
        top = .948 - ri * .208
        # Includes old ticks/panel letter, and all new right-axis text.
        REPLACEMENT_REGIONS.append((.772 * 1584, (top-.211)*907.2,
                                    1584, (top-.001)*907.2))
    fig.legend(handles=[Line2D([], [], color=GAP_COLOR, lw=1.6, marker='o',
                               markersize=3, label='Conventional - Arctic (right axis)'),
                        Line2D([], [], color=GAP_COLOR, lw=.8, ls=(0, (2.5, 2.5)),
                               label='Equal trade shares')],
               loc='lower left', bbox_to_anchor=(.788, .075), frameon=False,
               fontsize=8, handlelength=2, borderaxespad=0, labelspacing=.5)
    # The legend occupies formerly empty space; keep it in the audited edit mask.
    legend_region = (.772*1584, .070*907.2, 1584, .107*907.2)
    REPLACEMENT_REGIONS.append(legend_region)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = []
    for left, right in pairs:
        bounds = right.yaxis.label.get_window_extent(renderer)
        assert bounds.x1 <= fig.bbox.x1, 'Right-axis label falls outside the page'
        text_boxes.append(list(bounds.bounds))
    buf = BytesIO()
    fig.savefig(buf, format='pdf', transparent=True)
    fig.savefig(QA / 'dual_axis_overlay.svg', transparent=True)
    plt.close(fig)
    buf.seek(0)
    writer = PdfWriter()
    source = SOURCE / 'figures/MainFig3.pdf'
    writer.add_page(PdfReader(source).pages[0])
    page = writer.pages[0]
    clip_original(page, writer, REPLACEMENT_REGIONS)
    page.merge_page(PdfReader(buf).pages[0])
    writer.add_metadata({'/Title': 'Figure 3: conditional route layer reversals',
                         '/Subject': 'Unchanged routing outcomes with Conventional minus Arctic share contrast'})
    target = FIG / 'MainFig3.pdf'
    with target.open('wb') as handle:
        writer.write(handle)
    image = render(target, FIG / 'MainFig3.png', 220/72)
    image.resize((2376, round(image.height*2376/image.width)), Image.Resampling.LANCZOS).save(
        QA / 'MainFig3_overview.png')
    after = render(target, QA / 'MainFig3_for_manuscript.png', 3.2)
    # One enlarged view of the four changed panels supports direct visual QA.
    after.crop((round(.772*after.width), round(.032*after.height), after.width,
                round(.875*after.height))).save(QA / 'four_dual_axis_panels.png')
    pixel_check = verify_pixels(source, target, REPLACEMENT_REGIONS)
    assert all(sha(p) == digest for p, digest in source_hashes.items())
    audit = dict(status='PASS', input_sha256=source_hashes,
                 input_files_unchanged=True, changed_panels=LETTERS,
                 source_row_count=1152, plotted_stage_rows=len(values),
                 raw_routing_recalculated=False, selection_rule_unchanged=True,
                 difference='Conventional share minus Arctic share, percentage points',
                 denominator='original covered trade value in the same vessel-month',
                 right_axis_limits=RIGHT_LIMITS, right_axis_label_boxes=text_boxes,
                 caveat='Conditional selected extremes; not proof of annual Arctic dominance',
                 edited_pdf_regions=REPLACEMENT_REGIONS, **pixel_check)
    (HERE / 'figure_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print('FIGURE_READY', target)


if __name__ == '__main__':
    main()
