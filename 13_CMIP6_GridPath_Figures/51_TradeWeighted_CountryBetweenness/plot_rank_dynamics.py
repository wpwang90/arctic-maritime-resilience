"""A standalone rank-only figure; no maps or absolute centrality panels."""
from pathlib import Path
import calendar
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_betweenness import table, save, SHIPS, NAMES, SHORT

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = HERE / "figures"
ACCENTS = {
    "RUS": "#ba6679", "CAN": "#3c9c92", "DNK": "#9475b3",
    "USA": "#487eae", "GBR": "#c59a50", "JPN": "#72a163",
}
BACKGROUND = "#b6bec4"
RANK_CMAP = LinearSegmentedColormap.from_list(
    "ordinal_rank", ["#f2dfb2", "#9cc8bb", "#739ebd", "#72618f"], N=256)
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "signed_rank_gain", ["#ac566c", "#e4b4a9", "#fafafa", "#a0ccc6", "#347b82"], N=1025)
LABEL_BOXES = []


def label(name):
    return SHORT.get(name, name)


def select_countries(data, ship):
    base = table(data, ship, "traditional", 0).drop(index="TWN", errors="ignore")
    traditional = table(data, ship, "traditional", 5).loc[base.index]
    embedded = table(data, ship, "embedded_distance", 5).loc[base.index]
    selected = list(dict.fromkeys(
        base.sort_values(["rank", "name"]).head(4).index.tolist()
        + embedded.sort_values(["rank", "name"]).head(4).index.tolist()))
    gain = traditional["rank"] - embedded["rank"]
    importance = pd.concat([base["rank"], traditional["rank"], embedded["rank"]], axis=1).min(axis=1)
    candidates = pd.DataFrame({"gain": gain, "best_rank": importance, "name": base.name})
    candidates = candidates[candidates.best_rank.le(30)]
    # Include both upward and downward changes, not only the largest gainers.
    for positive in [True, False]:
        pool = candidates[candidates.gain.gt(0) if positive else candidates.gain.lt(0)]
        pool = pool.drop(index=selected, errors="ignore").sort_values(
            ["gain", "best_rank", "name"], ascending=[not positive, True, True])
        selected.extend(pool.head(2).index.tolist())
    histories = [base["rank"]] + [table(data, ship, "embedded_distance", k).loc[base.index, "rank"] for k in range(6)]
    ranks = pd.concat(histories, axis=1)
    candidates["rank_range"] = ranks.max(axis=1) - ranks.min(axis=1)
    for iso in candidates.sort_values(["rank_range", "best_rank", "name"], ascending=[False, True, True]).index:
        if len(selected) >= 12:
            break
        if iso not in selected:
            selected.append(iso)
    assert len(selected) == 12
    return base.loc[selected].sort_values(["rank", "name"]).index.tolist()


def spread_labels(values, low, high, gap):
    """Move labels only; leader lines retain exact, unjittered rank endpoints."""
    order = np.argsort(values, kind="stable")
    desired = np.asarray(values, float)[order]
    placed = np.clip(desired, low, high)
    for j in range(1, len(placed)):
        placed[j] = max(placed[j], placed[j-1] + gap)
    if placed[-1] > high:
        placed[-1] = high
        for j in range(len(placed)-2, -1, -1):
            placed[j] = min(placed[j], placed[j+1] - gap)
    if placed[0] < low:
        placed += low - placed[0]
    result = np.empty_like(placed)
    result[order] = placed
    return result


def rank_trajectories(ax, data, ship, ids):
    base = table(data, ship, "traditional", 0).loc[ids]
    states = [base] + [table(data, ship, "embedded_distance", k).loc[ids] for k in range(6)]
    ranks = np.column_stack([state["rank"].to_numpy() for state in states])
    support = np.column_stack([state.betweenness_pct.round(10).gt(0).to_numpy() for state in states])
    high = max(40, int(np.ceil(ranks.max()/10)*10)) + 4
    x = np.arange(-1, 6)
    ax.axvspan(-1.12, -.5, color="#f4f5f6", zorder=0)
    ax.axvline(-.5, color="#d1d8dc", lw=.8, ls=(0, (3, 3)), zorder=1)
    ax.axhspan(.5, 10.5, color="#f6f8f8", zorder=0)
    for k in x:
        ax.axvline(k, color="#e9edef", lw=.55, zorder=0)
    left = spread_labels(ranks[:, 0], 2.5, high-2, (high-1)*.050)
    right = spread_labels(ranks[:, -1], 2.5, high-2, (high-1)*.050)
    for j, iso in enumerate(ids):
        colour = ACCENTS.get(iso, BACKGROUND)
        highlighted = iso in ACCENTS
        ax.plot(x, ranks[j], color=colour, lw=1.65 if highlighted else .95,
                alpha=.95 if highlighted else .72, zorder=3 if highlighted else 2)
        ax.scatter(x, ranks[j], s=17 if highlighted else 12,
                   facecolors=[colour if yes else "white" for yes in support[j]],
                   edgecolors=colour, lw=.9, zorder=5 if highlighted else 4)
        texts = []
        for endpoint, px, target, anchor, direction in [
            (0, -1.29, left[j], "right", -1),
            (-1, 5.29, right[j], "left", 1),
        ]:
            ax.plot([x[endpoint], px-.035*direction], [ranks[j, endpoint], target],
                    color=colour, alpha=.85, lw=.65, clip_on=False, zorder=2)
            texts.append(ax.text(px, target, f"{iso}  {int(ranks[j, endpoint])}",
                color=colour if highlighted else "#66757f", ha=anchor, va="center",
                fontsize=7.8, weight="medium" if highlighted else "normal"))
        LABEL_BOXES.append(texts)
    ax.set(xlim=(-2.13, 6.26), ylim=(high, .5),
           xticks=x, xticklabels=["Open\ntraditional", "0\nArctic option", "1", "2", "3", "4", "5"],
           yticks=[1] + list(range(20, high, 20)))
    ax.set_ylabel("Country rank", fontsize=8.5, labelpad=4)
    ax.tick_params(axis="x", labelsize=8, length=0, pad=5)
    ax.tick_params(axis="y", labelsize=7.8, length=2, colors="#6b7881")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#d6dde0")
    ax.grid(axis="y", lw=.5, color="#e7ebed", zorder=0)
    return ranks


def heatmap(ax, values, ids, names, norm, cmap, columns, signed=True, font=8):
    ax.imshow(values, aspect="auto", norm=norm, cmap=cmap, interpolation="nearest")
    for row in range(len(ids)):
        for col in range(values.shape[1]):
            value = int(values[row, col])
            text = f"{value:+d}" if signed and value else str(value)
            rgba = cmap(norm(value))
            luminance = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(col, row, text, ha="center", va="center", fontsize=font,
                    color="white" if luminance < .53 else "#3f505a")
    ax.set(xticks=np.arange(len(columns)), xticklabels=columns,
           yticks=np.arange(len(ids)), yticklabels=[label(names[iso]) for iso in ids])
    ax.tick_params(axis="both", length=0, labelsize=8.2, pad=5)
    for tick, iso in zip(ax.get_yticklabels(), ids):
        if iso in ACCENTS:
            tick.set_color(ACCENTS[iso])
            tick.set_fontweight("medium")
    ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ids), 1), minor=True)
    ax.grid(which="minor", color="white", lw=.65)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main_figure(data, selection):
    fig = plt.figure(figsize=(17, 14.5))
    fig.text(.325, .983, "Country rank trajectories", ha="center", fontsize=11)
    fig.text(.845, .983, "Arctic-only rank gain", ha="center", fontsize=11)
    gains = []
    for ship, ids in selection.items():
        for step in range(6):
            gains.extend((table(data, ship, "traditional", step).loc[ids, "rank"]
                         - table(data, ship, "embedded_distance", step).loc[ids, "rank"]).tolist())
    limit = max(10, int(np.ceil(np.abs(gains).max()/10)*10))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    records, panel_metadata = [], []
    for row, ship in enumerate(SHIPS):
        bottom = .756-row*.214
        base = table(data, ship, "traditional", 0)
        ids = selection[ship]
        month = int(base.month.iloc[0])
        ax = fig.add_axes([.058, bottom, .556, .177])
        ranks = rank_trajectories(ax, data, ship, ids)
        ax.text(-.075, 1.025, chr(97+row*2), transform=ax.transAxes, fontsize=12, weight="bold")
        fig.text(.011, bottom+.0885, f"{NAMES[ship]} | {calendar.month_abbr[month]}",
                 rotation=90, ha="center", va="center", fontsize=10.5, weight="normal")
        if row == len(SHIPS)-1:
            ax.set_xlabel("Closed chokepoints", fontsize=8.6, labelpad=3)
        gain = np.column_stack([(table(data, ship, "traditional", k).loc[ids, "rank"]
            - table(data, ship, "embedded_distance", k).loc[ids, "rank"]).to_numpy() for k in range(6)])
        ax2 = fig.add_axes([.735, bottom, .237, .177])
        heatmap(ax2, gain, ids, base.name.to_dict(), norm, GAIN_CMAP, range(6))
        ax2.text(-.43, 1.025, chr(98+row*2), transform=ax2.transAxes, fontsize=12, weight="bold")
        if row == len(SHIPS)-1:
            ax2.set_xlabel("Closed chokepoints", fontsize=8.6, labelpad=3)
        for j, iso in enumerate(ids):
            records.append(dict(shiptype=ship, iso3=iso, name=base.loc[iso, "name"],
                original_rank=int(ranks[j, 0]), embedded_open_rank=int(ranks[j, 1]),
                **{f"embedded_step{k}_rank": int(ranks[j, k+1]) for k in range(1, 6)},
                **{f"arctic_only_gain_step{k}": int(gain[j, k]) for k in range(6)},
                net_gain_vs_original=int(ranks[j, 0]-ranks[j, -1])))
        panel_metadata.append(dict(shiptype=ship, month=month, countries=ids,
            rank_axis_limits=[1, int(np.ceil(ranks.max()/10)*10)+4]))
    fig.legend(handles=[Line2D([], [], lw=2, color=colour, label=label(table(data, "bulk", "traditional", 0).loc[iso, "name"]))
        for iso, colour in ACCENTS.items()] + [Line2D([], [], lw=1.3, color=BACKGROUND, label="Other displayed countries"),
            Line2D([], [], lw=0, marker="o", mfc="white", mec="#697d88", label="Zero support; tied rank")],
        loc="lower center", bbox_to_anchor=(.5, .014), ncol=8, frameon=False,
        fontsize=8, handlelength=1.7, columnspacing=1.2)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=GAIN_CMAP),
        cax=fig.add_axes([.738, .052, .231, .009]), orientation="horizontal")
    cb.set_ticks([-limit, -40, 0, 40, limit])
    cb.ax.tick_params(labelsize=7.8, length=2)
    cb.set_label("Conventional rank - Arctic rank (same closures)", fontsize=8.2, labelpad=3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    endpoint_boxes = []
    for pair in LABEL_BOXES:
        for text in pair:
            bb = text.get_window_extent(renderer)
            assert bb.x0 >= 0 and bb.x1 <= fig.bbox.width and bb.y0 >= 0 and bb.y1 <= fig.bbox.height
            endpoint_boxes.append((text.axes, text.get_ha(), bb))
    intersections = 0
    for j, (axis, side, box) in enumerate(endpoint_boxes):
        for other_axis, other_side, other in endpoint_boxes[j+1:]:
            if axis is other_axis and side == other_side and box.overlaps(other):
                intersections += 1
    assert intersections == 0, "Endpoint country labels overlap"
    save(fig, "MainFig_CountryRank_Dynamics")
    pd.DataFrame(records).to_csv(OUT/"main_rank_dynamics_source_data.csv", index=False)
    (OUT/"rank_dynamics_design.json").write_text(json.dumps(dict(
        independent_main_figure=True, replaces_existing_main_figure_3=False,
        maps_included=False, absolute_centrality_panels_included=False,
        rank_universe=int(data.iso3.nunique()), rows=4, columns=2, panels=panel_metadata,
        line_states=["Open conventional", "Open embedded"]+[f"Embedded after {k} closures" for k in range(1,6)],
        heatmap_reference="Same-closure conventional rank minus embedded rank; positive = higher rank with Arctic option",
        selection="First four entries in original and embedded-five rankings; then up to two additional Arctic-only gainers and two decliners with a top-30 rank in at least one endpoint or five-closure counterfactual; filled to 12 by rank range; alphabetical tie breaks",
        selection_uses_absolute_scores=False, country_order="Original conventional rank, then name",
        display_exclusion=["TWN"], exclusion_changes_underlying_ranks=False,
        no_spline_interpolation=True, endpoints_not_jittered=True, label_overlaps=intersections,
        rank_ties="Competition rank after rounding percentage scores to 10 decimal places",
        inspiration=dict(title="Dynamics of ranking", journal="Nature Communications", year=2022,
            doi="10.1038/s41467-022-29256-x", figure="1a and 1c", borrowed="Rank trajectories, top rank above lower ranks; not its stochastic ranking model")), indent=2))


def radius_figure(data, selection):
    radii = [25, 50, 100, 200]
    rank_count = int(data.iso3.nunique())
    magnitudes = [abs(int(table(data, ship, "traditional", 5, r).loc[iso, "rank"]
        - table(data, ship, "embedded_distance", 5, r).loc[iso, "rank"]))
        for ship, ids in selection.items() for iso in ids for r in radii]
    limit = max(10, int(np.ceil(max(magnitudes)/10)*10))
    fig = plt.figure(figsize=(13.6, 12.6))
    fig.text(.30, .98, "Country rank after five closures", ha="center", fontsize=10.5)
    fig.text(.79, .98, "Arctic-only rank gain", ha="center", fontsize=10.5)
    for row, ship in enumerate(SHIPS):
        ids = selection[ship]
        names = table(data, ship, "traditional", 0).name.to_dict()
        bottom = .745-row*.213
        ranks = np.column_stack([table(data, ship, "embedded_distance", 5, r).loc[ids, "rank"] for r in radii])
        gain = np.column_stack([(table(data, ship, "traditional", 5, r).loc[ids, "rank"]
            - table(data, ship, "embedded_distance", 5, r).loc[ids, "rank"]) for r in radii])
        for col, (values, norm, cmap) in enumerate([
            (ranks, Normalize(1, rank_count), RANK_CMAP),
            (gain, TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), GAIN_CMAP)]):
            ax = fig.add_axes([.16+col*.49, bottom, .30, .175])
            heatmap(ax, values, ids, names, norm, cmap, radii, signed=bool(col), font=8)
            ax.text(-.45, 1.025, chr(97+2*row+col), transform=ax.transAxes, weight="bold", fontsize=11)
            if row == len(SHIPS)-1:
                ax.set_xlabel("Coastal proximity (km)", fontsize=8.5)
        fig.text(.012, bottom+.0875, NAMES[ship], rotation=90, ha="center", va="center", fontsize=10)
    for x, norm, cmap, name in [(.16, Normalize(1, rank_count), RANK_CMAP, "Rank (1 = highest)"),
        (.65, TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), GAIN_CMAP, "Conventional rank - Arctic rank")]:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
            cax=fig.add_axes([x, .053, .30, .011]), orientation="horizontal")
        cb.set_label(name, fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    save(fig, "SI_CoastalRadius_CountryRankSensitivity")


def cost_figure(data, selection):
    fig, axs = plt.subplots(2, 2, figsize=(13.2, 9.4))
    fig.subplots_adjust(left=.16, right=.98, top=.95, bottom=.10, hspace=.32, wspace=.60)
    rows = []
    for k, (ax, ship) in enumerate(zip(axs.ravel(), SHIPS)):
        a = table(data, ship, "embedded_distance", 5)
        b = table(data, ship, "embedded_premium05", 5).loc[a.index]
        ids = selection[ship]
        y = np.arange(len(ids))
        for row, iso in enumerate(ids):
            ax.plot([a.loc[iso, "rank"], b.loc[iso, "rank"]], [row, row],
                    color="#c3cdd2", lw=1.3, zorder=1)
        ax.scatter(a.loc[ids, "rank"], y, s=27, marker="o", facecolor="#4b98a1",
                   edgecolor="white", lw=.55, zorder=3)
        ax.scatter(b.loc[ids, "rank"], y, s=27, marker="D", facecolor="#b97480",
                   edgecolor="white", lw=.55, zorder=4)
        high = int(np.ceil(max(a.loc[ids, "rank"].max(), b.loc[ids, "rank"].max())/10)*10)
        ax.set(xlim=(-2, high+3), ylim=(len(ids)-.5, -.5),
            yticks=y, yticklabels=[label(a.loc[iso, "name"]) for iso in ids],
            xlabel="Country rank (1 = highest)")
        ax.tick_params(axis="y", labelsize=8, length=0, pad=5)
        ax.text(-.42, 1.025, chr(97+k), transform=ax.transAxes, weight="bold", fontsize=11)
        ax.text(.0, 1.025, NAMES[ship], transform=ax.transAxes, fontsize=10.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.13, lw=.5)
        for iso in a.index:
            rows.append(dict(shiptype=ship, iso3=iso,
                distance_rank=int(a.loc[iso, "rank"]), premium05_rank=int(b.loc[iso, "rank"]),
                rank_gain_with_premium=int(a.loc[iso, "rank"]-b.loc[iso, "rank"])))
    fig.legend(handles=[
        Line2D([], [], lw=0, marker="o", color="#4b98a1", label="Physical-distance route choice"),
        Line2D([], [], lw=0, marker="D", color="#b97480", label="5% Arctic route-choice premium")],
        loc="lower center", bbox_to_anchor=(.5, .025), frameon=False, ncol=2, fontsize=9)
    save(fig, "SI_Distance_vs_CostQualified_CountryRanks")
    pd.DataFrame(rows).to_csv(OUT/"cost_selection_rank_sensitivity.csv", index=False)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.labelcolor": "#40525d",
        "text.color": "#283c48", "xtick.color": "#526571", "ytick.color": "#526571"})
    data = pd.read_csv(OUT/"country_betweenness_all_states.csv")
    selection = {ship: select_countries(data, ship) for ship in SHIPS}
    data[["shiptype", "month", "step", "network", "radius_km", "iso3", "name", "rank"]].to_csv(
        OUT/"country_rank_only_all_states.csv", index=False)
    changes = pd.read_csv(OUT/"country_rank_changes.csv")
    changes[["shiptype", "step", "radius_km", "iso3", "name", "open_original_rank",
        "traditional_rank", "embedded_rank", "rank_gain", "rank_gain_vs_open"]].to_csv(
        OUT/"country_rank_only_changes.csv", index=False)
    main_figure(data, selection)
    radius_figure(data, selection)
    cost_figure(data, selection)


if __name__ == "__main__":
    main()
