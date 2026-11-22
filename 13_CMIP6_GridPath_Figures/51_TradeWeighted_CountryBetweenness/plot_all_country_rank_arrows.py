"""All-country tapered dumbbells, inspired by Lindersson et al. Fig. 2."""
from pathlib import Path
import calendar
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd

from plot_rank_dynamics import table, SHIPS, NAMES, label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
FIG = HERE / "figures"
GAIN, LOSS, SAME = "#518eae", "#df746b", "#a7b0b7"
MISSING = "#c9ced2"
ALIASES = {
    "Antigua and Barbuda": "Antigua & Barbuda",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Democratic Republic of the Congo": "DR Congo",
    "Republic of the Congo": "Congo Republic",
    "Saint Kitts and Nevis": "St Kitts & Nevis",
    "Saint Vincent and the Grenadines": "St Vincent & Grenadines",
    "Saint Lucia": "St Lucia", "Trinidad and Tobago": "Trinidad & Tobago",
    "Federated States of Micronesia": "Micronesia",
    "S\u00e3o Tom\u00e9 and Principe": "Sao Tome & Principe",
    "United Arab Emirates": "UAE", "United Republic of Tanzania": "Tanzania",
}
REGION_BLOCKS = [["Europe", "Americas", "Oceania"], ["Asia", "Africa"]]


def country_metadata(data):
    world = gpd.read_file(ROOT / "data/public/naturalearth_10m_countries/ne_10m_admin_0_countries.shp")
    world["admin_iso3"] = world.ISO_A3_EH.where(world.ISO_A3_EH.ne("-99"), world.ISO_A3)
    world["admin_iso3"] = world.admin_iso3.where(world.admin_iso3.ne("-99"), world.ADM0_A3)
    mapping = pd.read_csv(OUT / "country_code_mapping.csv")
    assert mapping.groupby("admin_iso3").iso3.nunique().eq(1).all()
    mapping = mapping[["admin_iso3", "iso3"]].drop_duplicates()
    world = world.merge(mapping, on="admin_iso3", validate="many_to_one")
    world["parent"] = world.ADMIN.eq(world.SOVEREIGNT)
    world = world.sort_values("parent", ascending=False).drop_duplicates("iso3").set_index("iso3")
    records = data.drop_duplicates("iso3").set_index("iso3")[["name"]]
    records["region"] = world.REGION_UN.reindex(records.index)
    records["display_name"] = [ALIASES.get(n, label(n)) for n in records.name]
    records["displayed"] = records.index != "TWN"
    records["rank_universe_count"] = len(records)
    sort_state = data[data.network.eq("embedded_distance") & data.step.eq(5)
        & data.radius_km.eq(100)].pivot(index="iso3", columns="shiptype", values="rank")
    sort_state = sort_state.reindex(index=records.index, columns=SHIPS)
    assert sort_state.notna().all().all()
    records["sort_mean_arctic_five_rank"] = sort_state.mean(axis=1)
    ordered = []
    for regions in REGION_BLOCKS:
        for region in regions:
            group = records[records.displayed & records.region.eq(region)].sort_values(
                ["sort_mean_arctic_five_rank", "display_name"])
            records.loc[group.index, "order_within_region"] = np.arange(1, len(group)+1)
            ordered.extend(group.index.tolist())
    assert len(records) == 155 and records.region.notna().all()
    records.reset_index().to_csv(OUT / "all_country_rank_display_metadata.csv", index=False)
    return records.loc[ordered]


def layout_rows(metadata):
    blocks = []
    for regions in REGION_BLOCKS:
        rows, y = [], 0.
        for region in regions:
            group = metadata[metadata.region.eq(region)].sort_values(
                ["sort_mean_arctic_five_rank", "display_name"])
            rows.append(dict(kind="region", name=region, y=y))
            y += 1.35
            for iso, country in group.iterrows():
                rows.append(dict(kind="country", name=country.display_name, iso3=iso, y=y))
                y += 1.
            y += .45
        blocks.append(dict(regions=regions, rows=rows, height=y))
    return blocks


def source_rows(data, metadata, step, reference, radius=100):
    records = []
    baseline_step = step if reference == "same_closure" else 0
    for ship in SHIPS:
        old = table(data, ship, "traditional", baseline_step, radius).loc[metadata.index]
        new = table(data, ship, "embedded_distance", step, radius).loc[metadata.index]
        for iso in metadata.index:
            old_support = bool(round(float(old.loc[iso, "betweenness_pct"]), 10) > 0)
            new_support = bool(round(float(new.loc[iso, "betweenness_pct"]), 10) > 0)
            gain = int(old.loc[iso, "rank"]-new.loc[iso, "rank"])
            category = "unsupported" if not (old_support or new_support) else (
                "gain" if gain > 0 else "loss" if gain < 0 else "unchanged")
            records.append(dict(shiptype=ship, month=int(old.loc[iso, "month"]),
                step=step, radius_km=radius, reference=reference, reference_step=baseline_step,
                iso3=iso, country=metadata.loc[iso, "display_name"], region=metadata.loc[iso, "region"],
                conventional_rank=int(old.loc[iso, "rank"]), arctic_rank=int(new.loc[iso, "rank"]),
                rank_gain=gain, reference_has_support=old_support, arctic_has_support=new_support,
                pair_has_support=old_support or new_support, visual_category=category))
    return pd.DataFrame(records)


def figure(data, metadata, blocks, step=5, reference="same_closure", radius=100):
    source = source_rows(data, metadata, step, reference, radius)
    lookup = source.set_index(["shiptype", "iso3"])
    fig = plt.figure(figsize=(17.5, 14.0))
    max_height = max(block["height"] for block in blocks)
    country_texts = []
    for block_id, block in enumerate(blocks):
        left = .018+block_id*.50
        width = .465
        bottom, height = .105, .828
        fig.text(left+width*.5, .985, " | ".join(block["regions"]),
            ha="center", va="center", fontsize=10.5, weight="bold", color="#526672")
        # One shared country column between two rank panels on either side.
        panel_rel = [(0., .185), (.205, .185), (.610, .185), (.815, .185)]
        country_ax = fig.add_axes([left+width*.397, bottom, width*.206, height])
        country_ax.set(xlim=(0, 1), ylim=(max_height-.25, -1))
        country_ax.set_axis_off()
        for row in block["rows"]:
            if row["kind"] == "region":
                country_ax.text(.5, row["y"], row["name"].upper(), ha="center", va="center",
                    fontsize=8.1, color="#526672", weight="bold")
            else:
                country_texts.append(country_ax.text(.5, row["y"], row["name"],
                    ha="center", va="center", fontsize=7.25, color="#31444f"))
        for col, (ship, (x, w)) in enumerate(zip(SHIPS, panel_rel)):
            ax = fig.add_axes([left+width*x, bottom, width*w, height])
            ax.set(xlim=(-4, 160), ylim=(max_height-.25, -1), yticks=[],
                xticks=[1, 50, 100, 155], xticklabels=["1", "50", "100", "155"])
            ax.set_xlabel("Country rank", fontsize=8.1, labelpad=3)
            ax.tick_params(axis="x", length=0, pad=4, labelsize=7.3)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis="x", color="#cbd4da", lw=.5, zorder=0)
            ax.text(.0, 1.029, chr(97+block_id*4+col), transform=ax.transAxes,
                fontsize=11, weight="bold", ha="left")
            month = int(source.loc[source.shiptype.eq(ship), "month"].iloc[0])
            ax.text(.5, 1.029, f"{NAMES[ship]} | {calendar.month_abbr[month]}", transform=ax.transAxes,
                fontsize=8.8, ha="center")
            for row in block["rows"]:
                y = row["y"]
                if row["kind"] == "region":
                    ax.axhspan(y-.5, y+.5, color="#f1f4f5", lw=0, zorder=0)
                    continue
                z = lookup.loc[(ship, row["iso3"])]
                if not z.pair_has_support:
                    ax.text(.5, y, "-", transform=ax.get_yaxis_transform(), ha="center", va="center",
                        fontsize=7.7, color=MISSING)
                    continue
                old, new = z.conventional_rank, z.arctic_rank
                colour = GAIN if z.rank_gain > 0 else LOSS if z.rank_gain < 0 else SAME
                ax.plot([157, -1], [y, y], color="#b9c4ca", lw=.38, ls=(0, (3, 4)), alpha=.50, zorder=0)
                if old != new:
                    ax.add_patch(Polygon([(old, y), (new, y-.24), (new, y+.24)],
                        closed=True, facecolor=colour, edgecolor="none", alpha=.52, zorder=1))
                ax.scatter([old], [y], s=4.6,
                    facecolor=colour if z.reference_has_support else "white",
                    edgecolor=colour, lw=.55, zorder=2)
                ax.scatter([new], [y], s=15.0,
                    facecolor=colour if z.arctic_has_support else "white",
                    edgecolor=colour, lw=.6, zorder=3)
    reference_label = "Without Arctic (same closures)" if reference == "same_closure" else "Open conventional reference"
    handles = [
        Line2D([], [], marker="o", lw=0, markersize=2.7, color="#7a8992", label=reference_label),
        Line2D([], [], marker="o", lw=0, markersize=4.6, color="#7a8992", label="With Arctic"),
        Patch(facecolor=GAIN, alpha=.7, label="Rank improves (moves left)"),
        Patch(facecolor=LOSS, alpha=.7, label="Rank declines (moves right)"),
        Patch(facecolor=SAME, alpha=.7, label="Same rank"),
        Line2D([], [], marker="o", mfc="white", mec="#7a8992", lw=0, markersize=4.5, label="Zero-support endpoint"),
        Line2D([], [], marker="_", color=MISSING, lw=0, markersize=7, label="No transit support in either network"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .039),
        ncol=4, frameon=False, fontsize=8.2, columnspacing=1.8, handlelength=1.6)
    closure_note = f"Both networks: {step} vessel-specific chokepoints closed" if reference == "same_closure" else f"Open conventional (0 closures) vs Arctic ({step} closures)"
    fig.text(.5, .018, f"{closure_note} | {radius}-km coastal proximity | Rank 1 = highest; improvement moves left",
        ha="center", fontsize=8.3, color="#586e7a")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in country_texts:
        bb = t.get_window_extent(renderer)
        ab = t.axes.get_window_extent(renderer)
        assert bb.x0 >= ab.x0 and bb.x1 <= ab.x1, (t.get_text(), "country label exceeds column")
    for j, t in enumerate(country_texts):
        for other in country_texts[j+1:]:
            if t.axes is other.axes:
                assert not t.get_window_extent(renderer).overlaps(other.get_window_extent(renderer))
    return fig, source


def save_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG/(stem+"."+ext), dpi=300, facecolor="white")
    plt.close(fig)
    print("Saved", stem, flush=True)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": "#31444f",
        "xtick.color": "#637782", "axes.labelcolor": "#516873"})
    data = pd.read_csv(OUT/"country_betweenness_all_states.csv")
    metadata = country_metadata(data)
    blocks = layout_rows(metadata)
    fig, primary = figure(data, metadata, blocks, step=5, reference="same_closure")
    save_figure(fig, "MainFig_AllCountry_RankArrows")
    primary.to_csv(OUT/"all_country_rank_arrows_primary.csv", index=False)
    fig, original = figure(data, metadata, blocks, step=5, reference="original_open")
    save_figure(fig, "SI_AllCountry_RankArrows_OriginalReference")
    original.to_csv(OUT/"all_country_rank_arrows_original_reference.csv", index=False)
    states = []
    pairs = [(step, 100) for step in range(6)]+[(5, radius) for radius in [25, 50, 200]]
    with PdfPages(FIG/"SI_AllCountry_RankArrows_Stages_and_CoastalSensitivity.pdf") as pdf:
        for step, radius in pairs:
            fig, source = figure(data, metadata, blocks, step=step,
                reference="same_closure", radius=radius)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            states.append(source)
    pd.concat(states, ignore_index=True).to_csv(OUT/"all_country_rank_arrows_stages_and_sensitivity.csv", index=False)
    names = metadata.index.tolist()
    assert len(names) == 154 and primary.groupby("shiptype").size().eq(154).all()
    summary = primary.groupby(["shiptype", "visual_category"]).size().unstack(fill_value=0)
    summary.to_csv(OUT/"all_country_rank_arrows_category_counts.csv")
    (OUT/"all_country_rank_arrows_design.json").write_text(json.dumps(dict(
        independent_main_figure=True, existing_main_figures_1_to_5_modified=False,
        display_country_count=len(names), full_rank_universe=155, vessels=SHIPS,
        countries=names, geography=REGION_BLOCKS,
        panels=[dict(label=chr(97+block_id*4+col), shiptype=ship,
            vessel=NAMES[ship], representative_month=int(table(data,ship,"traditional",0).month.iloc[0]),
            regions=regions)
            for block_id,regions in enumerate(REGION_BLOCKS) for col,ship in enumerate(SHIPS)],
        country_order="Ascending mean Arctic-embedded rank across four vessel types at five closures and 100 km; country name breaks ties",
        order_reference_fixed_across_main_and_SI=True,
        order_aggregation="Unweighted mean of four ranks, used only for shared row order; not a new centrality score or trade-weighted country ranking",
        main_comparison="Conventional and Arctic-embedded under identical five-closure sets",
        complementary_comparison="Open conventional versus Arctic-embedded after five closures",
        supplementary_stages=list(range(6)), supplementary_coastal_radii=[25,50,100,200],
        supplementary_pages=[dict(closures=step,radius_km=radius) for step,radius in pairs],
        ranks_recomputed_on_display_subset=False,
        unsupported_pair="Dash, no plotted rank movement; actual tied ranks retained in data",
        one_zero_endpoint="Hollow marker; numeric tied rank retained",
        x_axis="1 to 155, ascending left to right; rank improvement leftward",
        x_axis_limits=[-4,160], x_axis_ticks=[1,50,100,155],
        direction_colours=dict(gain=GAIN, loss=LOSS, unchanged=SAME),
        number_of_primary_pairs=len(primary), country_label_overlaps=0,
        reference=dict(authors="Lindersson et al.", journal="Nature Sustainability", year=2023,
            title="The wider the gap between rich and poor the higher the flood mortality",
            doi="10.1038/s41893-023-01107-7", figure=2,
            borrowed="Small baseline dots, larger comparison dots, tapered connecting polygons and central country labels; no source-paper observations reused")), indent=2))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
