"""Four vessel panels, common top 50, population-proportional circle areas."""
from pathlib import Path
import calendar
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd

from plot_all_country_rank_arrows import source_rows, GAIN, LOSS, SAME, MISSING
from plot_rank_dynamics import SHIPS, NAMES

HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
FIG = HERE/"figures"
REGIONS = ["Europe", "Americas", "Oceania", "Asia", "Africa"]
AREA_PER_BILLION = 60.


def rows_and_metadata():
    m = pd.read_csv(OUT/"top50_country_population_2024.csv").set_index("iso3")
    rows, ids, y = [], [], 0.
    for region in REGIONS:
        group = m[m.region.eq(region)].sort_values(["sort_mean_arctic_five_rank", "display_name"])
        if group.empty:
            continue
        rows.append(dict(kind="region", name=region, y=y))
        y += 1.5
        for iso, row in group.iterrows():
            rows.append(dict(kind="country", iso3=iso, name=row.display_name, y=y))
            ids.append(iso)
            y += 1.
        y += .5
    assert len(ids) == len(set(ids)) == 50
    return rows, m.loc[ids], y


def circle_area(population):
    return AREA_PER_BILLION*np.asarray(population)/1e9


def figure(data, rows, metadata, total_height, step=5, reference="same_closure", radius=100):
    source = source_rows(data, metadata, step, reference, radius)
    source = source.merge(metadata[["population", "population_year", "population_source",
        "population_source_url", "population_status", "selection_position", "sort_mean_arctic_five_rank"]],
        left_on="iso3", right_index=True, validate="many_to_one")
    source["circle_area_pt2"] = circle_area(source.population)
    lookup = source.set_index(["shiptype", "iso3"])
    fig = plt.figure(figsize=(12.7, 12.0))
    bottom, height = .185, .755
    ymax = total_height-.25
    names = []
    for row in rows:
        fy = bottom+height*(ymax-row["y"])/(ymax+1)
        names.append(fig.text(.5, fy, row["name"].upper() if row["kind"] == "region" else row["name"],
            ha="center", va="center", fontsize=9 if row["kind"] == "region" else 8,
            weight="bold" if row["kind"] == "region" else "normal", color="#526672" if row["kind"] == "region" else "#31444f"))
    panel_rel = [(0., .185), (.205, .185), (.610, .185), (.815, .185)]
    for col, (ship, (x, w)) in enumerate(zip(SHIPS, panel_rel)):
        ax = fig.add_axes([.03+.94*x, bottom, .94*w, height])
        ax.set(xlim=(-4, 160), ylim=(ymax, -1), yticks=[],
            xticks=[1, 50, 100, 155], xticklabels=["1", "50", "100", "155"])
        ax.set_xlabel("Country rank", fontsize=8.5, labelpad=4)
        ax.tick_params(axis="x", length=0, pad=4, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color="#cbd4da", lw=.5, zorder=0)
        month = int(source.loc[source.shiptype.eq(ship), "month"].iloc[0])
        ax.text(0, 1.028, chr(97+col), transform=ax.transAxes, fontsize=11, weight="bold")
        ax.text(.53, 1.028, f"{NAMES[ship]} | {calendar.month_abbr[month]}",
            transform=ax.transAxes, fontsize=9.5, ha="center")
        for row in rows:
            y = row["y"]
            if row["kind"] == "region":
                ax.axhspan(y-.5, y+.5, color="#f1f4f5", lw=0, zorder=0)
                continue
            z = lookup.loc[(ship, row["iso3"])]
            if not z.pair_has_support:
                ax.text(.5, y, "-", transform=ax.get_yaxis_transform(), ha="center", va="center",
                    fontsize=8.5, color=MISSING)
                continue
            old, new = z.conventional_rank, z.arctic_rank
            colour = GAIN if z.rank_gain > 0 else LOSS if z.rank_gain < 0 else SAME
            ax.plot([-1,157], [y,y], color="#b9c4ca", lw=.38, ls=(0,(3,4)), alpha=.50, zorder=0)
            if old != new:
                ax.add_patch(Polygon([(old,y),(new,y-.23),(new,y+.23)], closed=True,
                    facecolor=colour, edgecolor="none", alpha=.52, zorder=1))
            ax.scatter([old],[y],s=6,marker="D",facecolor=colour if z.reference_has_support else "white",
                edgecolor=colour,lw=.5,zorder=3)
            ax.scatter([new],[y],s=z.circle_area_pt2,marker="o",
                facecolor=colour if z.arctic_has_support else "white",edgecolor=colour,lw=.4,zorder=4)
    reference_label = "Without Arctic (same closures)" if reference == "same_closure" else "Open conventional reference"
    handles = [
        Line2D([],[],marker="D",lw=0,markersize=3,color="#7a8992",label=reference_label),
        Line2D([],[],marker="o",lw=0,markersize=5,color="#7a8992",label="With Arctic; circle area = population"),
        Patch(facecolor=GAIN,alpha=.7,label="Rank improves (left)"),
        Patch(facecolor=LOSS,alpha=.7,label="Rank declines (right)"),
        Patch(facecolor=SAME,alpha=.7,label="Same rank"),
        Line2D([],[],marker="o",mfc="white",mec="#7a8992",lw=0,markersize=4,label="Zero-support endpoint"),
        Line2D([],[],marker="_",lw=0,color=MISSING,markersize=7,label="Neither network has transit support"),
    ]
    legend = fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.078),
        ncol=3,frameon=False,fontsize=8,handlelength=1.5,columnspacing=1.5)
    pop_handles = [Line2D([],[],marker="o",lw=0,mfc="#8799a5",mec="#8799a5",mew=.4,
        markersize=float(np.sqrt(circle_area(p))),label=f"{int(p/1e6):,} million") for p in [10e6,100e6,1000e6]]
    pop_legend = fig.legend(handles=pop_handles,title="Population (2024; circle area proportional to total population)",
        loc="lower center",bbox_to_anchor=(.5,.025),ncol=3,frameon=False,fontsize=8,title_fontsize=8,
        handlelength=1.8,columnspacing=2)
    closure_note = f"Both networks: {step} vessel-specific chokepoints closed" if reference == "same_closure" else f"Open conventional vs Arctic after {step} closures"
    fig.text(.5,.012,f"{closure_note} | {radius}-km coastal proximity | 1 = highest rank",ha="center",fontsize=8,color="#586e7a")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for j,t in enumerate(names):
        b = t.get_window_extent(renderer)
        assert b.x0 >= .403*fig.bbox.width and b.x1 <= .597*fig.bbox.width,t.get_text()
        for other in names[j+1:]:
            assert not b.overlaps(other.get_window_extent(renderer)),t.get_text()
    assert not legend.get_window_extent(renderer).overlaps(pop_legend.get_window_extent(renderer))
    assert len(fig.axes) == 4
    return fig, source


def save(fig, stem):
    for ext in ["png","pdf","svg"]:
        fig.savefig(FIG/(stem+"."+ext),dpi=300,facecolor="white")
    plt.close(fig)
    print("Saved",stem,flush=True)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none",
        "text.color":"#31444f","xtick.color":"#637782","axes.labelcolor":"#516873"})
    data = pd.read_csv(OUT/"country_betweenness_all_states.csv")
    rows,metadata,total_height = rows_and_metadata()
    fig,primary = figure(data,rows,metadata,total_height)
    save(fig,"MainFig_Top50_CountryRank_Population")
    primary.to_csv(OUT/"top50_rank_population_primary.csv",index=False)
    fig,original = figure(data,rows,metadata,total_height,reference="original_open")
    save(fig,"SI_Top50_CountryRank_Population_OriginalReference")
    original.to_csv(OUT/"top50_rank_population_original_reference.csv",index=False)
    cases = [(k,100) for k in range(6)]+[(5,r) for r in [25,50,200]]
    sources = []
    with PdfPages(FIG/"SI_Top50_CountryRank_Population_Stages_and_Sensitivity.pdf") as pdf:
        for step,radius in cases:
            fig,source = figure(data,rows,metadata,total_height,step=step,radius=radius)
            pdf.savefig(fig,facecolor="white")
            plt.close(fig)
            sources.append(source)
    pd.concat(sources,ignore_index=True).to_csv(OUT/"top50_rank_population_stages_and_sensitivity.csv",index=False)
    (OUT/"top50_rank_population_design.json").write_text(json.dumps(dict(
        panels=4,display_country_count=50,full_rank_universe=155,countries=metadata.index.tolist(),
        selection="First 50 displayed statistical units by unweighted mean of four Arctic-embedded ranks at five closures and 100 km; alphabetical ties",
        country_order="Region groups, then ascending mean rank; identical countries and order in all panels and supplements",
        region_order=REGIONS,region_counts=metadata.groupby("region").size().to_dict(),
        colours=dict(gain=GAIN,loss=LOSS,unchanged=SAME),x_axis_ticks=[1,50,100,155],
        population_year=2024,circle_area_formula_pt2="60 * population / 1,000,000,000",
        population_area_offset=0,population_size_clipping=False,population_log_transform=False,
        network_encoding="Fixed-size diamond = without Arctic; population-area circle = with Arctic",
        zero_support="Hollow endpoint or dash if neither network has transit support",
        country_label_overlaps=0,legend_overlaps=0,original_main_figures_1_to_5_modified=False,
        supplementary_pages=[dict(closures=k,radius_km=r) for k,r in cases]),indent=2))


if __name__ == "__main__":
    main()
