"""Independent vessel top-50 ranks: population circles and betweenness diamonds."""
from pathlib import Path
import calendar
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd

from plot_all_country_rank_arrows import source_rows, GAIN, LOSS, SAME, MISSING
from plot_rank_dynamics import SHIPS, NAMES
from plot_top50_rank_population import REGIONS, circle_area

HERE = Path(__file__).resolve().parent
OUT,FIG = HERE/"results",HERE/"figures"


def betweenness_area(score_pct):
    return np.asarray(score_pct,dtype=float)


def layouts():
    selected = pd.read_csv(OUT/"vessel_top50_selection.csv")
    metadata,rows,heights = {},{},{}
    for ship in SHIPS:
        m = selected[selected.shiptype.eq(ship)].set_index("iso3")
        r,ids,y = [],[],0.
        for region in REGIONS:
            group = m[m.region.eq(region)].sort_values(["selection_arctic_rank","display_name"])
            if group.empty:
                continue
            r.append(dict(kind="region",name=region,y=y))
            y += 1.5
            for iso,row in group.iterrows():
                r.append(dict(kind="country",iso3=iso,name=row.display_name,y=y))
                ids.append(iso)
                y += 1
            y += .5
        assert len(ids) == len(set(ids)) == 50
        metadata[ship],rows[ship],heights[ship] = m.loc[ids],r,y
    return metadata,rows,max(heights.values())


def figure(data,metadata,rows,total_height,step=5,radius=100,reference="same_closure"):
    pieces = []
    for ship in SHIPS:
        m = metadata[ship]
        z = source_rows(data,m,step,reference,radius)
        z = z[z.shiptype.eq(ship)].merge(m[["selection_position","selection_arctic_rank","population",
            "population_year","population_source","population_source_url","population_status"]],
            left_on="iso3",right_index=True,validate="one_to_one")
        old_step = step if reference == "same_closure" else 0
        old = data[data.shiptype.eq(ship)&data.network.eq("traditional")
            &data.step.eq(old_step)&data.radius_km.eq(radius)].set_index("iso3")
        new = data[data.shiptype.eq(ship)&data.network.eq("embedded_distance")
            &data.step.eq(step)&data.radius_km.eq(radius)].set_index("iso3")
        z["conventional_betweenness_pct"] = z.iso3.map(old.betweenness_pct)
        z["arctic_betweenness_pct"] = z.iso3.map(new.betweenness_pct)
        z["start_population_area_pt2"] = circle_area(z.population)
        z["end_betweenness_area_pt2"] = betweenness_area(z.arctic_betweenness_pct)
        z["start_marker"] = "o"
        z["end_marker"] = "D"
        z["end_zero_support_reference_symbol"] = ~z.arctic_has_support
        pieces.append(z)
    source = pd.concat(pieces,ignore_index=True)
    fig = plt.figure(figsize=(18,13))
    bottom,height,ymax = .17,.77,total_height-.25
    labels = []
    axes = []
    for col,ship in enumerate(SHIPS):
        left,width = .108+.245*col,.132
        ax = fig.add_axes([left,bottom,width,height])
        axes.append(ax)
        ax.set(xlim=(-4,160),ylim=(ymax,-1),yticks=[],xticks=[1,50,100,155],
            xticklabels=["1","50","100","155"])
        ax.set_xlabel("Country rank",fontsize=8.5,labelpad=4)
        ax.tick_params(axis="x",length=0,labelsize=8,pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x",color="#cbd4da",lw=.5,zorder=0)
        g = source[source.shiptype.eq(ship)].set_index("iso3")
        month = int(g.month.iloc[0])
        ax.text(-.72,1.025,chr(97+col),transform=ax.transAxes,fontsize=12,weight="bold")
        ax.text(.2,1.025,f"{NAMES[ship]} | {calendar.month_abbr[month]}",
            transform=ax.transAxes,ha="center",fontsize=10)
        for row in rows[ship]:
            y = row["y"]
            is_region = row["kind"] == "region"
            t = ax.text(-.025,y,row["name"].upper() if is_region else row["name"],
                transform=ax.get_yaxis_transform(),ha="right",va="center",
                fontsize=8.1 if is_region else 7.5,weight="bold" if is_region else "normal",
                color="#526672" if is_region else "#31444f",clip_on=False)
            labels.append((col,t))
            if is_region:
                ax.axhspan(y-.5,y+.5,color="#f1f4f5",lw=0,zorder=0)
                continue
            z = g.loc[row["iso3"]]
            old,new = z.conventional_rank,z.arctic_rank
            colour = (GAIN if z.rank_gain > 0 else LOSS if z.rank_gain < 0 else SAME) if z.pair_has_support else MISSING
            ax.plot([-1,157],[y,y],color="#b9c4ca",lw=.38,ls=(0,(3,4)),alpha=.50,zorder=0)
            if old != new:
                ax.add_patch(Polygon([(old,y),(new,y-.23),(new,y+.23)],closed=True,
                    facecolor=colour,edgecolor="none",alpha=.52,zorder=1))
            ax.scatter([old],[y],s=z.start_population_area_pt2,marker="o",
                facecolor=to_rgba(colour,.28) if z.reference_has_support else "none",
                edgecolor=colour,lw=.55,zorder=4)
            if z.arctic_has_support:
                ax.scatter([new],[y],s=z.end_betweenness_area_pt2,marker="D",
                    facecolor=to_rgba(colour,.85),edgecolor=colour,lw=.4,zorder=3)
            else:
                # This hollow locator is not part of the quantitative size scale.
                ax.scatter([new],[y],s=5,marker="D",facecolor="none",
                    edgecolor=colour,lw=.5,zorder=5)
    reference_label = "Without Arctic (same closures)" if reference == "same_closure" else "Open conventional reference"
    legend = fig.legend(handles=[
        Line2D([],[],marker="o",lw=0,markersize=5,mfc=to_rgba("#7a8992",.28),mec="#7a8992",
            label=reference_label+"; population"),
        Line2D([],[],marker="D",lw=0,markersize=5,color="#7a8992",label="With Arctic; weighted betweenness"),
        Patch(facecolor=GAIN,alpha=.7,label="Rank improves (left)"),
        Patch(facecolor=LOSS,alpha=.7,label="Rank declines (right)"),
        Patch(facecolor=SAME,alpha=.7,label="Same rank"),
        Line2D([],[],marker="o",mfc="none",mec="#7a8992",lw=0,markersize=4,label="Hollow: zero transit support")],
        loc="lower center",bbox_to_anchor=(.5,.085),ncol=3,frameon=False,fontsize=8.2,
        handlelength=1.6,columnspacing=1.8)
    pop_handles = [Line2D([],[],marker="o",lw=0,mfc=to_rgba("#8799a5",.28),mec="#8799a5",mew=.4,
        markersize=float(np.sqrt(circle_area(p))),label=f"{int(p/1e6):,} million") for p in [10e6,100e6,1000e6]]
    pop_legend = fig.legend(handles=pop_handles,title="Start circle: population (2024)",
        loc="lower center",bbox_to_anchor=(.28,.032),ncol=3,frameon=False,fontsize=8,title_fontsize=8,
        handlelength=1.8,columnspacing=2)
    b_handles = [Line2D([],[],marker="D",lw=0,mfc=to_rgba("#8799a5",.85),mec="#8799a5",mew=.4,
        markersize=float(np.sqrt(betweenness_area(b))),label=f"{b:g}%") for b in [1,10,50]]
    b_legend = fig.legend(handles=b_handles,title="End diamond: trade-weighted betweenness",
        loc="lower center",bbox_to_anchor=(.73,.032),ncol=3,frameon=False,fontsize=8,title_fontsize=8,
        handlelength=1.8,columnspacing=2)
    note = f"Both networks: {step} vessel-specific chokepoints closed" if reference == "same_closure" else f"Open conventional vs Arctic after {step} closures"
    fig.text(.5,.013,f"Independent top 50 per vessel | {note} | {radius}-km coastal proximity | 1 = highest rank",
        ha="center",fontsize=8,color="#586e7a")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for j,(col,t) in enumerate(labels):
        b = t.get_window_extent(renderer)
        assert b.x0 >= 0 and b.x1 <= fig.bbox.width,t.get_text()
        if col > 0:
            assert b.x0 > axes[col-1].get_window_extent(renderer).x1,t.get_text()
        for other_col,other in labels[j+1:]:
            if col == other_col:
                assert not b.overlaps(other.get_window_extent(renderer)),t.get_text()
    assert not legend.get_window_extent(renderer).overlaps(pop_legend.get_window_extent(renderer))
    assert not legend.get_window_extent(renderer).overlaps(b_legend.get_window_extent(renderer))
    assert not pop_legend.get_window_extent(renderer).overlaps(b_legend.get_window_extent(renderer))
    assert len(fig.axes) == 4
    return fig,source


def save(fig,stem):
    for ext in ["png","pdf","svg"]:
        fig.savefig(FIG/(stem+"."+ext),dpi=300,facecolor="white")
    plt.close(fig)
    print("Saved",stem,flush=True)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none",
        "text.color":"#31444f","xtick.color":"#637782","axes.labelcolor":"#516873"})
    data = pd.read_csv(OUT/"country_betweenness_all_states.csv")
    metadata,rows,total_height = layouts()
    fig,source = figure(data,metadata,rows,total_height)
    save(fig,"MainFig_VesselTop50_CountryRank_Population")
    source.to_csv(OUT/"vessel_top50_rank_population_primary.csv",index=False)
    fig,source = figure(data,metadata,rows,total_height,reference="original_open")
    save(fig,"SI_VesselTop50_CountryRank_Population_OriginalReference")
    source.to_csv(OUT/"vessel_top50_rank_population_original_reference.csv",index=False)
    cases = [(k,100) for k in range(6)]+[(5,r) for r in [25,50,200]]
    sources = []
    with PdfPages(FIG/"SI_VesselTop50_CountryRank_Population_Stages_and_Sensitivity.pdf") as pdf:
        for step,radius in cases:
            fig,source = figure(data,metadata,rows,total_height,step=step,radius=radius)
            pdf.savefig(fig,facecolor="white")
            plt.close(fig)
            sources.append(source)
    pd.concat(sources,ignore_index=True).to_csv(OUT/"vessel_top50_rank_population_stages_and_sensitivity.csv",index=False)
    design = dict(panels=4,countries_per_vessel=50,full_rank_universe=155,
        union_countries=len(set().union(*(set(m.index) for m in metadata.values()))),
        selection="Independent first 50 eligible countries in each vessel's embedded-distance ranking at five closures and 100 km; country name breaks rank ties",
        country_order="Region groups, then ascending vessel-specific embedded rank; name breaks ties",
        vessel_country_order={ship:m.index.tolist() for ship,m in metadata.items()},
        region_counts={ship:m.groupby("region").size().to_dict() for ship,m in metadata.items()},
        colours=dict(gain=GAIN,loss=LOSS,unchanged=SAME),x_axis_ticks=[1,50,100,155],
        population_year=2024,start_marker="o",end_marker="D",
        start_population_area_formula_pt2="60 * population / 1,000,000,000",
        end_betweenness_area_formula_pt2="1 * arctic_betweenness_pct",
        betweenness_common_scale_across_vessels_and_supplements=True,
        betweenness_size_clipping=False,positive_betweenness_size_floor=False,
        zero_betweenness_locator="Hollow diamond of fixed s=5; reference symbol only, excluded from size scale",
        population_size_clipping=False,population_is_not_a_selection_weight=True,
        fixed_country_selection_and_order_in_supplements=True,country_label_overlaps=0,
        original_main_figures_1_to_5_modified=False,
        supplementary_pages=[dict(closures=k,radius_km=r) for k,r in cases])
    (OUT/"vessel_top50_rank_population_design.json").write_text(json.dumps(design,indent=2))


if __name__ == "__main__":
    main()
