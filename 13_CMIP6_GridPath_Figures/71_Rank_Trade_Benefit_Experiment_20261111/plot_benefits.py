"""Draw paired country incidence across the four vessel-commodity markets."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
FOCUS = ("CHN", "USA", "RUS", "GBR", "DEU")
LABELS = {"container":"Container / electronics", "bulk":"Dry bulk / metals",
          "tanker":"Tanker / chemicals", "lng":"LNG / natural gas"}


def select(lookup, ship):
    data = [r for (s, i, c), r in lookup.items() if s == ship and c == "joint" and r["loss_avoided_pp_mean"]]
    absolute = lambda r: float(r["loss_avoided_m_mean"])
    best = sorted((r for r in data if absolute(r) > 0), key=absolute, reverse=True)[:2]
    worst = sorted((r for r in data if absolute(r) < 0), key=absolute)[:2]
    names = set(FOCUS) | {r["iso3"] for r in best+worst}
    return sorted((r for r in data if r["iso3"] in names),
                  key=lambda r: float(r["loss_avoided_pp_mean"]), reverse=True)


def main():
    with (HERE/"country_benefits.csv").open() as f:
        lookup = {(r["ship"],r["iso3"],r["contrast"]):r for r in csv.DictReader(f)}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42})
    fig, axs = plt.subplots(2,2,figsize=(10.35,6.95))
    fig.subplots_adjust(left=.10,right=.975,top=.93,bottom=.16,hspace=.48,wspace=.28)
    selected = {ship:select(lookup,ship) for ship in LABELS}
    config_low = "loss_avoided_pp_min_config_low"
    config_high = "loss_avoided_pp_max_config_high"
    maximum = max(abs(float(lookup[ship,j["iso3"],c][m])) for ship in LABELS for j in selected[ship]
                  for c in ("joint", "rank_at_arctic") for m in (config_low,config_high))
    limit = max(.6, 1.18*maximum)
    ticks = [t for t in (-10,-5,-2,-1,-.2,0,.2,1,2,5,10) if abs(t) < limit*.98]
    if limit > 20:
        ticks = [-20]+ticks+[20]
    styles = [("joint","#146b8c","o",-.15),("rank_at_arctic","#be562d","D",.15)]
    for ax, (ship, rows), letter in zip(axs.flat,selected.items(),"abcd"):
        yy = np.arange(len(rows))
        ax.set_xscale("symlog",linthresh=.20,linscale=.65)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda t,p: f"{t:g}".replace("-","\u2212")))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(-limit,limit)
        ax.set_axisbelow(True)
        ax.axvspan(-.1,.1,color="#ebedef",zorder=0)
        ax.grid(axis="x",color="#e0e5e8",linewidth=.55)
        ax.axvline(0,color="#596570",linewidth=.7)
        ax.hlines(yy,-limit,limit,color="#edf0f1",linewidth=.55,zorder=1)
        for contrast,color,marker,offset in styles:
            v = [lookup[ship,j["iso3"],contrast] for j in rows]
            mean = np.array([float(j["loss_avoided_pp_mean"]) for j in v])
            low = [float(j[config_low]) for j in v]
            high = [float(j[config_high]) for j in v]
            ax.hlines(yy+offset, low, high, color=color,alpha=.7,linewidth=1.0,zorder=3)
            ax.scatter(mean, yy+offset, color=color, marker=marker, s=20, zorder=4,
                       edgecolors="white",linewidths=.32)
        ax.set_yticks(yy,[r["iso3"] for r in rows])
        ax.set_ylim(len(yy)-.52,-.56)
        ax.tick_params(axis="both",length=0,pad=5)
        ax.set_title(LABELS[ship],loc="left",fontsize=10.5,fontweight="medium",pad=7)
        ax.text(-.145,1.04,letter,transform=ax.transAxes,ha="left",va="bottom",fontweight="bold",fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.54,.105,"Avoided import disruption burden (percentage points of reference imports)",
             ha="center",fontsize=10)
    fig.legend([Line2D([],[],color=z[1],marker=z[2],linewidth=1.1,markersize=5) for z in styles],
               ["Routes and ranks updated","Rank update only (Arctic routes fixed)"],
               ncols=2,loc="lower center",bbox_to_anchor=(.54,.012),frameon=False,fontsize=9)
    for fmt in ("png","pdf"):
        fig.savefig(HERE/f"Supplementary_Figure_8.{fmt}",dpi=350,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
