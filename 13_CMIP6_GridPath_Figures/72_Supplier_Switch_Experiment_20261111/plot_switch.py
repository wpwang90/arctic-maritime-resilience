"""Render the procurement and incidence panels from the audited source tables."""
import csv
import itertools

import numpy as np

from switch_market import HERE

SHIPS = {"container":"Container electronics", "bulk":"Dry-bulk metals",
         "tanker":"Tanker chemicals", "lng":"LNG natural gas"}
FOCUS = {"CHN":"China","DEU":"Germany","USA":"United States","GBR":"United Kingdom","RUS":"Russia"}


def read(filename):
    with (HERE/filename).open(newline="") as f:
        return list(csv.DictReader(f))


def plot():
    import matplotlib.pyplot as plt

    market = read("market_switch_effects.csv")
    country = {(r["ship"],r["contrast"],r["iso3"]):r for r in read("country_switch_benefits.csv")}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none",
                         "axes.spines.right":False,"axes.spines.top":False})
    fig = plt.figure(figsize=(7.1,5.85))
    gs = fig.add_gridspec(2,2, height_ratios=[1.17,1], wspace=.16, hspace=.48,
                          left=.244, right=.97, bottom=.19, top=.93)
    a,b,c = fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])
    colors = {"proposal_change_pct":"#7A4E87","lead_switch_pct":"#007D81",
              "joint":"#007D81","rank_at_arctic":"#B95338"}
    def dot(ax,z,y,key,label=None):
        ax.hlines(y,min(z),max(z),color=colors[key],lw=1.3)
        ax.scatter(np.mean(z),y,color=colors[key],s=23,label=label,zorder=3)
    for i,(ship,kind) in enumerate(itertools.product(SHIPS,("proposal_change_pct","lead_switch_pct"))):
        rows = [r for r in market if r["ship"]==ship and r["setting"]=="primary" and r["contrast"]=="rank_at_arctic"]
        dot(a,[float(r[kind+"_mean"]) for r in rows],i//2+(-.15 if i%2==0 else .15),kind,
            ("Proposed contract changes","Realised lead supplier switches")[i] if i<2 else None)
    a.set_yticks(range(4),SHIPS.values())
    a.set_ylim(3.5,-.5)
    a.set_xlim(left=0)
    a.set_xlabel("Importers changing supplier decisions at fixed Arctic routes (%)")
    a.legend(loc="lower right",frameon=False,fontsize=7.4)
    for ax,field in ((b,"loss_reduction_pct"),(c,"export_change_pp")):
        for i,iso in enumerate(FOCUS):
            for contrast,delta,marker in (("joint",-.13,"o"),("rank_at_arctic",.13,"D")):
                val = country["bulk",contrast,iso][field]
                if val:
                    ax.scatter(float(val),i+delta,marker=marker,color=colors[contrast],s=19,zorder=3)
        ax.set_ylim(4.5,-.5)
        ax.set_yticks(range(5),FOCUS.values() if ax is b else [""]*5)
        ax.axvline(0,color=".38",lw=.8)
    b.set_xlabel("Import burden avoided (%)")
    c.set_xlabel("Export delivery change (pp)")
    for ax,label in zip((a,b,c),"abc",strict=True):
        ax.text(-.04,1.06,label,transform=ax.transAxes,fontweight="bold",fontsize=11)
        ax.grid(axis="x",color=".86",lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both",length=2,pad=4)
    handles = [plt.Line2D([],[],color=colors[k],marker=m,linestyle="",markersize=4.5,label=label)
               for k,m,label in (("joint","o","Routes and ranks change"),("rank_at_arctic","D","Only ranks change"))]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.60,.052),ncol=2,frameon=False,fontsize=7.6)
    for ext in ("png","pdf"):
        fig.savefig(HERE/f"Supplementary_Figure_9.{ext}",dpi=300)


if __name__=="__main__":
    plot()
