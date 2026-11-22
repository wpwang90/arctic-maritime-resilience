"""Validate new route geometries and render admission-threshold SI figures."""
import calendar
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from compute_sensitivity import HERE, ROOT, RESULTS, ROUTES, THRESHOLDS, KEYS, routing, digest

FIGURES = HERE / "figures"
CASES = [(2050,"ssp245"),(2050,"ssp585"),(2100,"ssp245"),(2100,"ssp585")]


def validate():
    audit = json.loads((RESULTS / "calculation_audit.json").read_text())
    candidates = pd.read_csv(RESULTS / "rerouted_candidate_admission.csv")
    members = pd.read_csv(RESULTS / "threshold_model_year_evidence.csv")
    paths = pd.read_parquet(RESULTS / "threshold_candidate_paths.parquet")
    group_keys = KEYS+["search_threshold_pct"]
    m = members.groupby(group_keys).path_passable.agg(["sum","size"])
    c = candidates.rename(columns={"threshold_pct":"search_threshold_pct"}).set_index(group_keys).join(m)
    assert (c["size"]==60).all() and (c["sum"]==c.supported_model_years).all()
    assert (c.admitted == c.supported_model_years.ge(c.required_model_years)).all()
    assert (c.common_path_support_pct <= c.known_model_years/60*100+1e-10).all()
    for path, value in audit["protected_sha256"].items():
        assert digest(path)==value
    land = routing.prepare_land()
    geometry = []
    for route in ROUTES:
        grid = dict(np.load(routing.OUT / f"geometry_{route}.npz"))
        edges = set(zip(grid["row"].tolist(),grid["col"].tolist()))
        for path_id, p in paths[paths.route.eq(route)].groupby("path_id",sort=False):
            # An identical node sequence can be reused in several scenarios.
            one = next(iter(p.groupby(group_keys)))[1].sort_values("order")
            ids, xy = one.node.to_numpy(), one[["lon","lat"]].to_numpy()
            assert len(ids)==len(set(ids))
            assert np.allclose(xy,grid["points"][ids])
            assert set(ids[[0,-1]])==set(grid["endpoints"])
            assert all((a,b) in edges or (b,a) in edges for a,b in zip(ids[:-1],ids[1:]))
            assert not routing.crosses_land(land,xy[:-1],xy[1:]).any()
            if route=="TSR":
                assert xy[:,1].max()>=85
            geometry.append(dict(route=route,path_id=path_id,nodes=len(ids),segments=len(ids)-1,
                                 land_crossings=0,repeated_nodes=0))
    pd.DataFrame(geometry).to_csv(RESULTS / "independent_geometry_validation.csv",index=False)
    audit.update(status="validated",unique_geometries_checked=len(geometry),
                 unique_segments_checked=sum(g["segments"] for g in geometry),
                 land_crossings=0,repeated_nodes=0,member_counts_independently_checked=True,
                 originals_unchanged=True)
    (RESULTS / "calculation_audit.json").write_text(json.dumps(audit,indent=2))
    print("Validated",len(geometry),"unique geometries",flush=True)


def cosmetics(ax, title, letter):
    ax.tick_params(length=0,pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(.5,1.16,title,transform=ax.transAxes,ha="center",fontsize=10)
    ax.text(-.09,1.16,letter,transform=ax.transAxes,weight="bold",fontsize=14)


def case_label(period,ssp):
    return f"{2041 if period==2050 else 2091}-{period} | "+{"ssp245":"SSP2-4.5","ssp585":"SSP5-8.5"}[ssp]


def save(fig,stem):
    for ext in ["png","pdf","svg"]:
        fig.savefig(FIGURES / f"{stem}.{ext}",dpi=260,facecolor="white")
    plt.close(fig)


def count_panels(annual, method, stem, delta=False):
    data = annual.query("screen == 'ice_capable' and method == @method")
    fig,axes = plt.subplots(2,2,figsize=(7.3,5.4))
    fig.subplots_adjust(left=.10,right=.985,bottom=.24,top=.88,wspace=.25,hspace=.65)
    colors = plt.get_cmap("YlGnBu")(np.linspace(0,1,256))
    colors[0] = matplotlib.colors.to_rgba("#eeeeee")
    cmap = ListedColormap(colors)
    vmax = 12
    if delta:
        maximum = (annual.pivot(index=KEYS[:-1]+["threshold_pct"],columns="method",values="admitted_months")
                   .assign(difference=lambda x:x.candidate_pool-x.fixed)).difference.max()
        vmax = max(int(maximum),1)
    for i,(period,ssp) in enumerate(CASES):
        ax = axes.flat[i]
        z = data.query("period == @period and experiment == @ssp").pivot(index="route",columns="threshold_pct",values="admitted_months").loc[ROUTES,THRESHOLDS]
        if delta:
            base = annual.query("period == @period and experiment == @ssp and screen == 'ice_capable' and method == 'fixed'")
            z = z-base.pivot(index="route",columns="threshold_pct",values="admitted_months").loc[ROUTES,THRESHOLDS]
        mesh = ax.pcolormesh(range(5),range(5),z,cmap=cmap,vmin=0,vmax=vmax,edgecolors="white",linewidth=1)
        for r in range(4):
            for c in range(4):
                val = int(z.iloc[r,c])
                ax.text(c+.5,r+.5,f"+{val}" if delta and val else str(val),ha="center",va="center",
                        weight="bold",fontsize=11,color="white" if val>=.65*vmax else "#17282f")
        ax.set_xticks(np.arange(4)+.5,[f"{t}%" for t in THRESHOLDS],fontsize=10)
        ax.set_yticks(np.arange(4)+.5,ROUTES,fontsize=10)
        ax.set_ylim(4,0)
        cosmetics(ax,case_label(period,ssp),"abcd"[i])
    cbax=fig.add_axes([.24,.10,.56,.026])
    cb=fig.colorbar(mesh,cax=cbax,orientation="horizontal",ticks=np.arange(0,vmax+1,1 if delta else 2))
    cb.outline.set_visible(False)
    cb.set_label("Additional admitted months after candidate search" if delta else "Admitted calendar months (out of 12)",fontsize=10)
    fig.text(.53,.185,"Whole-path support threshold",ha="center",fontsize=10)
    save(fig,stem)


def monthly_panels(fixed):
    fig,axes=plt.subplots(4,1,figsize=(7.3,7.4))
    fig.subplots_adjust(left=.10,right=.99,bottom=.16,top=.94,hspace=.75)
    colors=plt.get_cmap("YlGnBu")(np.linspace(0,1,256)); colors[0]=matplotlib.colors.to_rgba("#eeeeee")
    base=fixed.query("period == 2100 and experiment == 'ssp585' and screen == 'ice_capable'")
    for i,tau in enumerate(THRESHOLDS):
        ax=axes[i]
        data=base[base.threshold_pct.eq(tau)]
        z=data.pivot(index="route",columns="month",values="common_path_support_pct").loc[ROUTES,range(1,13)]
        ok=data.pivot(index="route",columns="month",values="admitted").loc[ROUTES,range(1,13)]
        exists=data.pivot(index="route",columns="month",values="common_candidate_exists").loc[ROUTES,range(1,13)]
        mesh=ax.pcolormesh(range(13),range(5),z,cmap=ListedColormap(colors),vmin=0,vmax=100,edgecolors="white",linewidth=.8)
        for r in range(4):
            for c in range(12):
                value=z.iloc[r,c]
                if value>0 and not ok.iloc[r,c]:
                    ax.add_patch(Rectangle((c,r),1,1,fill=False,hatch="///",edgecolor="#bac6c9",linewidth=0))
                label=f"{value:.1f}".rstrip("0").rstrip(".") if value else "0"
                if not exists.iloc[r,c]:
                    label+=r"$^{\dagger}$"
                ax.text(c+.5,r+.5,label,ha="center",va="center",fontsize=8.5,
                        color="white" if value>=65 else "#17282f",weight="bold" if ok.iloc[r,c] else "normal")
        ax.set_xticks(np.arange(12)+.5,[calendar.month_abbr[m] for m in range(1,13)],fontsize=9)
        ax.set_yticks(np.arange(4)+.5,ROUTES,fontsize=10)
        ax.set_ylim(4,0)
        cosmetics(ax,f"Admission threshold {tau}% ({60*tau//100}/60)","abcd"[i])
    cax=fig.add_axes([.25,.105,.55,.018])
    cb=fig.colorbar(mesh,cax=cax,orientation="horizontal",ticks=[0,20,40,60,80,100])
    cb.outline.set_visible(False)
    cb.set_label("Common-path model-year support (%)",fontsize=9)
    fig.legend(handles=[Patch(facecolor="#e5eeee",edgecolor="#b5c1c6",hatch="///",label="Positive support below admission threshold"),
                        Patch(facecolor="#eeeeee",edgecolor="none",label=r"0$^{\dagger}$: no baseline candidate found")],
               loc="lower center",bbox_to_anchor=(.5,-.005),ncol=1,frameon=False,fontsize=8)
    save(fig,"SupplementaryFig15_Monthly_Threshold_Admission")


def search_exceptions():
    data=pd.read_csv(RESULTS / "additional_months_found_by_candidate_search.csv")
    data=data.sort_values(["threshold_pct","screen","period","experiment","route","month"])
    y=np.arange(len(data))
    before=data.supported_model_years.to_numpy()/60*100
    after=data.maximum_supported_model_years.to_numpy()/60*100
    fig,ax=plt.subplots(figsize=(8.1,5.2))
    fig.subplots_adjust(left=.49,right=.98,bottom=.14,top=.85)
    for i,(a,b) in enumerate(zip(before,after)):
        ax.plot([a,b],[i,i],color="#9aabad",lw=1.3,zorder=1)
    ax.scatter(before,y,s=35,color="#3478a6",label="Original candidate",zorder=3)
    ax.scatter(after,y,s=40,color="#d4813a",marker="D",label="Best support in candidate pool",zorder=3)
    ax.scatter(data.threshold_pct,y,s=100,marker="|",color="#1d2a2d",label="Admission threshold",zorder=4)
    names={"ice_capable":"Main", "low_ice_sensitivity":"Low ice", "observed_thickness_sensitivity":"No positive-SIC fill"}
    labels=[f"{r.period} | {r.experiment.upper().replace('SSP245','SSP2-4.5').replace('SSP585','SSP5-8.5')} | {r.route} | {calendar.month_abbr[r.month]} | {names[r.screen]}"
            for r in data.itertuples(index=False)]
    ax.set_yticks(y,labels,fontsize=8)
    ax.set_ylim(len(y)-.4,-.6)
    ax.set_xlim(-3,80)
    ax.set_xticks([0,20,40,60,80])
    ax.set_xlabel("Common-path model-year support (%)",fontsize=10)
    ax.tick_params(axis="y",length=0,pad=8)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x",color="#e5e5e5",lw=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.99),frameon=False,ncol=1,fontsize=9)
    save(fig,"SupplementaryFig16_Candidate_Search_Sensitivity")


def main():
    validate()
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"Arial","pdf.fonttype":42,"svg.fonttype":"none","hatch.linewidth":.35})
    annual=pd.read_csv(RESULTS / "annual_admission_summary.csv")
    fixed=pd.read_csv(RESULTS / "fixed_candidate_admission.csv")
    count_panels(annual,"fixed","SupplementaryFig14_Admission_Threshold_Season_Length")
    monthly_panels(fixed)
    search_exceptions()
    print("Rendered three supplementary figures",flush=True)


if __name__=="__main__":
    main()
