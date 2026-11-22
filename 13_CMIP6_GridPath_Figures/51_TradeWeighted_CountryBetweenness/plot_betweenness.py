"""Country transit centrality maps, rank reordering and definition sensitivity."""
from pathlib import Path
import calendar
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm,LinearSegmentedColormap,Normalize,TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
FIG=HERE/"figures"
SHIPS=["container","bulk","tanker","lng"]
NAMES=dict(container="Container",bulk="Dry bulk",tanker="Tanker",lng="LNG")
BLUE,ORANGE,GREY="#3985a5","#d6814c","#9babad"
PALETTE=["#fffefa","#f1e6ae","#cddd9c","#92c4b5","#78aecb","#8d91be","#735e96","#493b69"]
SEQUENTIAL=LinearSegmentedColormap.from_list("coastal",PALETTE,N=1025)
SEQUENTIAL.set_bad("#d5d9db")
DIVERGING=LinearSegmentedColormap.from_list("rank_gain",["#b15b7b","#e6b799","#f4f3ee","#9ac8c4","#625b95"],N=1025)
SCORE_NORM=FuncNorm((np.log1p,np.expm1),vmin=0,vmax=100)
SHORT={"United States of America":"United States","United Republic of Tanzania":"Tanzania", "Republic of the Congo":"Congo", "Democratic Republic of the Congo":"DR Congo", "The Bahamas":"Bahamas","Republic of Serbia":"Serbia"}


def save(fig,name):
    for ext in ["png","pdf","svg"]:
        fig.savefig(FIG/(name+"."+ext),dpi=300,facecolor="white")
    plt.close(fig)
    print("Saved",name,flush=True)


def table(data,ship,network,step,radius=100):
    return data[data.shiptype.eq(ship)&data.network.eq(network)&data.step.eq(step)&data.radius_km.eq(radius)].set_index("iso3")


def select_countries(data,ship):
    a=table(data,ship,"traditional",0)
    b=table(data,ship,"embedded_distance",5)
    a=a.drop(index="TWN",errors="ignore")
    b=b.loc[a.index]
    selected=list(dict.fromkeys(a.sort_values(["rank","name"]).head(10).index.tolist()+b.sort_values(["rank","name"]).head(10).index.tolist()))
    strength=pd.concat([a.betweenness_pct,b.betweenness_pct],axis=1).max(axis=1)
    for iso in strength.sort_values(ascending=False).index:
        if iso not in selected:
            selected.append(iso)
        if len(selected)>=20:
            break
    return a.loc[selected].sort_values(["rank","name"]).index.tolist()


def map_panel(ax,world,values,bounds):
    value=world.iso3.map(values)
    colours=SEQUENTIAL(SCORE_NORM(np.ma.masked_invalid(value.to_numpy(float))))
    world.plot(ax=ax,color=colours,edgecolor=(.28,.37,.41,.20),linewidth=.24)
    ax.set(xlim=bounds[:2],ylim=bounds[2:],aspect="equal",xticks=[],yticks=[])
    ax.set_axis_off()


def main_figure(data,world,selection):
    fig=plt.figure(figsize=(22,14.2))
    for x,label in [(.017,"Open conventional network"),(.253,"Arctic option after five closures"),(.564,"Country rank comparison"),(.805,"Arctic-only rank gain")]:
        fig.text(x,.977,label,fontsize=10.8,weight="bold")
    bb=world.total_bounds
    bounds=[bb[0]-.4,bb[2]+.4,bb[1]-.2,bb[3]+.2]
    all_gain=[]
    for ship,ids in selection.items():
        for step in range(1,6):
            a=table(data,ship,"traditional",step).loc[ids]
            b=table(data,ship,"embedded_distance",step).loc[ids]
            all_gain.extend((a["rank"]-b["rank"]).tolist())
    limit=max(10,int(np.ceil(np.max(np.abs(all_gain))/10)*10))
    rank_norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    selected_rows=[]
    panels=[]
    for i,ship in enumerate(SHIPS):
        ids=selection[ship]
        base=table(data,ship,"traditional",0).loc[ids]
        trad=table(data,ship,"traditional",5).loc[ids]
        arc=table(data,ship,"embedded_distance",5).loc[ids]
        top=.944-i*.207
        month=int(base.month.iloc[0])
        fig.text(.017,top,NAMES[ship]+" | "+calendar.month_abbr[month],fontsize=10.8,weight="bold")
        for j,(x,values) in enumerate([(.008,table(data,ship,"traditional",0).betweenness_pct),(.244,table(data,ship,"embedded_distance",5).betweenness_pct)]):
            ax=fig.add_axes([x,top-.190,.224,.163])
            map_panel(ax,world,values.drop(index="TWN",errors="ignore"),bounds)
            ax.text(.004,1.02,chr(97+4*i+j),transform=ax.transAxes,fontsize=11,weight="bold")
        y=np.arange(len(ids))
        ax=fig.add_axes([.565,top-.183,.177,.164])
        for k,iso in enumerate(ids):
            ax.plot([trad.loc[iso,"rank"],arc.loc[iso,"rank"]],[k,k],lw=1.05,color="#bfcbd0",zorder=1)
        ax.scatter(base["rank"],y,s=15,facecolor="white",edgecolor=GREY,lw=.85,zorder=2)
        ax.scatter(trad["rank"],y,s=17,color=BLUE,zorder=3)
        ax.scatter(arc["rank"],y,s=18,color=ORANGE,zorder=4)
        high=int(np.ceil(max(base["rank"].max(),trad["rank"].max(),arc["rank"].max())/10)*10)
        ax.set(xlim=(0,high+3),ylim=(len(ids)-.5,-.5),yticks=y,
            yticklabels=[SHORT.get(n,n) for n in base.name],xlabel="Rank (1 = highest)")
        ax.tick_params(axis="y",labelsize=7.0,length=0,pad=4)
        ax.tick_params(axis="x",labelsize=7.2)
        ax.xaxis.label.set_size(8)
        ax.spines[["right","top","left"]].set_visible(False)
        ax.grid(axis="x",alpha=.2,lw=.5)
        ax.text(-.40,1.025,chr(99+4*i),transform=ax.transAxes,fontsize=11,weight="bold")
        gain=np.column_stack([(table(data,ship,"traditional",step).loc[ids,"rank"]-table(data,ship,"embedded_distance",step).loc[ids,"rank"]).to_numpy() for step in range(1,6)])
        ax=fig.add_axes([.800,top-.183,.181,.164])
        ax.imshow(gain,aspect="auto",cmap=DIVERGING,norm=rank_norm,interpolation="nearest")
        for iy in range(len(ids)):
            for ix in range(5):
                v=int(gain[iy,ix])
                ax.text(ix,iy,f"{v:+d}" if v else "0",ha="center",va="center",fontsize=6.8,
                    color="white" if abs(v)>.68*limit else "#384b54")
        ax.set(xticks=np.arange(5),xticklabels=np.arange(1,6),yticks=[],xlabel="Closed chokepoints")
        ax.tick_params(labelsize=7.4,length=0)
        ax.xaxis.label.set_size(8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(-.12,1.025,chr(100+4*i),transform=ax.transAxes,fontsize=11,weight="bold")
        for iso in ids:
            selected_rows.append(dict(shiptype=ship,iso3=iso,name=base.loc[iso,"name"],
                open_rank=int(base.loc[iso,"rank"]),traditional_five_rank=int(trad.loc[iso,"rank"]),
                arctic_five_rank=int(arc.loc[iso,"rank"]),open_pct=base.loc[iso,"betweenness_pct"],
                traditional_five_pct=trad.loc[iso,"betweenness_pct"],arctic_five_pct=arc.loc[iso,"betweenness_pct"],
                rank_gain_vs_same_closure=int(trad.loc[iso,"rank"]-arc.loc[iso,"rank"]),
                rank_gain_vs_open=int(base.loc[iso,"rank"]-arc.loc[iso,"rank"]),
                betweenness_change_vs_open_pp=arc.loc[iso,"betweenness_pct"]-base.loc[iso,"betweenness_pct"]))
        panels.append(dict(shiptype=ship,month=month,countries=ids))
    cbax=fig.add_axes([.045,.078,.416,.012])
    cb=fig.colorbar(ScalarMappable(norm=SCORE_NORM,cmap=SEQUENTIAL),cax=cbax,orientation="horizontal")
    cb.set_ticks([0,1,5,10,25,50,100],labels=["0","1","5","10","25","50","100"])
    cb.minorticks_off()
    cb.set_label("Trade-weighted coastal transit betweenness (%; log10(1 + x) scale)",fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cbax=fig.add_axes([.800,.078,.181,.012])
    cb=fig.colorbar(ScalarMappable(norm=rank_norm,cmap=DIVERGING),cax=cbax,orientation="horizontal")
    cb.set_label("Rank gain: conventional - Arctic",fontsize=8.1)
    cb.ax.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([],[],marker="o",lw=0,mfc="white",mec=GREY,label="Open conventional"),
        Line2D([],[],marker="o",lw=0,color=BLUE,label="Conventional, 5 closures"),
        Line2D([],[],marker="o",lw=0,color=ORANGE,label="Arctic option, 5 closures"),
        Patch(facecolor="#d5d9db",label="No coastline / not displayed")],
        loc="lower center",bbox_to_anchor=(.5,.026),ncol=4,fontsize=8.5,frameon=False)
    fig.text(.5,.010,"100-km coastal proximity; origin/destination countries excluded. Routes are distance-shortest feasible candidates; rankings concern represented vessel-month trade, not maritime jurisdiction.",ha="center",fontsize=8.1,color="#536771")
    save(fig,"MainFig_CountryBetweenness_RankReordering")
    pd.DataFrame(selected_rows).to_csv(OUT/"main_selected_country_rankings.csv",index=False)
    (OUT/"figure_design.json").write_text(json.dumps(dict(rows=4,columns=4,panels=panels,
        countries_per_vessel=20,selection="Union of top 10 open and top 10 embedded-five countries, filled to 20 by maximum score, sorted by open rank",
        rank_change_reference="Same closed chokepoint set, conventional versus embedded distance-shortest network",
        map_metric_range=[0,100],rank_gain_colour_limit=limit,excluded_from_display=["TWN"]),indent=2))


def radius_figure(data,selection):
    radii=[25,50,100,200]
    rank_cmap=LinearSegmentedColormap.from_list("rank",["#f4edb6","#9ec8bd","#7b9ec3","#685785"],N=256)
    fig=plt.figure(figsize=(13.8,15.0))
    magnitudes=[]
    for ship,ids in selection.items():
        for radius in radii:
            magnitudes.extend((table(data,ship,"traditional",5,radius).loc[ids,"rank"]-
                table(data,ship,"embedded_distance",5,radius).loc[ids,"rank"]).abs().tolist())
    limit=max(10,int(np.ceil(max(magnitudes)/10)*10))
    for i,ship in enumerate(SHIPS):
        ids=selection[ship]
        top=.955-i*.218
        base=table(data,ship,"traditional",0).loc[ids]
        fig.text(.07,top,NAMES[ship],fontsize=11,weight="bold")
        ranks=np.column_stack([table(data,ship,"embedded_distance",5,r).loc[ids,"rank"] for r in radii])
        gain=np.column_stack([table(data,ship,"traditional",5,r).loc[ids,"rank"]-table(data,ship,"embedded_distance",5,r).loc[ids,"rank"] for r in radii])
        for j,values in enumerate([ranks,gain]):
            ax=fig.add_axes([.18+j*.43,top-.185,.29,.164])
            norm=Normalize(1,163) if j==0 else TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
            cmap=rank_cmap if j==0 else DIVERGING
            ax.imshow(values,aspect="auto",norm=norm,cmap=cmap,interpolation="nearest")
            for iy in range(20):
                for ix in range(4):
                    value=int(values[iy,ix])
                    ax.text(ix,iy,str(value) if j==0 else f"{value:+d}" if value else "0",ha="center",va="center",fontsize=7.2,
                        color="white" if (j==0 and value>105) or (j==1 and abs(value)>55) else "#344c56")
            ax.set(xticks=np.arange(4),xticklabels=radii,yticks=np.arange(20) if j==0 else [],
                yticklabels=[SHORT.get(n,n) for n in base.name] if j==0 else [],xlabel="Coastal proximity threshold (km)")
            ax.tick_params(labelsize=7.3,length=0)
            ax.xaxis.label.set_size(8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(-.40 if j==0 else -.08,1.03,chr(97+2*i+j),transform=ax.transAxes,fontsize=11,weight="bold")
    fig.text(.18,.984,"Rank after five closures with Arctic options",fontsize=10,weight="bold")
    fig.text(.61,.984,"Arctic-only rank gain at five closures",fontsize=10,weight="bold")
    for key,rect,norm,cmap,label in [(0,[.18,.055,.29,.012],Normalize(1,163),rank_cmap,"Rank (1 = highest)"),
        (1,[.61,.055,.29,.012],TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),DIVERGING,"Conventional rank - Arctic rank")]:
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes(rect),orientation="horizontal")
        cb.set_label(label,fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.text(.5,.012,"Countries and row order match the main figure. Coastal proximity is a geographic definition, not a change to territorial-water boundaries.",ha="center",fontsize=8.1,color="#536771")
    save(fig,"SI_CoastalRadius_CountryRankSensitivity")


def cost_figure(data):
    fig,axs=plt.subplots(2,2,figsize=(11.5,9.5))
    fig.subplots_adjust(left=.09,right=.97,top=.94,bottom=.10,hspace=.34,wspace=.24)
    rows=[]
    for ax,ship in zip(axs.ravel(),SHIPS):
        a=table(data,ship,"embedded_distance",5)
        b=table(data,ship,"embedded_premium05",5).loc[a.index]
        ax.scatter(a["rank"],b["rank"],s=18,color="#638eaa",alpha=.55,edgecolor="none")
        ax.plot([0,165],[0,165],color="#87999f",lw=.8,ls="--")
        for iso in ["RUS","CAN","DNK","EGY","GBR","USA","CHN"]:
            if iso in a.index and (a.loc[iso,"rank"]<70 or b.loc[iso,"rank"]<70):
                ax.annotate(iso,(a.loc[iso,"rank"],b.loc[iso,"rank"]),xytext=(3,3),textcoords="offset points",fontsize=7)
        ax.set(xlim=(0,165),ylim=(0,165),xlabel="Rank: distance-shortest",ylabel="Rank: 5% Arctic premium")
        ax.set_title(NAMES[ship],loc="left",fontsize=11,weight="bold")
        ax.spines[["top","right"]].set_visible(False)
        for iso in a.index:
            rows.append(dict(shiptype=ship,iso3=iso,distance_rank=int(a.loc[iso,"rank"]),premium05_rank=int(b.loc[iso,"rank"]),
                distance_pct=a.loc[iso,"betweenness_pct"],premium05_pct=b.loc[iso,"betweenness_pct"]))
    fig.text(.5,.025,"Same closure sequence, country definition, OD weights and 100-km proximity. This sensitivity changes Arctic route selection, not shortest-path edge distances.",ha="center",fontsize=8.3,color="#536771")
    save(fig,"SI_Distance_vs_CostQualified_CountryRanks")
    pd.DataFrame(rows).to_csv(OUT/"cost_selection_rank_sensitivity.csv",index=False)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
    data=pd.read_csv(OUT/"country_betweenness_all_states.csv")
    world=gpd.read_parquet(OUT/"coastal_countries.parquet").to_crs(8857)
    world.geometry=world.geometry.simplify(12000,preserve_topology=True)
    world.geometry=world.geometry.scale(xfact=1e-6,yfact=1e-6,origin=(0,0))
    world=world.set_crs(None,allow_override=True)
    selection={ship:select_countries(data,ship) for ship in SHIPS}
    main_figure(data,world,selection)
    radius_figure(data,selection)
    cost_figure(data)


if __name__=="__main__":
    main()
