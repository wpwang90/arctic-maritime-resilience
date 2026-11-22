"""Multihue log-coloured cartograms with translucent borders and advected gateways."""
from pathlib import Path
import calendar
import hashlib
import json
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import FuncNorm, LinearSegmentedColormap, LogNorm, to_hex, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import shapely

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"46_RegionalTrade_DistancePrototype"))
import build_distance_cartograms as cart

RESULTS=HERE/"results"
FIG=HERE/"figures"
FIG.mkdir(exist_ok=True)
SHIPS=["container","bulk","tanker","lng"]
NAMES=dict(container="Container",bulk="Dry bulk",tanker="Tanker",lng="LNG")
EDGE="#536971"
REF="#b3bec3"
BORDER_ALPHA=.20
REFERENCE_ALPHA=.12
NO_DATA="#d6d9db"
COLOUR_SCALES={}
OLD="#178878"
NEW="#d65a3a"
VERSION="transparent_exact_point_advection_v1"
GATES=pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
GATES["gate_id"]=np.arange(len(GATES))
GP=gpd.GeoSeries(gpd.points_from_xy(GATES.lon,GATES.lat),crs=4326).to_crs(8857)
GATE_XY=np.column_stack([GP.x,GP.y])/1e6
SHORT={0:"Bab el-Mandeb",3:"Cape of Good Hope",5:"Gibraltar",6:"Korea Strait",7:"Lombok",8:"Luzon",11:"Malacca",12:"Mona Passage",15:"Panama",16:"Hormuz",17:"Suez",18:"Sunda",19:"Taiwan Strait",21:"Tsugaru",22:"Windward",23:"Yucatan"}
OFFSETS={0:(-12,-20),3:(-5,-17),5:(-17,20),6:(20,17),7:(28,-14),8:(-20,-26),11:(-35,12),12:(28,12),15:(-24,-21),18:(-24,-21),19:(26,-16),21:(22,19),22:(-42,-14),23:(-34,20)}


def signed_log10(values):
    """Values are in percentage points, with a fixed one-point log offset."""
    values=np.asanyarray(values)
    return np.sign(values)*np.log10(1+np.abs(values))


def inverse_signed_log10(values):
    values=np.asanyarray(values)
    return np.sign(values)*np.expm1(np.abs(values)*np.log(10))


def colour_scale_definitions():
    return dict(
        original=dict(metric="Trade-weighted original mean maritime distance",unit="km",
            transform="log10(distance_km)",vmin=100.,vmax=30000.,
            ticks=[100,300,1000,3000,10000,30000],ticklabels=["0.1","0.3","1","3","10","30"],
            label="Original mean distance (1,000 km; log10 scale)",
            palette_description="Pastel yellow, coral, rose, lavender and blue; decreasing luminance",
            colours=["#fff9da","#e8d49d","#dda58e","#c884a2","#a282b8","#777aaf","#4a608b"]),
        original_relative=dict(metric="100 * (Arctic-option distance / original distance - 1)",unit="percent",
            transform="sign(x) * log10(1 + abs(x) / 1 percentage point)",vmin=-200.,vmax=200.,
            ticks=[-200,-50,-10,-1,0,1,10,50,200],
            ticklabels=["-200","-50","-10","-1","0","+1","+10","+50","+200"],
            label="Distance change vs original (%; signed log10 scale)",
            palette_description="Cool purple-blue-green shortening; off-white zero; warm yellow-coral-rose lengthening",
            colours=["#51436f","#7b79b4","#71aec6","#92c6b5","#e2edc2","#fffefa","#f7e5b2","#ecc18c","#dd9390","#bb6f94","#8e426a"]),
        detour_benefit=dict(metric="100 * (1 - Arctic-option distance / same-closure detour distance)",unit="percent",
            transform="log10(1 + saving_pct / 1 percentage point)",vmin=0.,vmax=50.,
            ticks=[0,1,5,10,20,50],ticklabels=["0","1","5","10","20","50"],
            label="Saving vs same-closure detour (%; log10(1 + x) scale)",
            palette_description="Off-white zero, pale yellow-green, teal, blue and purple; decreasing luminance",
            colours=["#fffefa","#f4edb9","#cddd9c","#91c6b7","#77b0cf","#898fc0","#706095","#493b69"]))


def build_colour_scales():
    scales={}
    for key,spec in colour_scale_definitions().items():
        norm=LogNorm(spec["vmin"],spec["vmax"]) if key=="original" else FuncNorm(
            (signed_log10,inverse_signed_log10),vmin=spec["vmin"],vmax=spec["vmax"])
        cmap=LinearSegmentedColormap.from_list(key,spec["colours"],N=1025)
        cmap.set_bad(NO_DATA)
        scales[key]=(cmap,norm)
    return scales


def colour_key(role,mode):
    return "original" if role=="original" else mode


def colour_values(current,role,mode):
    if role=="original":
        values=current.distance_km.to_numpy(float)
    else:
        values=current.change_pct.to_numpy(float)*(1 if mode=="original_relative" else -1)
        values[np.abs(values)<1e-9]=0.
        if mode=="detour_benefit":
            assert np.nanmin(values)>=-1e-8
            values=np.maximum(values,0.)
    return np.where(current.has_data.to_numpy(bool),values,np.nan)


def face_colours(current,role,mode):
    values=colour_values(current,role,mode)
    cmap,norm=COLOUR_SCALES[colour_key(role,mode)]
    return cmap(norm(np.ma.masked_invalid(values)))


def styling_metadata(audit):
    audit.update(country_fill="quantitative_log10_distance",country_edge=EDGE,
        country_edge_alpha=BORDER_ALPHA,reference_outline_alpha=REFERENCE_ALPHA)
    return audit


class AnchorCartogram(cart.VectorCartogram):
    def __init__(self,*args,anchor_xy,**kwargs):
        self.__dict__["anchor_xy"]=np.asarray(anchor_xy,float).copy()
        super().__init__(*args,**kwargs)

    def _transform(self):
        self.geometry=self.geometry.buffer(0.)
        self.iteration=0
        while self.iteration<self.max_iterations and self.average_error>self.max_average_error:
            features=self._cartogram_features
            factor=self._reduction_factor
            # The auxiliary gateway points receive the exact same force field
            # as every polygon vertex, once per iteration.
            anchors=self._transform_vertices(self.__dict__["anchor_xy"],features,factor)
            geometries=[self._transform_geometry(g,features=features,reduction_factor=factor) for g in self.geometry]
            self.geometry=geometries
            self.__dict__["anchor_xy"]=anchors
            self._invalidate_cached_properties()
            self.iteration+=1
        self.geometry=self.geometry.buffer(0.)


def source_records(case):
    folder=HERE/case["folder"]
    country=pd.read_csv(folder/"country_origin_distances.csv")
    country=country[country.shiptype.eq(case["shiptype"])]
    return folder,country


def distort(world,case,step,mode):
    folder,country=source_records(case)
    destination=RESULTS/"cartograms"/mode/case["shiptype"]/f"to_{int(case['child_count'])}"
    destination.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256((folder/"country_origin_distances.csv").read_bytes()).hexdigest()
    cache=destination/f"step_{step}.parquet"
    marker=destination/f"step_{step}.json"
    anchors=destination/f"step_{step}_gates.csv"
    r=country[country.step.eq(step)].set_index("origin_iso3")
    if cache.exists() and marker.exists():
        audit=json.loads(marker.read_text())
        if audit.get("version")==VERSION:
            saved=gpd.read_parquet(cache)
            expected_distance=saved.iso3.map(r.baseline_km if step==0 else r.with_arctic_km)
            expected_reference=saved.iso3.map(r.baseline_km if step==0 or mode=="original_relative" else r.traditional_km)
            equal=all(np.allclose(a,b,equal_nan=True,rtol=1e-12,atol=1e-12) for a,b in [
                (saved.distance_km,expected_distance),(saved.reference_km,expected_reference),
                (saved.matched_trade_usd,saved.iso3.map(r.matched_trade_usd)),
                (saved.matched_od_count,saved.iso3.map(r.matched_od_count))])
            if equal:
                audit["source_sha256"]=fingerprint
                styling_metadata(audit)
                marker.write_text(json.dumps(audit,indent=2))
                return saved,pd.read_csv(anchors),audit
    data=world.copy()
    data["distance_km"]=data.iso3.map(r.baseline_km if step==0 else r.with_arctic_km)
    reference=r.baseline_km if step==0 or mode=="original_relative" else r.traditional_km
    data["reference_km"]=data.iso3.map(reference)
    data["distance_ratio"]=data.distance_km/data.reference_km
    data["change_pct"]=100*(data.distance_ratio-1)
    data["has_data"]=data.distance_ratio.notna()
    data["matched_trade_usd"]=data.iso3.map(r.matched_trade_usd)
    data["matched_od_count"]=data.iso3.map(r.matched_od_count)
    data["target_area"]=data.reference_area*data.distance_ratio.fillna(1)
    xy=GATE_XY.copy()
    result=data.copy()
    iteration=0
    if not np.allclose(data.distance_ratio.fillna(1),1):
        components=cart.component_targets(data)
        transform=AnchorCartogram(components,"target_area",max_iterations=100,max_average_error=.0005,anchor_xy=xy)
        iteration=int(transform.iteration)
        transformed=gpd.GeoDataFrame(transform,geometry="geometry",crs=None)
        for index,g in transformed.groupby("parent_index"):
            result.loc[index,"geometry"]=shapely.union_all(g.geometry.array)
        scale=np.sqrt(data.target_area.sum()/shapely.area(result.geometry.array).sum())
        result.geometry=shapely.transform(result.geometry.array,lambda p:p*scale)
        xy=transform.__dict__["anchor_xy"]*scale
    result["realized_area"]=shapely.area(result.geometry.array)
    result["area_error"]=np.abs(result.realized_area/result.target_area-1)
    assert shapely.is_valid(result.geometry.array).all()
    assert np.isfinite(xy).all()
    total=result.realized_area.sum()
    union=shapely.area(shapely.union_all(result.geometry.array))
    observed=result[result.has_data]
    major=observed[observed.matched_trade_usd.ge(.01*observed.matched_trade_usd.sum())]
    gates=GATES[["gate_id","key","name","lon","lat"]].copy()
    gates["x"],gates["y"]=xy[:,0],xy[:,1]
    audit=dict(status="PASS",version=VERSION,shiptype=case["shiptype"],step=step,mode=mode,
        source_sha256=fingerprint,source=str((folder/"country_origin_distances.csv").relative_to(ROOT)),
        iterations=iteration,trade_weighted_area_error_pct=100*np.average(observed.area_error,weights=observed.matched_trade_usd),
        max_major_country_area_error_pct=100*major.area_error.max(),max_country_area_error_pct=100*observed.area_error.max(),
        overlap_pct=100*max(0,total-union)/union,
        anchor_method="All 24 chokepoints advected with the same per-iteration DCN force field and final area scale as the polygon vertices.",
        reference="Original open conventional distance" if mode=="original_relative" else "Conventional detour under identical closures",
        country_fill="quantitative_log10_distance",country_edge=EDGE,
        country_edge_alpha=BORDER_ALPHA,reference_outline_alpha=REFERENCE_ALPHA)
    assert audit["trade_weighted_area_error_pct"]<.25
    assert audit["overlap_pct"]<.01
    result.to_parquet(cache,index=False)
    gates.to_csv(anchors,index=False)
    marker.write_text(json.dumps(audit,indent=2))
    print(mode,case["shiptype"],step,"area error",round(audit["trade_weighted_area_error_pct"],4),flush=True)
    return result,gates,audit


def draw_gates(ax,gates,case,role):
    if role=="original":
        return
    selected=json.loads((HERE/case["folder"]/"selected_case.json").read_text())
    old=selected["before"]
    ids=old+([selected["added_gate"]] if role=="after" else [])
    for gate in ids:
        g=gates[gates.gate_id.eq(gate)].iloc[0]
        added=gate==selected["added_gate"]
        color=NEW if added else OLD
        ax.scatter(g.x,g.y,s=25 if added else 20,marker="D" if added else "o",facecolor=color,
            edgecolor="white",linewidth=.5,zorder=9)
        offset=OFFSETS.get(gate,(12,15))
        ax.annotate(SHORT.get(gate,g["name"]),(g.x,g.y),xytext=offset,textcoords="offset points",
            color=color,fontsize=6.6,ha="right" if offset[0]<0 else "left",va="center",
            bbox=dict(boxstyle="square,pad=.08",facecolor="white",edgecolor="none",alpha=.88),
            arrowprops=dict(arrowstyle="-",color=color,lw=.5,shrinkA=1,shrinkB=3),zorder=10)


def panel(ax,base,current,gates,bounds,case,role,mode,labels=True):
    colours=face_colours(current,role,mode)
    current.plot(ax=ax,color=colours,edgecolor=to_rgba(EDGE,BORDER_ALPHA),linewidth=.32,zorder=2)
    if role!="original":
        base.boundary.plot(ax=ax,color=REF,linewidth=.30,linestyle=":",alpha=REFERENCE_ALPHA,zorder=3)
    if labels:
        for iso in ["CHN","USA","RUS","AUS","CAN","BRA"]:
            q=current[current.iso3.eq(iso)&current.has_data]
            if q.empty:
                continue
            r=q.iloc[0]
            parts=list(r.geometry.geoms) if r.geometry.geom_type=="MultiPolygon" else [r.geometry]
            p=max(parts,key=lambda g:g.area).representative_point()
            if role=="original":
                value=f"{r.distance_km/1000:.1f}k"
            elif mode=="original_relative":
                value=f"{r.change_pct:+.1f}%"
            else:
                value=f"{max(0,-r.change_pct):.1f}%"
            t=ax.text(p.x,p.y,iso+"\n"+value,ha="center",va="center",fontsize=6.4,color="#344951",zorder=6)
            t.set_path_effects([pe.withStroke(linewidth=2,foreground="white",alpha=.9)])
    draw_gates(ax,gates,case,role)
    ax.set(xlim=bounds[:2],ylim=bounds[2:],aspect="equal",xticks=[],yticks=[],xlabel="",ylabel="")
    ax.set_axis_off()


def save(fig,name):
    for ext in ["png","pdf","svg"]:
        fig.savefig(FIG/(name+"."+ext),dpi=250,facecolor="white")
    plt.close(fig)
    print("Saved",name,flush=True)


def legend(fig,y):
    fig.legend(handles=[Patch(facecolor=NO_DATA,edgecolor=to_rgba(EDGE,BORDER_ALPHA),label="No comparable OD data"),
        Patch(facecolor="#fffefa",edgecolor=to_rgba(EDGE,BORDER_ALPHA),label="Zero distance change / saving"),
        Line2D([],[],color=to_rgba(EDGE,BORDER_ALPHA),lw=.8,label="Country borders (80% transparent)"),
        Line2D([],[],color=to_rgba(REF,REFERENCE_ALPHA),ls=":",lw=.8,label="Geographical reference outline"),
        Line2D([],[],marker="o",color=OLD,lw=0,markersize=6,label="Already closed in parent"),
        Line2D([],[],marker="D",color=NEW,lw=0,markersize=6,label="Newly closed in child")],
        loc="lower center",bbox_to_anchor=(.5,y),ncol=3,frameon=False,fontsize=8.1)


def colourbars(fig,mode,y,height):
    for key,rect in [("original",[.055,y,.235,height]),(mode,[.405,y,.54,height])]:
        spec=colour_scale_definitions()[key]
        cmap,norm=COLOUR_SCALES[key]
        ax=fig.add_axes(rect)
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ax,orientation="horizontal")
        cb.set_ticks(spec["ticks"],labels=spec["ticklabels"])
        cb.minorticks_off()
        cb.ax.tick_params(labelsize=8,length=3,pad=2)
        cb.set_label(spec["label"],fontsize=8.5,labelpad=3)
        cb.outline.set_linewidth(.5)


def atlas(cases,frames,bounds,mode):
    fig=plt.figure(figsize=(19,14.2))
    titles=["Original network","Before the added closure","After the added closure"]
    for j,title in enumerate(titles):
        fig.text(.027+.33*j,.974,title,fontsize=12,weight="bold")
    for i,ship in enumerate(SHIPS):
        row=cases[ship]
        top=.938-i*.201
        n,m=int(row["parent_count"]),int(row["child_count"])
        fig.text(.025,top,f"{NAMES[ship]} | {calendar.month_abbr[int(row['month'])]} | {n} to {m} closures | added {row['added_gate']}",fontsize=10.4,weight="bold")
        if mode=="original_relative":
            text=f"Weighted distance vs original: {row['before_vs_original_pct']:+.2f}% to {row['after_vs_original_pct']:+.2f}%"
        else:
            text=f"Saving vs same-closure detour: {row['before_saving_pct']:.2f}% to {row['after_saving_pct']:.2f}%"
        fig.text(.025,top-.019,text+f"; Arctic trade share {row['B_before_pct']:.2f}% to {row['B_after_pct']:.2f}%",fontsize=9,color="#4c6069")
        for j,(step,role) in enumerate([(0,"original"),(n,"before"),(m,"after")]):
            ax=fig.add_axes([.015+.33*j,top-.197,.31,.168])
            frame,gates,_=frames[(mode,ship,step)]
            panel(ax,frames[(mode,ship,0)][0],frame,gates,bounds,row,role,mode)
            ax.text(.002,.995,chr(97+i*3+j),transform=ax.transAxes,va="top",weight="bold",fontsize=11)
    colourbars(fig,mode,.103,.012)
    legend(fig,.033)
    if mode=="original_relative":
        definition="Middle/right: area ratio = embedded-network distance / original-network distance; labels show signed distance changes. Left labels: original 1,000 km."
    else:
        definition="Middle/right: area ratio = Arctic-option distance / SAME-closure detour distance; labels show distance savings. Left labels: original 1,000 km."
    fig.text(.5,.023,definition,ha="center",fontsize=8.2,color="#50656e")
    fig.text(.5,.010,"Colour limits are shared across all vessels and before/after; log scaling changes fill only, not map areas. Grey indicates missing data, not zero.",ha="center",fontsize=8.2,color="#50656e")
    fig.text(.5,.001,"All 1-to-2 through 4-to-5 stages compared; shown distance jumps are conditional on stage-specific maximum-Arctic-flow representatives.",ha="center",fontsize=8.1,color="#50656e")
    save(fig,"Multihue_Log10_"+("OriginalRelative" if mode=="original_relative" else "DetourBenefit")+"_MaxJump_Atlas")


def centered(shape):
    parts=list(shape.geoms) if shape.geom_type=="MultiPolygon" else [shape]
    p=max(parts,key=lambda p:p.area)
    center=np.array([p.centroid.x,p.centroid.y])
    return shapely.transform(p,lambda xy:xy-center)


def individual(case,frames,bounds,mode):
    ship=case["shiptype"]
    n,m=int(case["parent_count"]),int(case["child_count"])
    fig=plt.figure(figsize=(18,9.3))
    for j,(step,role) in enumerate([(0,"original"),(n,"before"),(m,"after")]):
        ax=fig.add_axes([.015+.33*j,.535,.31,.34])
        f,g,_=frames[(mode,ship,step)]
        panel(ax,frames[(mode,ship,0)][0],f,g,bounds,case,role,mode)
        ax.text(.002,1.04,chr(97+j)+"  "+["Original network",f"{n} closures",f"{m} closures"][j],transform=ax.transAxes,fontsize=11,weight="bold")
    country=pd.read_csv(HERE/case["folder"]/"country_distance_jumps.csv")
    candidate=country[country.matched_od_count.ge(3)].copy()
    field="original_relative_jump_pp" if mode=="original_relative" else "saving_jump_pp"
    candidate["rank"]=candidate[field].abs()*candidate.matched_trade_usd
    candidate=candidate.sort_values("rank",ascending=False).head(6)
    for i,(_,r) in enumerate(candidate.iterrows()):
        iso=r.origin_iso3
        ax=fig.add_axes([.03+i*.16,.25,.14,.17])
        shapes=[centered(frames[(mode,ship,step)][0].loc[lambda d:d.iso3.eq(iso),"geometry"].iloc[0]) for step in [0,n,m]]
        radius=np.max(np.abs(np.array([p.bounds for p in shapes])))*1.08
        for k,(shape,ls,width) in enumerate(zip(shapes,[":","--","-"],[.8,1.1,1.5])):
            role=["original","before","after"][k]
            step=[0,n,m][k]
            f=frames[(mode,ship,step)][0].loc[lambda d:d.iso3.eq(iso)]
            shifted=shapely.transform(shape,lambda xy:xy+np.array([(k-1)*2.2*radius,0]))
            gpd.GeoSeries([shifted]).plot(ax=ax,facecolor=face_colours(f,role,mode),
                edgecolor=to_rgba(EDGE,BORDER_ALPHA),linestyle=ls,linewidth=width)
            ax.text((k-1)*2.2*radius,-1.20*radius,["Original","Before","After"][k],ha="center",fontsize=6.7,color=EDGE)
        ax.set(xlim=(-3.3*radius,3.3*radius),ylim=(-1.4*radius,1.05*radius),aspect="equal",xticks=[],yticks=[],xlabel="",ylabel="")
        ax.set_axis_off()
        ax.text(.5,1.06,iso,transform=ax.transAxes,ha="center",weight="bold",fontsize=10)
        if mode=="original_relative":
            lines=[f"{r.baseline_km/1000:.1f} / {r.before_arctic_km/1000:.1f} / {r.after_arctic_km/1000:.1f}k km",
                f"{r.before_vs_original_pct:+.1f}% to {r.after_vs_original_pct:+.1f}% vs original"]
        else:
            lines=[f"{n} gates: {r.before_detour_km/1000:.1f} to {r.before_arctic_km/1000:.1f}k ({r.before_saving_pct:.1f}%)",
                f"{m} gates: {r.after_detour_km/1000:.1f} to {r.after_arctic_km/1000:.1f}k ({r.after_saving_pct:.1f}%)"]
        for k,line in enumerate(lines):
            ax.text(.5,-.22-.20*k,line,transform=ax.transAxes,ha="center",fontsize=7.7,color=EDGE)
    fig.text(.025,.973,NAMES[ship]+f" | {n} to {m} closures | "+calendar.month_name[int(case["month"])],fontsize=14,weight="bold")
    fig.text(.025,.932,"Previously closed: "+case["before"],fontsize=10)
    fig.text(.025,.901,"Added closure: "+case["added_gate"]+f"; additional Arctic trade US${case['delta_arctic_trade_usd']/1e9:.2f} bn",fontsize=10)
    colourbars(fig,mode,.155,.016)
    legend(fig,.034)
    reference="original open conventional network" if mode=="original_relative" else "conventional detour under identical closures"
    fig.text(.5,.022,"Country-area reference: "+reference+". Colour scales are shared across all vessels and scenarios.",ha="center",fontsize=8.4,color="#50656e")
    fig.text(.5,.008,"Lower silhouettes: original / before / after at a fixed scale within each country; same quantitative fills as the maps. Log colour does not amplify deformation.",ha="center",fontsize=8.1,color="#50656e")
    save(fig,f"Multihue_Log10_{mode}_{ship}_Detail")


def stage_chart(stages):
    fig,axs=plt.subplots(2,2,figsize=(13,8.8))
    fig.subplots_adjust(left=.09,right=.97,top=.92,bottom=.15,hspace=.44,wspace=.22)
    for ax,ship in zip(axs.ravel(),SHIPS):
        g=stages[stages.shiptype.eq(ship)].sort_values("child_count")
        x=np.arange(4)
        ax.bar(x-.18,g.saving_jump_pp,width=.33,color="#697e89",label="Additional saving vs same-closure detour")
        ax.bar(x+.18,g.original_relative_jump_pp,width=.33,color="#b8c3c8",label="Embedded-distance change / original distance")
        for j,(_,r) in enumerate(g.iterrows()):
            for offset,col in [(-.18,"saving_jump_pp"),(.18,"original_relative_jump_pp")]:
                v=r[col]
                ax.text(j+offset,v+(.25 if v>=0 else -.35),f"{v:.2f}",ha="center",va="bottom" if v>=0 else "top",fontsize=8)
        ax.set_xticks(x,["1 to 2","2 to 3","3 to 4","4 to 5"])
        ax.set_title(NAMES[ship],loc="left",fontsize=11,weight="bold")
        ax.set_ylabel("Jump (percentage points)",fontsize=9)
        ax.axhline(0,color="#596e78",lw=.6)
        ax.spines[["right","top"]].set_visible(False)
        ax.set_ylim(min(-1.,g.original_relative_jump_pp.min()-1),max(g.saving_jump_pp.max(),g.original_relative_jump_pp.max())+2.5)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.065),frameon=False,ncol=2,fontsize=9)
    fig.text(.5,.025,"Each stage uses its maximum-Arctic-trade-gain representative from the exhaustive search. The four bars are different scenarios, not one nested closure trajectory.",ha="center",fontsize=8.4,color="#50656e")
    save(fig,"SI_All_ClosureStages_DistanceJumps")


def main():
    global COLOUR_SCALES
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none"})
    world=cart.base_world()
    COLOUR_SCALES=build_colour_scales()
    cases={}
    for mode,name in [("detour_benefit","selected_maximum_jump_cases.csv"),("original_relative","selected_maximum_original_change_cases.csv")]:
        cases[mode]={r["shiptype"]:r for r in pd.read_csv(RESULTS/name).to_dict("records")}
    frames={}
    for mode,ships in cases.items():
        for ship,case in ships.items():
            for step in [0,int(case["parent_count"]),int(case["child_count"])]:
                frames[(mode,ship,step)]=distort(world,case,step,mode)
    bounds=cart.shared_bounds([v[0] for v in frames.values()])
    colour_rows=[]
    for (mode,ship,step),(frame,_,_) in frames.items():
        case=cases[mode][ship]
        role="original" if step==0 else "before" if step==int(case["parent_count"]) else "after"
        values=colour_values(frame,role,mode)
        key=colour_key(role,mode)
        spec=colour_scale_definitions()[key]
        finite=values[np.isfinite(values)]
        assert np.all((finite>=spec["vmin"]-1e-9)&(finite<=spec["vmax"]+1e-9)), (mode,ship,step)
        table=frame[["iso3","distance_km","reference_km","has_data","matched_trade_usd","matched_od_count"]].copy()
        table["mode"],table["shiptype"],table["step"],table["role"]=mode,ship,step,role
        table["colour_metric"],table["colour_value"],table["colour_hex"]=key,values,[to_hex(c) for c in face_colours(frame,role,mode)]
        table["log10_colour_coordinate"]=np.log10(values) if role=="original" else signed_log10(values)
        colour_rows.append(table)
    pd.concat(colour_rows,ignore_index=True).to_csv(RESULTS/"country_log10_colour_values.csv",index=False)
    (RESULTS/"log10_colour_scales.json").write_text(json.dumps(dict(
        classification="Continuous numeric mapping, not country identity or quantile classes",
        common_before_after_and_all_vessels=True,no_data_colour=NO_DATA,
        percentage_point_offset=1.,geometry_scaling="Unchanged linear distance ratio; no log geometry scaling",
        definitions=colour_scale_definitions()),indent=2))
    for mode in cases:
        atlas(cases[mode],frames,bounds,mode)
        for case in cases[mode].values():
            individual(case,frames,bounds,mode)
    stage_chart(pd.read_csv(RESULTS/"all_vessel_stage_comparisons.csv"))
    (RESULTS/"render_audit.json").write_text(json.dumps(dict(status="rendered",common_bounds=bounds,
        country_fill="quantitative_log10_distance",country_edge=EDGE,country_edge_alpha=BORDER_ALPHA,
        reference_outline_alpha=REFERENCE_ALPHA,no_data_colour=NO_DATA,colour_scales=colour_scale_definitions(),
        common_before_after_and_all_vessels=True,geometry_scaling="unchanged_linear_distance_ratio",
        existing_gate_color=OLD,new_gate_color=NEW,
        cases=cases,maps=[v[2] for v in frames.values()]),indent=2))


if __name__=="__main__":
    main()
