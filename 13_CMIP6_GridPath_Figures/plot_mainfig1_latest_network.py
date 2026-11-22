"""Three-row Main Figure 1 and separately recalculated appendix panels."""
from pathlib import Path
import argparse
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Rectangle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
import shapely

from rebuild_mainfig1_latest_network import ROOT, NET, OUT, SHIPS, ROUTES, sha
from plot_port_network_raster import COLORS, LABELS, raster, contains_any, draw_panel, annotate
from repair_cmip6_monthly_routes import repaired_route_metadata
from prepare_mainfig1_density_region_sankey import REGIONS, NONE

OLD = ROOT/"figures/Port_network_1390_raster"
PUB = ROOT/"data/public/network_map_context"
DISPLAY = {"container": "Container", "bulk": "Dry bulk", "tanker": "Tanker", "lng": "LNG"}
FIG = OUT/"figures"
DEEP_CMAP=LinearSegmentedColormap.from_list("deep_water_support",["#edf5fc","#b6d6ed","#6da6d1","#2368a0","#123b66"])
REGION_COLORS={"Asia":"#0077BB","Europe":"#009988","North America":"#EE7733",
               "South America":"#AA4499","Africa":"#CC6677","Oceania":"#8D9A3D"}
INSETS = [
    ("I", "Great Lakes - St Lawrence", [-94,40,-64,51], .025),
    ("II", "Baltic - Saimaa - Onega", [10,55,45,64], .025),
    ("III", "Caspian - Volga - Don", [25,36,65,51], .025),
]
COUNTRIES = {"CHN":"China", "USA":"United States", "GBR":"United Kingdom", "FRA":"France", "RUS":"Russia",
    "JPN":"Japan", "KOR":"South Korea", "SGP":"Singapore", "NLD":"Netherlands", "BEL":"Belgium",
    "DEU":"Germany", "ESP":"Spain", "ITA":"Italy", "AUS":"Australia", "BRA":"Brazil", "CAN":"Canada",
    "MYS":"Malaysia", "IDN":"Indonesia", "IND":"India", "VNM":"Viet Nam", "THA":"Thailand",
    "NOR":"Norway", "TUR":"Turkiye", "SAU":"Saudi Arabia", "ARE":"United Arab Emirates", "EGY":"Egypt",
    "MEX":"Mexico", "ZAF":"South Africa", "GRC":"Greece", "TTO":"Trinidad and Tobago",
    "MAR":"Morocco", "NZL":"New Zealand", "LKA":"Sri Lanka", "PRT":"Portugal"}


def samples_and_context():
    meta = json.loads((OLD/"sample_cache_manifest.json").read_text())
    assert meta["graph_sha256"] == sha(NET/"physical_edges.parquet")
    assert meta["geometry_sha256"] == sha(NET/"edge_geometry.npz")
    graph = pd.read_parquet(OLD/"classified_route_samples.parquet")
    cache = OUT/"selected_AIS_classified_map_points.parquet"
    land = gpd.read_file(ROOT/"data/public/naturalearth_10m/ne_10m_land.shp")
    lakes = gpd.read_file(PUB/"ne_10m_lakes.shp")
    if cache.exists():
        ais = pd.read_parquet(cache)
    else:
        ais = pd.read_parquet(OUT/"selected_AIS_map_points.parquet")
        point = shapely.points(ais[["lon", "lat"]].to_numpy())
        inland = contains_any(land.geometry.to_numpy(), point)
        classes = np.where(inland, 7, 2).astype(np.uint8)
        deep = gpd.read_file(PUB/"ne_10m_bathymetry_K_200.shp")
        classes[contains_any(deep.geometry.to_numpy(), point) & ~inland] = 1
        lake = gpd.read_file(ROOT/"data/public/gshhg/GSHHS_shp/f/GSHHS_f_L2.shp")
        classes[contains_any(lake.geometry.to_numpy(), point)] = 4
        river = gpd.read_file(PUB/"ne_10m_rivers_lake_centerlines.shp")
        classes[contains_any(shapely.buffer(river.geometry.to_numpy(), .02), point) & (classes == 7)] = 3
        for f in (ROOT/"data/public/inland_navigation").glob("*.parquet"):
            d = pd.read_parquet(f, columns=["class", "geometry"])
            geom = shapely.make_valid(shapely.from_wkb(d.geometry.to_numpy()))
            for name, code in [("lake",4),("reservoir",4),("river",3),("canal",5)]:
                arr = geom[d["class"].eq(name).to_numpy()]
                line = shapely.get_type_id(arr) == 1
                arr[line] = shapely.buffer(arr[line], .00002)
                classes[contains_any(arr, point)] = code
        ais["class_code"] = classes
        ais.to_parquet(cache, index=False)
    samples = pd.concat([graph[["lon","lat","class_code"]], ais], ignore_index=True)
    return samples, land, lakes


def save_map_grid(name, display, presence, bounds, step, output_dir=None):
    folder=Path(output_dir) if output_dir is not None else OUT/"maps"
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/(name+".npz"),display_class=display,class_presence=presence,
        bounds=np.asarray(bounds),resolution_degrees=step,class_names=np.array(LABELS[1:]),
        crs="EPSG:4326",row_order="south_to_north")
    with (folder/(name+".asc")).open("w") as stream:
        stream.write(f"ncols {display.shape[1]}\nnrows {display.shape[0]}\nxllcorner {bounds[0]}\nyllcorner {bounds[1]}\ncellsize {step}\nNODATA_value 0\n")
        np.savetxt(stream,display[::-1],fmt="%d")
    (folder/(name+".prj")).write_text('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')


def deep_water_layer(ax,display,bounds,step,name,support_path=None,output_dir=None,colour_max=None):
    source=Path(support_path) if support_path is not None else OUT/"panel_a_file_month_support_1deg.npz"
    support=np.load(source)["S_g"]
    xx=bounds[0]+(np.arange(display.shape[1])+.5)*step
    yy=bounds[1]+(np.arange(display.shape[0])+.5)*step
    ix=np.floor(np.mod(xx+180,360)).astype(int)
    iy=np.floor(np.clip(yy,-89.999999,89.999999)+90).astype(int)
    local=support[iy[:,None],ix[None,:]]
    values=np.log10(1+local)
    vmax=float(np.ceil(np.log10(1+support.max())*2)/2) if colour_max is None else float(colour_max)
    norm=Normalize(0,vmax)
    rgba=np.asarray([to_rgba(c) for c in COLORS])[display]
    original=rgba.copy();deep=display==1
    rgba[deep]=DEEP_CMAP(norm(values[deep]))
    np.testing.assert_array_equal(rgba[~deep],original[~deep])
    ax.images[-1].set_data(rgba)
    folder=Path(output_dir) if output_dir is not None else OUT/"maps"
    np.savez_compressed(folder/(name+"_deep_support.npz"),
        D_g=np.where(deep,values,np.nan),S_g=np.where(deep,local,-1),deep_water_mask=deep,
        statistical_resolution_degrees=1,display_resolution_degrees=step,bounds=np.asarray(bounds),
        row_order="south_to_north",crs="EPSG:4326")
    (folder/(name+"_colour_audit.json")).write_text(json.dumps(dict(
        non_deep_RGBA_unchanged=True,statistical_resolution_degrees=1,
        deep_cells=int(deep.sum()),deep_cells_without_recorded_AIS_support=int((deep&(local==0)).sum()),
        non_deep_palette={LABELS[i]:COLORS[i] for i in range(2,8)},colour_min=0,colour_max=norm.vmax),indent=2))
    return ScalarMappable(norm=norm,cmap=DEEP_CMAP)


def map_a(fig, spec, context, standalone=False,support_path=None,output_dir=None,colour_max=None,map_note=None):
    samples, land, lakes = context
    ports = pd.read_csv(NET/"canonical_port_registry.csv")
    locks = pd.read_csv(OLD/"documented_lock_reaches.csv")
    registry = pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
    sub = spec.subgridspec(4, 3, height_ratios=[2.8, 1.02, .20, .25], hspace=.16, wspace=.15)
    ax = fig.add_subplot(sub[0,:])
    ax.set_gid("main_a_world")
    display, presence = raster(samples, [-180,-90,180,90], .25)
    save_map_grid("Global_AISfirst_025deg",display,presence,[-180,-90,180,90],.25,output_dir)
    draw_panel(ax, [-180,-90,180,90], .25, display, land, lakes, ports, locks, "a", True)
    sm=deep_water_layer(ax,display,[-180,-90,180,90],.25,"Global_AISfirst_025deg",support_path,output_dir,colour_max)
    ax.set_aspect(1)
    for text in ax.texts:
        if text.get_text()=="a":
            text.set_position((-.055,1.0)); text.set_va("center")
    ax.set_xlabel(""); ax.set_ylabel("")
    note=map_note or "1,390 economic records  |  1,386 unique port entries"
    ax.text(.012,.03,note, transform=ax.transAxes,
            fontsize=10, bbox=dict(fc="white",ec="none",alpha=.9), zorder=10)
    ax.scatter(registry.lon, registry.lat, marker="^", color="#111111", s=20,
               edgecolors="white", linewidth=.45, zorder=9)
    offsets = {1:(7,-13), 2:(-16,15), 3:(6,7), 4:(6,-12), 5:(-32,5), 6:(-18,-14),
        7:(5,-15),8:(23,-18),9:(13,8),10:(8,-12),11:(10,6),12:(-20,2),13:(-16,-15),
        14:(35,-8),15:(1,16),16:(-24,-13),17:(10,-8),18:(-15,-12),19:(-9,-19),
        20:(-25,-4),21:(5,-12),22:(12,10),23:(0,14),24:(-20,9)}
    for r in registry.itertuples():
        ax.annotate(str(r.number),(r.lon,r.lat),xytext=offsets[r.number],textcoords="offset points",
                    fontsize=8.5,weight="bold",ha="center",va="center",zorder=10,
                    bbox=dict(fc="white",ec="none",alpha=.9,pad=.5),
                    arrowprops=dict(arrowstyle="-",lw=.45,color="#333333"))
    points, _ = repaired_route_metadata()
    for route, tx in zip(ROUTES, [-118,37,125,-15]):
        p = points[(points.route==route)&(points.month==9)].sort_values("order")
        if p.empty:
            continue
        q = p[["lon","lat"]].to_numpy()
        jumps = np.r_[False, np.abs(np.diff(q[:,0]))>180]
        xx=q[:,0].copy(); yy=q[:,1].copy(); xx[jumps]=np.nan; yy[jumps]=np.nan
        ax.plot(xx,yy,color="#625078",lw=1.05,ls="--",zorder=7)
        center=q[len(q)//2]
        ax.annotate(route,center,xytext=(tx,84),textcoords="data",ha="center",fontsize=9,
                    color="#58436b",zorder=10,bbox=dict(fc="white",ec="none",alpha=.85,pad=.5),
                    arrowprops=dict(arrowstyle="-",color="#625078",lw=.6))
    for i, (_, name, bounds, step) in enumerate(INSETS):
        roman=["I","II","III"][i]
        ax.add_patch(Rectangle((bounds[0],bounds[1]),bounds[2]-bounds[0],bounds[3]-bounds[1],
                              fill=False,ec="#333333",lw=.8,ls=":",zorder=8))
        ax.text(bounds[0],bounds[1]-5,roman,fontsize=9,weight="bold",zorder=10,
                bbox=dict(fc="white",ec="none",pad=.3))
        local = fig.add_subplot(sub[1,i])
        grid,presence=raster(samples,bounds,step)
        save_map_grid("Inset_"+roman+"_0025deg",grid,presence,bounds,step,output_dir)
        draw_panel(local,bounds,step,grid,land,lakes,ports,locks,"",False)
        deep_water_layer(local,grid,bounds,step,"Inset_"+roman+"_0025deg",support_path,output_dir,colour_max)
        local.set_aspect(1/np.cos(np.radians((bounds[1]+bounds[3])/2)))
        local.set_xlabel(""); local.set_ylabel("")
        local.text(.015,.96,roman+"  "+name,transform=local.transAxes,va="top",fontsize=9,
                   bbox=dict(fc="white",ec="none",alpha=.9),zorder=10)
        if i==0:
            annotate(local,"Welland",(-79.21,43.1),(-46,-22))
            annotate(local,"St Lawrence locks",(-74.2,45.15),(-10,-22))
        elif i==1:
            annotate(local,"Saimaa locks",(28.5,61),(-74,-35))
            annotate(local,"Sodertalje",(17.63,59.19),(-15,-35))
        else:
            annotate(local,"Volga-Don locks",(44.5,48.5),(-42,-25))
            annotate(local,"Caspian Sea",(50.8,41),(-38,-12))
    colorax=fig.add_subplot(sub[2,:]);colorax.axis("off")
    barax=colorax.inset_axes([.20,.55,.60,.40])
    cb=fig.colorbar(sm,cax=barax,orientation="horizontal")
    cb.ax.invert_xaxis()
    cb.set_ticks(np.arange(0,sm.norm.vmax+.01,.5))
    cb.ax.tick_params(labelsize=8,length=2)
    cb.set_label("Deep-water sea (>200 m)",fontsize=8.5,labelpad=3)
    legax = fig.add_subplot(sub[3,:]); legax.axis("off")
    handles=[Patch(fc=COLORS[i],label=LABELS[i]) for i in range(2,8)]
    handles += [Line2D([],[],ls="",marker="o",color="#293640",markersize=3,label="Port entry"),
                Line2D([],[],ls="",marker="^",color="black",markersize=4,label="Chokepoint 1-24"),
                Line2D([],[],ls="--",color="#625078",label="September Arctic core")]
    legax.legend(handles=handles,ncol=5,loc="center",frameon=False,fontsize=8.7,columnspacing=1.6)


def heat_b(fig,spec):
    d=pd.read_csv(OUT/"panel_b_exposure.csv")
    ax=fig.add_subplot(spec)
    ax.set_gid("main_b_heat")
    matrix=d.pivot(index="shiptype",columns="number",values="exposure_pct").reindex(SHIPS).to_numpy()
    image=ax.imshow(matrix,aspect="auto",cmap="YlGnBu",vmin=0,vmax=max(30,np.ceil(matrix.max()/10)*10))
    names=d.drop_duplicates("number").sort_values("number")
    labels=[f"{r.number:02d}  {r.name.replace(' Strait','').replace('Strait of ','').replace(' Channel','')}" for r in names.itertuples()]
    ax.set_xticks(range(24),labels,rotation=48,ha="right",rotation_mode="anchor",fontsize=8)
    ax.set_yticks(range(4),[DISPLAY[s] for s in SHIPS],fontsize=10)
    ax.set_xticks(np.arange(-.5,24,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which="minor",color="white",lw=.6);ax.tick_params(which="both",length=0)
    for y in range(4):
        for x in range(24):
            value=matrix[y,x]
            ax.text(x,y,"0" if value==0 else ("<0.1" if value<.05 else f"{value:.1f}"),ha="center",va="center",fontsize=7.7,
                    color="white" if value>image.norm.vmax*.54 else "#152e3a")
    ax.text(-.055,1.11,"b",transform=ax.transAxes,weight="bold",fontsize=16)
    # A dedicated outside colourbar does not steal width from this row.
    cax=ax.inset_axes([1.018,0,.014,1])
    cb=fig.colorbar(image,cax=cax)
    cb.set_label("Activity exposure (%)",fontsize=9);cb.ax.tick_params(labelsize=8)
    ax.text(0,1.08,"Each cell uses the vessel-type annual activity denominator; one route may encounter multiple gates.",
            transform=ax.transAxes,fontsize=9,color="#53616a")


def ribbon(ax,x0,x1,y0,y1,height,color,alpha=.65):
    c=(x1-x0)*.43
    vertices=[(x0,y0),(x0+c,y0),(x1-c,y1),(x1,y1),(x1,y1+height),
              (x1-c,y1+height),(x0+c,y0+height),(x0,y0+height),(x0,y0)]
    codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,
           MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(vertices,codes),facecolor=color,alpha=alpha,edgecolor="none"))


def sankey_c(fig,spec):
    ax=fig.add_subplot(spec);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    ax.set_gid("main_c_sankey")
    flow=pd.read_csv(OUT/"panel_c_region_gate_flows.csv")
    gates=pd.read_csv(OUT/"panel_c_gate_shares.csv").sort_values(["allocated_weight","number"],ascending=[False,True])
    total=flow.allocated_weight.sum();scale=.36/total
    def positions(weights,gap,top):
        y=top;res={}
        for name,weight in weights.items():
            h=float(weight)*scale;y-=h;res[name]=[y,h,float(weight)];y-=gap
        return res
    left=positions(flow.groupby("origin").allocated_weight.sum().reindex(REGIONS,fill_value=0),.065,.90)
    right=positions(flow.groupby("dest").allocated_weight.sum().reindex(REGIONS,fill_value=0),.065,.90)
    middle=positions(gates.set_index("gate").allocated_weight,.020,.945)
    lc={k:y+h for k,(y,h,w) in left.items()};rc={k:y+h for k,(y,h,w) in right.items()}
    for gate,(gy,gh,gw) in middle.items():
        cursor=gy+gh
        for origin in REGIONS:
            subflow=flow[(flow.gate==gate)&(flow.origin==origin)].set_index("dest").reindex(REGIONS)
            weight=float(subflow.allocated_weight.fillna(0).sum())
            if weight==0:
                continue
            h=weight*scale;lc[origin]-=h;cursor-=h
            ribbon(ax,.145,.385,lc[origin],cursor,h,REGION_COLORS[origin],.55)
            # A single logical middle node has two aligned edges around a label gutter.
            ax.add_patch(Rectangle((.385,cursor),.23,h,fc=REGION_COLORS[origin],alpha=.10,ec="none"))
            for x in [.385,.605]:
                ax.add_patch(Rectangle((x,cursor),.010,h,fc=REGION_COLORS[origin],ec="none"))
            next_y=cursor+h
            for dest in REGIONS:
                w=subflow.loc[dest,"allocated_weight"]
                if not np.isfinite(w) or w<=0:
                    continue
                height=float(w)*scale;next_y-=height;rc[dest]-=height
                ribbon(ax,.615,.865,next_y,rc[dest],height,REGION_COLORS[origin],.55)
        assert np.isclose(cursor,gy,atol=1e-10)
    for x,positions_,leftside in [(.135,left,True),(.865,right,False)]:
        for region,(y,h,weight) in positions_.items():
            start=0 if leftside else .865
            ax.add_patch(Rectangle((start,y),.135,h,fc=REGION_COLORS[region],alpha=.10,ec="none"))
            ax.add_patch(Rectangle((0 if leftside else .990,y),.010,h,fc=REGION_COLORS[region],ec="none"))
            ax.add_patch(Rectangle((x,y),.010,h,fc=REGION_COLORS[region],ec="none"))
            ax.text(.068 if leftside else .932,y+h/2,f"{region}\n{100*weight/total:.1f}%",
                ha="center",va="center",fontsize=9.2,bbox=dict(fc="white",ec="none",alpha=.92,pad=1),zorder=8)
    for row in gates.itertuples():
        y,h,_=middle[row.gate]
        name=row.name
        label=(f"{int(row.number):02d}  " if row.number else "")+name
        pct="<0.01%" if 0<row.percent<.005 else f"{row.percent:.2f}%"
        ax.text(.50,y+h/2,label+"  "+pct,ha="center",va="center",fontsize=7.8,
            bbox=dict(fc="white",ec="none",alpha=.94,pad=.8),zorder=7)
    ax.text(.068,.986,"Shipping origin\nregion",ha="center",fontsize=10.5,weight="bold")
    ax.text(.50,.986,"Listed chokepoint / none",ha="center",fontsize=10.5,weight="bold")
    ax.text(.932,.986,"Shipping destination\nregion",ha="center",fontsize=10.5,weight="bold")
    ax.text(-.055,1.005,"c",transform=ax.transAxes,weight="bold",fontsize=16)
    audit=json.loads((OUT/"panel_c_region_sankey_audit.json").read_text())
    ax.text(0,.035,f"Estimated sea-linked value: US${audit['included_trade_weight_usd']/1e12:.2f} trillion | {audit['included_country_pairs']:,} country pairs | {audit['included_intercontinental_rows']:,} OD/type/month records.\n2024 BACI x fixed 2021 sea shares; {audit['modal_imputation']['included_imputed_pairs']} pairs median-filled ({audit['modal_imputation']['imputed_proxy_value_pct']:.1f}% of weight). Multi-gate weights split equally.",
        fontsize=8.5,color="#53616a")


def main_figure(context):
    left,right,top,bottom=.080,.930,.980,.033
    width=19
    # Reserve exactly a 2:1 box for the world map, without stretching geography.
    map_height=width*(right-left)/2*(2.8+1.02+.20+.25)/2.8*(1+.16*3/4)
    heights=[map_height,.45,2.6,2.05,9.2]
    fig=plt.figure(figsize=(width,sum(heights)/(top-bottom)))
    gs=fig.add_gridspec(5,1,height_ratios=heights,hspace=0,left=left,right=right,top=top,bottom=bottom)
    map_a(fig,gs[0],context);heat_b(fig,gs[2]);sankey_c(fig,gs[4])
    fig.text(left,.013,"Panel c uses estimated trade weights with 2021 modal shares, not observed 2024 cargo values. Shipping endpoints are not final producers or consumers.",
             fontsize=9,color="#59666d")
    save(fig,"MainFig1_AISfirst_24Chokepoints")
    fig=plt.figure(figsize=(19,15.5))
    gs=fig.add_gridspec(1,1,left=.052,right=.986,top=.975,bottom=.21)
    map_a(fig,gs[0],context,True)
    registry=pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
    for k,row in enumerate(registry.itertuples()):
        col=k//6; y=.172-(k%6)*.023
        fig.text(.063+col*.235,y,f"{row.number:02d}  {row.name}",fontsize=8.5)
    fig.text(.063,.014,"Deep-water colours: 1-degree log10(1 + trajectory-file/month support). Other waterway categories retain their original colours.\nThe 0.25-degree global and 0.025-degree inset masks preserve channel shapes; support has not been statistically downscaled.",fontsize=8.5)
    save(fig,"MainFig1a_GlobalPorts_ThreeInsets")
    fig=plt.figure(figsize=(19,11.5))
    gs=fig.add_gridspec(1,1,left=.065,right=.968,top=.95,bottom=.08)
    sankey_c(fig,gs[0])
    fig.text(.065,.025,"Each annual sea-linked estimate is allocated once across matched OD/type/month records. Fixed 2021 shares and median imputations are not 2024 cargo observations.",fontsize=8.5)
    save(fig,"MainFig1c_Region_Chokepoint_Region")


def appendix():
    d=pd.read_csv(OUT/"appendix_country_exposure.csv")
    all_den=d.groupby(["country","side"]).denominator.first().groupby("country").sum().sort_values(ascending=False)
    top=[x for x in all_den.index if x not in ["TWN","UNK"]][:20]
    countries=top+[c for c in ["CHN","USA","GBR","FRA","RUS"] if c not in top]
    # Incident activity includes both endpoints; no world-total summation is claimed.
    values=d.groupby(["country","number"]).agg(exposed=("exposed_weight","sum"),denominator=("denominator","sum"))
    values["pct"]=100*values.exposed/values.denominator
    matrix=values.pct.unstack().reindex(countries).to_numpy()
    fig=plt.figure(figsize=(19,15))
    gs=fig.add_gridspec(2,2,height_ratios=[2.35,1],left=.12,right=.96,top=.962,bottom=.07,hspace=.42,wspace=.20)
    ax=fig.add_subplot(gs[0,:])
    cmap=plt.get_cmap("YlGnBu").copy();cmap.set_bad("#b6bec4")
    image=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",cmap=cmap,vmin=0,vmax=100)
    registry=pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
    ax.set_xticks(range(24),[f"{r.number:02d} {r.name.replace(' Strait','').replace('Strait of ','')}" for r in registry.itertuples()],rotation=48,ha="right",fontsize=8)
    labels=[COUNTRIES.get(c,c)+( " *" if c in ["CHN","USA","GBR","FRA","RUS"] else "") for c in countries]
    ax.set_yticks(range(len(countries)),labels,fontsize=9)
    for y in range(len(countries)):
        for x in range(24):
            v=matrix[y,x]
            label="NA" if not np.isfinite(v) else ("0" if v==0 else ("<1" if v<1 else f"{v:.0f}"))
            ax.text(x,y,label,ha="center",va="center",fontsize=6.7,color="white" if v>55 else "#243846")
    ax.set_xticks(np.arange(-.5,24,1),minor=True);ax.set_yticks(np.arange(-.5,len(countries),1),minor=True)
    ax.grid(which="minor",lw=.4,color="white");ax.tick_params(which="both",length=0)
    ax.text(-.12,1.0,"a",transform=ax.transAxes,fontsize=17,weight="bold",va="center")
    ax.text(0,1.03,"Top 20 countries by endpoint activity, plus all five permanent UN Security Council members (*)",transform=ax.transAxes,fontsize=10)
    fig.colorbar(image,ax=ax,pad=.015,fraction=.016,label="Endpoint activity exposed (%)")
    monthly=pd.read_csv(OUT/"appendix_monthly_coverage.csv")
    summaries=pd.read_csv(OUT/"single_closure_results.csv")
    month=summaries[summaries.scenario_index>0].groupby(["shiptype","month"])[["affected_weight","arctic_weight"]].sum()
    month["arctic_pct"]=100*month.arctic_weight/month.affected_weight
    for k in [0,1]:
        ax=fig.add_subplot(gs[1,k])
        a=(monthly.pivot(index="shiptype",columns="start_month",values="observed_pct") if k==0 else month.arctic_pct.unstack()).reindex(SHIPS).to_numpy()
        im=ax.imshow(a,aspect="auto",cmap="YlGnBu" if k==0 else "YlOrBr",vmin=0,vmax=100 if k==0 else max(5,np.ceil(a.max()/5)*5))
        ax.set_xticks(range(12),["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],rotation=45,ha="right",fontsize=9)
        ax.set_yticks(range(4),[DISPLAY[s] for s in SHIPS],fontsize=10)
        for y in range(4):
            for x in range(12):
                ax.text(x,y,f"{a[y,x]:.1f}",ha="center",va="center",fontsize=8,color="white" if a[y,x]>im.norm.vmax*.57 else "#243846")
        ax.text(-.13,1.1,"b" if k==0 else "c",transform=ax.transAxes,weight="bold",fontsize=17)
        ax.text(0,1.07,"Observed-AIS baseline activity (%)" if k==0 else "Arctic option among exposed activity (%)",transform=ax.transAxes,fontsize=10)
        fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.07,pad=.23)
    fig.text(.12,.02,"Main-figure d/e topics relocated here. Grey/NA means no matched demand, not zero trade.\nAll values are activity-index shares, not cargo or economic-impact shares.",fontsize=9)
    save(fig,"SupplementaryFig_CountryDependence_MonthlyCoverage")
    pd.DataFrame({"order":range(1,len(countries)+1),"iso3":countries,"country":labels}).to_csv(OUT/"appendix_country_order.csv",index=False)


def save(fig,name,output_dir=None):
    folder=Path(output_dir) if output_dir is not None else FIG
    folder.mkdir(parents=True,exist_ok=True)
    # Keep labels readable when the 19-inch working canvas is reduced for print.
    for text in fig.findobj(match=Text):
        if text.get_text():
            text.set_fontsize(max(13, text.get_fontsize()*1.55))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    rows={ax.get_gid():list(map(float,ax.get_position().bounds)) for ax in fig.axes
        if ax.get_gid() in ["main_a_world","main_b_heat","main_c_sankey"]}
    aligned=None
    if len(rows)==3:
        edges=np.array([[box[0],box[0]+box[2]] for box in rows.values()])
        np.testing.assert_allclose(edges,np.tile(edges[0],(3,1)),atol=1e-8,rtol=0)
        aligned=True
    outside=[]
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        box=text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            outside.append(dict(text=text.get_text(),bounds=list(map(float,box.bounds))))
    (folder/(name+"_layout_audit.json")).write_text(json.dumps(dict(texts_outside_canvas=outside,
        main_row_axes_bounds=rows,main_rows_same_left_and_right=aligned),indent=2))
    fig.savefig(folder/(name+".png"),dpi=220)
    fig.savefig(folder/(name+".pdf"),dpi=220)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--include-appendix",action="store_true")
    args=parser.parse_args()
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"axes.titlesize":11})
    context=samples_and_context()
    main_figure(context)
    if args.include_appendix:
        appendix()
    print("FIGURES", FIG, flush=True)
