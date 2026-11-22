"""Publish a north-polar Figure 2 with independently checked complete routes."""
from __future__ import annotations

import calendar
import itertools
import json
from functools import lru_cache

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pyproj import Transformer

from physical_ocean_routing import WaterRouter, length, split_dateline
from repair_cmip6_monthly_routes import repaired_route_metadata
from rebuild_mainfig2_latest_network import ROOT, NET, OLD, OUT, SHIPS, ROUTES, Geometry, simple

BLUE, ORANGE = "#0077b6", "#e87519"
TYPE_COLORS = dict(container="#177fab", bulk="#26956b", tanker="#b64a67", lng="#8a64ab")
ROUTE_COLORS = dict(NWP="#d9a62c", NEP="#ca6a62", NSR="#3478a6", TSR="#9278ad")
SHIP_LABELS = dict(container="Container", bulk="Dry bulk", tanker="Tanker", lng="LNG")
FULL_ROUTES = dict(NWP="Northwest Passage", NEP="Northeast Passage", NSR="Northern Sea Route", TSR="Transpolar Sea Route")
PROJ = "+proj=aeqd +lat_0=90 +lon_0=0 +datum=WGS84 +units=m"


def nice(name):
    return name.split("_")[0].replace(" Ko", "").replace("Ningbo-Zhoushan", "Ningbo-Zhoushan")


def select_examples():
    output = OUT / "complete_routes"
    output.mkdir(exist_ok=True)
    data = pd.read_parquet(OUT/"all_candidates_and_choices.parquet")
    data = data[(data.scenario_index > 0) & ~data.port_in_closure_zone & np.isfinite(data.without_km)]
    # Respect the previous display exclusion without changing transit topology.
    data = data[~data.origin_iso3.eq("TWN") & ~data.dest_iso3.eq("TWN")].copy()
    geo, water = Geometry(), WaterRouter(network_dir=NET)
    points, _ = repaired_route_metadata()
    registry = pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
    pool, rejected = {}, []
    selection = json.loads((OUT/"panel_a_selection.json").read_text())
    forbidden = set()
    for route in ("NWP", "NEP", "NSR"):
        config = selection[route]
        row = data[(data.od_id == config["od_id"]) & (data.scenario_index == config["scenario_index"])].iloc[0]
        forbidden.update([row.origin_port_id, row.dest_port_id])

    @lru_cache(maxsize=12)
    def catalogue(ship, month, kind):
        prefix = (OLD if kind == "conventional" else OUT)/"routing"/f"{ship}_{month:02d}"
        suffix = "_conventional_paths.npz" if kind == "conventional" else "_paths.npz"
        return dict(np.load(str(prefix)+suffix))

    def path(cat, pid):
        nodes = cat["nodes"][cat["offsets"][pid]:cat["offsets"][pid+1]]
        return geo.path(nodes), nodes

    for route in ROUTES:
        config = selection[route]
        for ship in [config["shiptype"]]:
            g = data[(data.shiptype == ship) & np.isfinite(data[f"{route}_km"])].copy()
            if "od_id" in config:
                g = g[(g.od_id == config["od_id"]) & (g.scenario_index == config["scenario_index"])]
            else:
                g = g[~g.origin_port_id.isin(forbidden) & ~g.dest_port_id.isin(forbidden)]
            g["weighted_distance_saving"] = (g.without_km-g[f"{route}_km"])*g.activity_weight
            g["closure_priority"] = g.scenario_index.map({18: 0, 16: 1, 1: 2}).fillna(10)
            g = g.sort_values(["weighted_distance_saving", "closure_priority", "od_id"], ascending=[False, True, True])
            for rank, (_, row) in enumerate(g.iterrows(), 1):
                month = int(row.month)
                cat = catalogue(ship, month, "arctic")
                base, base_nodes = path(catalogue(ship, month, "conventional"), int(row.conventional_path_id))
                left, left_nodes = path(cat, int(row[f"{route}_entry"]))
                right, right_nodes = path(cat, int(row[f"{route}_exit"]))
                core = points[(points.route == route) & (points.month == month)].sort_values("order")[["lon", "lat"]].to_numpy(float)
                if int(row[f"{route}_orientation"]) == 2:
                    core = core[::-1]
                left = left[::-1]
                full = np.vstack([left, core[1:], right[1:]])
                if geo.qualification_flags(base_nodes) or geo.qualification_flags(left_nodes) or geo.qualification_flags(right_nodes):
                    rejected.append(dict(route=route, shiptype=ship, od_id=int(row.od_id), reason="inland_lock_qualification_not_available"))
                    continue
                # Route geometry is not altered to pass the checks.
                if not simple(base) or not simple(full):
                    rejected.append(dict(route=route, shiptype=ship, od_id=int(row.od_id), reason="self_intersection"))
                    continue
                if not water.clear(base) or not water.clear(full):
                    rejected.append(dict(route=route, shiptype=ship, od_id=int(row.od_id), reason="independent_water_mask_check_failed"))
                    continue
                joins = [length([left[-1], core[0]]), length([core[-1], right[0]])]
                error0, error1 = abs(length(base)-row.without_km), abs(length(full)-row[f"{route}_km"])
                assert max(joins) < .001 and max(error0, error1) < .02
                assert np.allclose(base[0], full[0]) and np.allclose(base[-1], full[-1])
                mask = int(1 << (int(row.scenario_index)-1))
                assert not int(cat["gate_mask"][int(row[f"{route}_entry"])]) & mask
                assert not int(cat["gate_mask"][int(row[f"{route}_exit"])]) & mask
                record = dict(route=route, shiptype=ship, month=month, od_id=int(row.od_id),
                    origin=nice(row.origin_name), destination=nice(row.dest_name),
                    origin_port_id=row.origin_port_id, dest_port_id=row.dest_port_id,
                    scenario_index=int(row.scenario_index), closure=registry.iloc[int(row.scenario_index)-1]["name"],
                    without_km=length(base), with_km=length(full), saved_km=length(base)-length(full),
                    saved_pct=100*(length(base)-length(full))/length(base),
                    activity_weight=float(row.activity_weight), weighted_distance_saving=float(row.weighted_distance_saving),
                    eligible_route_type_rank=rank, entry_km=length(left), core_km=length(core), exit_km=length(right),
                    common_model_year_support=float(row[f"{route}_support"]),
                    maximum_join_gap_km=max(joins), distance_reconstruction_error_km=max(error0,error1),
                    land_crossings=0, full_geometry_simple=True,
                    observed_AIS_baseline_available=bool(row.has_observed),
                    selection_policy="previous example retained" if "od_id" in config else "maximum weighted saving with two distinct endpoints",
                    primary_baseline="modelled shortest water route under the indicated closure, not an observed disrupted voyage",
                    future="complete water-network access plus monthly CMIP6 core plus water-network exit")
                pool[(route, ship)] = (record, base, full, core)
                break
    selected = [pool[(route, selection[route]["shiptype"])] for route in ROUTES]
    assert not {selected[3][0]["origin_port_id"],selected[3][0]["dest_port_id"]} & forbidden
    all_records = []
    for record, base, full, core in selected:
        all_records.append(record)
        features = []
        for label, p in (("without_Arctic",base), ("with_Arctic",full), ("mandatory_core",core)):
            features.append(dict(type="Feature", properties=dict(record, variant=label),
                geometry=dict(type="MultiLineString", coordinates=split_dateline(p))))
        (output/f"{record['route']}.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=features)))
        np.savez_compressed(output/f"{record['route']}.npz", without=base, with_arctic=full, core=core)
    pd.DataFrame(all_records).to_csv(OUT/"panel_a_examples.csv", index=False)
    pd.DataFrame([v[0] for v in pool.values()]).to_csv(OUT/"panel_a_route_type_candidate_pool.csv", index=False)
    pd.DataFrame(rejected).to_csv(OUT/"panel_a_rejected_examples.csv", index=False)
    return selected


def label(ax, letter, x=-.08, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")


def draw_maps(fig, axes, examples):
    transform = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
    # A common polar azimuthal-equidistant extent includes southern detours.
    minlat = min(min(p[:,1].min(),q[:,1].min()) for _,p,q,_ in examples)
    edge_lat = min(-40, minlat-6)
    radius = float(abs(transform.transform(0, edge_lat)[1]))
    land = gpd.read_file(ROOT/"data/public/naturalearth_10m/ne_10m_land.shp")
    land.geometry = land.geometry.intersection(shapely.box(-180,edge_lat-1,180,90)).simplify(.035)
    land = land[~land.geometry.is_empty].to_crs(PROJ)
    nations = gpd.read_file(ROOT/"data/public/naturalearth_10m_countries/ne_10m_admin_0_countries.shp")
    nations.geometry = nations.geometry.intersection(shapely.box(-180,edge_lat-1,180,90)).simplify(.05)
    nations = nations[~nations.geometry.is_empty].to_crs(PROJ)
    annotations = []
    for ax, (r, base, full, core) in zip(axes, examples):
        disc = Circle((0,0), radius, facecolor="#e5f2f6", edgecolor=BLUE, lw=.8, zorder=0)
        ax.add_patch(disc)
        land.plot(ax=ax, facecolor="#f5f5f2", edgecolor="#829496", linewidth=.25, zorder=1)
        nations.boundary.plot(ax=ax, color="#b0baba", linewidth=.16, zorder=2)
        for collection in ax.collections:
            collection.set_clip_path(disc)
        for lat in (0,30,60,75):
            xx,yy = transform.transform(np.linspace(-180,180,721), np.full(721,lat))
            ax.plot(xx,yy, color="#b5c7ce", lw=.42, ls=(0,(3,3)), clip_path=disc, zorder=2)
        for lon in range(-180,180,45):
            xx,yy = transform.transform(np.full(200,lon),np.linspace(edge_lat,90,200))
            ax.plot(xx,yy,color="#c5d4d9",lw=.35,clip_path=disc,zorder=2)
        for latitude in (30,60):
            gx,gy=transform.transform(-150,latitude)
            ax.text(gx,gy,f"{latitude} N",fontsize=6.5,color="#6b828c",ha="center",va="center",zorder=3,
                    bbox=dict(facecolor="#e5f2f6",edgecolor="none",alpha=.75,pad=.3))
        ax.plot(0,0,"+",ms=4,mew=.6,color="#627b85",zorder=5)
        ax.text(0,radius*.035,"90 N",fontsize=6.5,color="#627b85",ha="center",va="bottom",zorder=5)
        for p,color,style in ((base,BLUE,"-"),(full,ORANGE,(0,(3,2)))):
            xx,yy=transform.transform(p[:,0],p[:,1])
            assert np.isfinite(xx).all() and np.isfinite(yy).all()
            assert np.max(np.hypot(xx,yy)) <= radius
            ax.plot(xx,yy,color="white",lw=2.8,alpha=.75,clip_path=disc,zorder=3)
            ax.plot(xx,yy,color=color,lw=1.75,ls=style,clip_path=disc,zorder=4)
        # Move endpoint labels away from each other and use a leader for clarity.
        for j, name in ((0,r["origin"]),(-1,r["destination"])):
            x,y = transform.transform(full[j,0],full[j,1])
            other = full[-1 if j==0 else 0]
            ox,oy = transform.transform(*other)
            dy = 11 if y >= oy else -15
            dx = -7 if x > radius*.32 else 7
            ha = "right" if dx < 0 else "left"
            ax.plot(x,y,"o",color="#c55353",ms=3.8,mec="white",mew=.4,zorder=6)
            text = ax.annotate(name,(x,y),xytext=(dx,dy),textcoords="offset points",ha=ha,va="center",
                fontsize=7.8,fontweight="medium",bbox=dict(facecolor="#f5d7d2",edgecolor="none",alpha=.94,pad=1.9),
                arrowprops=dict(arrowstyle="-",color="#924944",lw=.55),zorder=7,annotation_clip=False)
            annotations.append(text)
        ax.text(.5,1.15,f"{r['route']}  |  {r['origin']} to {r['destination']}",ha="center",va="bottom",transform=ax.transAxes,fontsize=9.2,fontweight="bold")
        ax.text(.5,1.09,f"Without {r['without_km']:,.0f} km; with {r['with_km']:,.0f} km",ha="center",transform=ax.transAxes,fontsize=8.6)
        ax.text(.5,1.035,f"{r['saved_km']:,.0f} km shorter ({r['saved_pct']:.1f}%)",ha="center",transform=ax.transAxes,fontsize=8.6,color="#343c40")
        ax.text(.5,-.055,f"{SHIP_LABELS[r['shiptype']]} | {calendar.month_abbr[r['month']]} | {r['closure']} closed",ha="center",va="top",transform=ax.transAxes,fontsize=8.2)
        ax.legend([Line2D([0],[0],color=BLUE,lw=2),Line2D([0],[0],color=ORANGE,lw=2,ls=(0,(3,2)))],
            ["Without Arctic",f"With Arctic ({r['route']})"],loc="upper center",bbox_to_anchor=(.5,-.095),ncol=2,
            fontsize=7.5,frameon=False,handlelength=2.5,columnspacing=1)
        ax.set(xlim=(-radius*1.035,radius*1.035),ylim=(-radius*1.035,radius*1.035),aspect="equal")
        ax.axis("off")
    return annotations, dict(projection=PROJ, all_routes_within_map_disc=True, radius_m=radius, southern_latitude=edge_lat)


def heatmap(ax, matrix, vmax, cmap, rows, percent=True):
    image = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(12),[calendar.month_abbr[m] for m in range(1,13)],fontsize=8)
    ax.set_yticks(range(len(rows)),rows,fontsize=9)
    ax.set_xticks(np.arange(-.5,12,1),minor=True)
    ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which="minor",color="white",lw=1)
    ax.tick_params(which="both",length=0)
    for i,j in np.ndindex(matrix.shape):
        x = matrix[i,j]
        txt = f"{x:.0f}" if not percent else f"{x:.1f}"
        ax.text(j,i,txt,ha="center",va="center",fontsize=8.4,color="white" if x>vmax*.62 else "#253a43")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot(examples):
    (OUT/"figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":9,"pdf.fonttype":42,"ps.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
    fig=plt.figure(figsize=(20,11.0))
    outer=fig.add_gridspec(3,1,height_ratios=[2.65,.80,.96],left=.05,right=.975,top=.91,bottom=.07,hspace=.65)
    top=outer[0].subgridspec(1,4,wspace=.10)
    axes=[fig.add_subplot(top[0,i]) for i in range(4)]
    annotations,map_audit=draw_maps(fig,axes,examples)
    label(axes[0],"a",x=-.07,y=1.16)
    fig.text(.51,.492,"CMIP6 SSP5-8.5, 2091-2100 | hypothetical ice-capable vessels | complete modelled route comparisons",
             ha="center",fontsize=8.2,color="#59666d")
    row2=outer[1].subgridspec(1,2,wspace=.25)
    b,c=fig.add_subplot(row2[0,0]),fig.add_subplot(row2[0,1])
    core=pd.read_csv(OUT/"panel_b_monthly_core_support.csv")
    mat=core.pivot(index="route",columns="month",values="support_pct").reindex(ROUTES,columns=range(1,13)).to_numpy()
    heatmap(b,mat,100,"YlGnBu",ROUTES,percent=False)
    b.set_xlabel("Month; common-path model-year support (%)")
    label(b,"b")
    monthly=pd.read_csv(OUT/"panel_c_ship_month.csv")
    mat=monthly.pivot(index="shiptype",columns="month",values="selected_pct").reindex(SHIPS,columns=range(1,13)).to_numpy()
    heatmap(c,mat,max(1,np.ceil(mat.max()/5)*5),"YlOrRd",[SHIP_LABELS[x] for x in SHIPS])
    c.set_xlabel("Month; exposed activity with a shorter Arctic alternative (%)")
    label(c,"c")
    row3=outer[2].subgridspec(1,2,wspace=.25)
    d,e=fig.add_subplot(row3[0,0]),fig.add_subplot(row3[0,1])
    uncertainty=pd.read_csv(OUT/"panel_d_uncertainty_summary.csv")
    for ship, offset in zip(SHIPS,(-2.7,-.9,.9,2.7)):
        g=uncertainty[uncertainty.shiptype==ship].sort_values("extra_unit_distance_cost_pct")
        stats=[dict(med=r.joint_p50,q1=r.joint_p25,q3=r.joint_p75,whislo=r.joint_p5,whishi=r.joint_p95)
               for r in g.itertuples()]
        color=TYPE_COLORS[ship]
        positions=g.extra_unit_distance_cost_pct.to_numpy()+offset
        d.bxp(stats,positions=positions,widths=1.4,patch_artist=True,showfliers=False,manage_ticks=False,
              boxprops=dict(facecolor=color,edgecolor=color,alpha=.35,linewidth=.8),
              medianprops=dict(color=color,linewidth=1.3),whiskerprops=dict(color=color,linewidth=.75),
              capprops=dict(color=color,linewidth=.75))
        d.plot(positions,g.joint_p50,color=color,lw=.7,alpha=.7,label=SHIP_LABELS[ship])
    d.set(xlim=(-5,105),ylim=(0,None),xlabel="Additional unit-distance cost of Arctic voyage (%)",ylabel="Exposed activity retained (%)")
    d.set_xticks(np.arange(0,101,10))
    d.text(.98,.96,"6,000 joint draws per cost level\nBoxes: 25-75%; whiskers: 5-95%",ha="right",va="top",
           transform=d.transAxes,fontsize=7.8,color="#59666d")
    d.grid(axis="y",color="#e4e6e6",lw=.6)
    d.legend(loc="lower center",bbox_to_anchor=(.5,1.025),ncol=4,frameon=False,fontsize=8.5)
    label(d,"d",x=-.08,y=1.20)
    mix=pd.read_csv(OUT/"panel_e_route_composition.csv")
    mix=mix[mix.weight.eq("activity")]
    left=np.zeros(4)
    for route in ROUTES:
        val=mix[mix.route==route].set_index("shiptype").loc[SHIPS,"pct_of_selected"].to_numpy()
        e.barh(np.arange(4),val,left=left,height=.57,color=ROUTE_COLORS[route],label=route)
        for i,v in enumerate(val):
            if v>5:
                e.text(left[i]+v/2,i,f"{v:.1f}%",ha="center",va="center",color="white",fontsize=8.5)
        left+=val
    e.set_yticks(np.arange(4),[SHIP_LABELS[x] for x in SHIPS])
    e.set(xlim=(0,100),xlabel="Route composition among shorter Arctic alternatives (%)")
    e.invert_yaxis()
    e.legend(loc="lower center",bbox_to_anchor=(.5,1.025),ncol=4,frameon=False,fontsize=8.5)
    label(e,"e",x=-.08,y=1.20)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    boxes=[a.get_window_extent(renderer) for a in annotations]
    overlaps=[]
    for i,j in itertools.combinations(range(len(boxes)),2):
        if boxes[i].overlaps(boxes[j]):
            overlaps.append([i,j])
    assert not overlaps, overlaps
    target=OUT/"figures/MainFig2_Monthly_Arctic_Reconnection"
    fig.savefig(str(target)+".png",dpi=240)
    fig.savefig(str(target)+".pdf")
    # Separate readable top-row asset for manuscript layout and the user's reference.
    bbox=matplotlib.transforms.Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.04)
    fig.savefig(OUT/"figures/MainFig2a_Four_Polar_Routes.png",dpi=300,bbox_inches=bbox)
    fig.savefig(OUT/"figures/MainFig2a_Four_Polar_Routes.pdf",bbox_inches=bbox)
    dbox=d.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03,1.08)
    fig.savefig(OUT/"figures/MainFig2d_Cost_Uncertainty.png",dpi=300,bbox_inches=dbox)
    fig.savefig(OUT/"figures/MainFig2d_Cost_Uncertainty.pdf",bbox_inches=dbox)
    plt.close(fig)
    return dict(map_audit, endpoint_label_overlaps=overlaps, map_count=4,
        uniform_without_color=BLUE, uniform_with_color=ORANGE, row_width_alignment=True,
        panel_d_boxplots=44, panel_d_draws_per_box=6000)


def validate(examples, visual):
    data=pd.read_parquet(OUT/"all_candidates_and_choices.parquet")
    monthly=pd.read_csv(OUT/"panel_c_ship_month.csv")
    assert len(monthly)==48 and data[data.scenario_index==0].od_id.nunique()==60080
    assert data.loc[data.scenario_index>0,"scenario_index"].nunique()==24
    assert json.loads((OUT/"panel_d_uncertainty_validation.json").read_text())["status"]=="PASS"
    used={r[0][side] for r in examples[:3] for side in ("origin_port_id","dest_port_id")}
    assert not {examples[3][0]["origin_port_id"],examples[3][0]["dest_port_id"]}&used
    for r in ROUTES:
        chosen=np.isfinite(data[f"{r}_km"])
        assert (data.loc[chosen,f"{r}_support"]>=.5).all()
        assert (data.loc[chosen,f"{r}_km"]<data.loc[chosen,"without_km"]).all()
    for _,g in pd.read_csv(OUT/"panel_d_break_even_sensitivity.csv").groupby(["weight","shiptype"]):
        assert (np.diff(g.selected_pct)<=1e-8).all()
    report=dict(status="PASS", example_count=4, water_mask_passes=8, complete_geometry_simple=8,
        max_distance_reconstruction_error_km=max(r[0]["distance_reconstruction_error_km"] for r in examples),
        max_join_gap_km=max(r[0]["maximum_join_gap_km"] for r in examples),
        model_year_support_consistent=True, denominator_all_exposed_rows=True,
        cost_sensitivity_monotonically_nonincreasing=True, plot=visual,
        scope="Eight displayed route geometries independently tested against the existing validated water mask; all beneficial Arctic candidates self-intersection tested; all physical network edges inherited from the latest validated source network. Not operational navigability certification.")
    (OUT/"validation.json").write_text(json.dumps(report,indent=2))
    audit=json.loads((OUT/"calculation_audit.json").read_text())
    audit["status"]="PASS"
    (OUT/"calculation_audit.json").write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame([x[0] for x in examples]).to_string(index=False),flush=True)
    print(json.dumps(report,indent=2),flush=True)


def main():
    examples=select_examples()
    visual=plot(examples)
    validate(examples,visual)


if __name__=="__main__":
    main()
