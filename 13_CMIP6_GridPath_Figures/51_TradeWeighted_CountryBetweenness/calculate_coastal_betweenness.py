"""Demand-weighted national coastal-proximity betweenness on physical sea routes.

This is an observed-OD, single-shortest-path group-betweenness estimator,
not all-pairs Brandes centrality or a claim of maritime jurisdiction.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pycountry
import shapely
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from rebuild_mainfig2_latest_network import Geometry,ROUTES
from repair_cmip6_monthly_routes import repaired_route_metadata

OUT=HERE/"results"
EARTH=6371.0088
SPACING=2.
RADII=[25.,50.,98.,100.,102.,200.]
SHIPS=["container","bulk","tanker","lng"]
COAST_SOURCE=ROOT/"data/public/naturalearth_10m_countries/ne_10m_admin_0_countries.shp"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xyz(points):
    lon,lat=np.deg2rad(np.asarray(points,float)).T
    return np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])


def sample_line(points,spacing=SPACING):
    """Arc-length samples on the original spherical piecewise-geodesic polyline."""
    points=np.asarray(points,float)
    if len(points)==1:
        return xyz(points)
    v=xyz(points)
    theta=2*np.arcsin(np.minimum(np.linalg.norm(np.diff(v,axis=0),axis=1)/2,1))
    take=np.r_[True,theta>1e-12]
    v=v[take]
    if len(v)==1:
        return v
    theta=2*np.arcsin(np.minimum(np.linalg.norm(np.diff(v,axis=0),axis=1)/2,1))
    cum=np.r_[0,np.cumsum(theta*EARTH)]
    targets=np.r_[np.arange(0,cum[-1],spacing),cum[-1]]
    i=np.minimum(np.searchsorted(cum,targets,side="right")-1,len(theta)-1)
    t=(targets-cum[i])/(cum[i+1]-cum[i])
    a=theta[i]
    q=(np.sin((1-t)*a)[:,None]*v[i]+np.sin(t*a)[:,None]*v[i+1])/np.sin(a)[:,None]
    q/=np.linalg.norm(q,axis=1)[:,None]
    return q


def coastal_parts(geometry):
    """Do not join disconnected shores or count map cuts at +/-180 as coast."""
    if geometry.geom_type in ["LineString","LinearRing"]:
        points=shapely.get_coordinates(geometry)
        if len(points)<2:
            return
        seam=(np.abs(np.abs(points[:-1,0])-180)<1e-8)&(np.abs(np.abs(points[1:,0])-180)<1e-8)
        start=None
        for i,keep in enumerate(~seam):
            if keep and start is None:
                start=i
            if start is not None and (not keep or i==len(seam)-1):
                stop=i+2 if keep else i+1
                yield points[start:stop]
                start=None
    elif hasattr(geometry,"geoms"):
        for part in geometry.geoms:
            yield from coastal_parts(part)


def countries():
    cache=OUT/"coastal_countries.parquet"
    mapping=OUT/"country_code_mapping.csv"
    if cache.exists() and mapping.exists():
        return gpd.read_parquet(cache),pd.read_csv(mapping)
    world=gpd.read_file(COAST_SOURCE)
    world["admin_iso3"]=world.ISO_A3_EH.where(world.ISO_A3_EH.ne("-99"),world.ISO_A3)
    world["admin_iso3"]=world.admin_iso3.where(world.admin_iso3.ne("-99"),world.ADM0_A3)
    # Aggregate dependent territories to their Natural Earth parent-country label.
    # This mapping is a statistical grouping, not a determination of sovereignty.
    main=world[world.ADMIN.eq(world.SOVEREIGNT)].drop_duplicates("SOVEREIGNT").set_index("SOVEREIGNT").admin_iso3
    world["iso3"]=world.SOVEREIGNT.map(main).fillna(world.admin_iso3)
    table=world[["admin_iso3","iso3","ADMIN","SOVEREIGNT"]].copy()
    world.geometry=shapely.make_valid(world.geometry.to_numpy())
    global_land=shapely.union_all(world.geometry.to_numpy())
    coast=global_land.boundary
    records=[]
    for iso,g in world.groupby("iso3"):
        shape=shapely.union_all(g.geometry.to_numpy())
        xmin,ymin,xmax,ymax=shape.bounds
        nearby=shapely.clip_by_rect(coast,xmin-1e-7,ymin-1e-7,xmax+1e-7,ymax+1e-7)
        shoreline=shapely.intersection(shape.boundary,nearby)
        is_coastal=not shoreline.is_empty and iso!="ATA"
        records.append(dict(iso3=iso,name=g.SOVEREIGNT.iloc[0],coastal=is_coastal,
            geometry=shape,coast_wkb=shapely.to_wkb(shoreline)))
    result=gpd.GeoDataFrame(records,geometry="geometry",crs=4326)
    result.to_parquet(cache,index=False)
    table.to_csv(mapping,index=False)
    print("Coastal countries",int(result.coastal.sum()),flush=True)
    return result,table


def catalogue(mapping):
    marker=OUT/"physical_route_catalogue.json"
    if marker.exists():
        return json.loads(marker.read_text()),pd.read_parquet(OUT/"od_route_states.parquet")
    bank_table=pd.read_csv(OUT/"route_bank_registry.csv")
    geo=Geometry()
    points,_=repaired_route_metadata()
    iso_map=mapping.drop_duplicates("admin_iso3").set_index("admin_iso3").iso3.to_dict()
    features={}
    feature_points=[]
    route_lookup={}
    route_features=[]
    rows=[]

    def feature(key,p):
        if key not in features:
            features[key]=len(features)
            feature_points.append(sample_line(p))
        return features[key]

    def edge_features(nodes):
        found=[]
        for a,b in zip(nodes[:-1],nodes[1:]):
            edge=geo.lookup[(min(a,b),max(a,b))]
            gid=int(edge.geometry_id)
            p=geo.xy[geo.offsets[gid]:geo.offsets[gid+1]]
            found.append(feature("edge_"+str(gid),p))
        if not found and len(nodes):
            found=[feature("node_"+str(nodes[0]),geo.nodes[nodes[:1]])]
        return found

    def intern(ids):
        key=tuple(sorted(set(ids)))
        if not key:
            return -1
        if key not in route_lookup:
            route_lookup[key]=len(route_features)
            route_features.append(list(key))
        return route_lookup[key]

    @lru_cache(maxsize=65536)
    def path_features(folder,kind,pid):
        cat=np.load(Path(folder)/(kind+"_paths.npz"))
        nodes=cat["nodes"][cat["offsets"][pid]:cat["offsets"][pid+1]]
        return tuple(edge_features(nodes))

    # Load each catalogue once; np.load's compressed arrays are otherwise
    # decompressed repeatedly when indexing many OD paths.
    cats={}
    for item in bank_table.to_dict("records"):
        folder=ROOT/item["bank"]
        key=str(folder)
        if key not in cats:
            cats[key]={kind:{k:v.copy() for k,v in np.load(folder/(kind+"_paths.npz")).items()} for kind in ["conventional","arctic"]}

    @lru_cache(maxsize=65536)
    def cached_path(folder,kind,pid):
        cat=cats[folder][kind]
        nodes=cat["nodes"][cat["offsets"][pid]:cat["offsets"][pid+1]]
        return tuple(edge_features(nodes))

    for item in bank_table.to_dict("records"):
        folder=ROOT/item["bank"]
        demand=pd.read_parquet(folder/"demand.parquet")
        arrays={k:v.copy() for k,v in np.load(folder/"scenario_routes.npz").items()}
        si=int(item["bank_state"])
        eligible=demand.shiptype.eq(item["shiptype"])&demand.trade_weight_usd.fillna(0).gt(0)
        for j in np.flatnonzero(eligible):
            d=demand.iloc[j]
            conventional=arrays["without_km"][si,j]
            arctic=arrays["arctic_km"][si,j]
            aid=-1
            if np.isfinite(conventional):
                aid=intern(cached_path(str(folder),"conventional",int(arrays["conventional_id"][si,j])))
            bid=-1
            route="traditional"
            if np.isfinite(arctic):
                route=ROUTES[int(arrays["route_code"][si,j])-1]
                core=points[points.route.eq(route)&points.month.eq(item["month"])].sort_values("order")[["lon","lat"]].to_numpy(float)
                fid=feature(f"core_{route}_{item['month']}",core)
                bid=intern(cached_path(str(folder),"arctic",int(arrays["entry_id"][si,j]))+
                    cached_path(str(folder),"arctic",int(arrays["exit_id"][si,j]))+(fid,))
            for network in ["traditional","embedded_distance","embedded_premium05"]:
                use_b=network!="traditional" and np.isfinite(arctic) and arctic*(1.05 if network=="embedded_premium05" else 1)<conventional-1e-6
                rid=bid if use_b else aid
                rows.append(dict(shiptype=item["shiptype"],month=int(item["month"]),step=int(item["step"]),
                    network=network,od_id=int(d.od_id),origin_iso3=iso_map.get(d.origin_iso3,d.origin_iso3),
                    dest_iso3=iso_map.get(d.dest_iso3,d.dest_iso3),trade_weight_usd=float(d.trade_weight_usd),
                    route_id=rid,route_family=route if use_b else "traditional",
                    distance_km=arctic if use_b else conventional,reachable=rid>=0))
        print("Catalogued",item["shiptype"],item["step"],len(route_features),"distinct paths",flush=True)
    offsets=np.r_[0,np.cumsum([len(p) for p in feature_points])]
    np.savez_compressed(OUT/"sampled_route_features.npz",xyz=np.vstack(feature_points).astype(np.float32),offsets=offsets)
    result=dict(spacing_km=SPACING,earth_radius_km=EARTH,feature_keys=list(features),route_features=route_features)
    marker.write_text(json.dumps(result))
    data=pd.DataFrame(rows)
    data.to_parquet(OUT/"od_route_states.parquet",index=False)
    return result,data


def coast_distance(world,catalog):
    cached=OUT/"route_country_distances_v2.npz"
    ids=world.loc[world.coastal,"iso3"].tolist()
    if cached.exists():
        a=np.load(cached)
        assert a["countries"].tolist()==ids
        return a["distance_km"],ids
    sample=np.load(OUT/"sampled_route_features.npz")
    cloud=sample["xyz"].astype(float)
    cloud/=np.linalg.norm(cloud,axis=1)[:,None]
    offsets=sample["offsets"]
    feat=np.full((len(offsets)-1,len(ids)),np.inf,np.float32)
    cap=2*np.sin(205./EARTH/2)
    for column,row in enumerate(world[world.coastal].itertuples()):
        line=shapely.from_wkb(row.coast_wkb)
        p=[sample_line(g) for g in coastal_parts(line)]
        shore=np.vstack(p)
        tree=cKDTree(shore)
        chord,_=tree.query(cloud,k=1,distance_upper_bound=cap,workers=1)
        near=2*EARTH*np.arcsin(np.minimum(chord/2,1))
        near[~np.isfinite(chord)]=np.inf
        feat[:,column]=np.minimum.reduceat(near,offsets[:-1])
        if column%10==0:
            print("Coastal distances",column+1,"/",len(ids),row.iso3,flush=True)
    route=np.stack([feat[np.asarray(f,int)].min(axis=0) for f in catalog["route_features"]])
    np.savez_compressed(cached,distance_km=route,countries=np.asarray(ids))
    np.savez_compressed(OUT/"feature_country_distances_v2.npz",distance_km=feat,countries=np.asarray(ids))
    return route,ids


def aggregate(states,distances,ids,world):
    # Natural Earth also contains reef banks and buffer zones. They are not
    # country ranking units; preserve their geometry cache but exclude ranks.
    iso_codes={country.alpha_3 for country in pycountry.countries}
    keep=np.asarray([iso in iso_codes for iso in ids])
    names=world.set_index("iso3")["name"].to_dict()
    pd.DataFrame([dict(iso3=iso,name=names[iso],included_in_rank_universe=bool(valid))
        for iso,valid in zip(ids,keep)]).to_csv(OUT/"country_rank_universe.csv",index=False)
    distances=distances[:,keep]
    ids=np.asarray(ids)[keep].tolist()
    rows=[]
    countries_array=np.asarray(ids)
    for key,group in states.groupby(["shiptype","month","step","network"],sort=False):
        ship,month,step,network=key
        w=group.trade_weight_usd.to_numpy()
        rid=group.route_id.to_numpy(int)
        d=np.full((len(group),len(ids)),np.inf,float)
        good=rid>=0
        d[good]=distances[rid[good]]
        endpoint=(group.origin_iso3.to_numpy()[:,None]==countries_array)|(group.dest_iso3.to_numpy()[:,None]==countries_array)
        d[endpoint]=np.inf
        total=w.sum()
        for radius in RADII:
            passed=d<=radius
            value=100*(w[:,None]*passed).sum(axis=0)/total
            rank=pd.Series(value).round(10).rank(method="min",ascending=False).astype(int).to_numpy()
            for col,country in enumerate(ids):
                rows.append(dict(shiptype=ship,month=month,step=step,network=network,radius_km=radius,
                    iso3=country,name=names[country],betweenness_pct=value[col],rank=rank[col],
                    transit_od_count=int(passed[:,col].sum()),transit_trade_usd=float(w[passed[:,col]].sum()),
                    total_trade_usd=total,od_count=len(group),reachable_trade_pct=100*w[good].sum()/total))
    results=pd.DataFrame(rows)
    results.to_csv(OUT/"country_betweenness_all_states.csv",index=False)
    primary=results[results.radius_km.eq(100)]
    diffs=[]
    for (ship,step,radius),g in results.groupby(["shiptype","step","radius_km"]):
        a=g[g.network.eq("traditional")].set_index("iso3")
        b=g[g.network.eq("embedded_distance")].set_index("iso3")
        original=results[results.shiptype.eq(ship)&results.step.eq(0)&results.radius_km.eq(radius)&results.network.eq("traditional")].set_index("iso3")
        z=a[["name","betweenness_pct","rank","transit_od_count"]].rename(columns={"betweenness_pct":"traditional_pct","rank":"traditional_rank","transit_od_count":"traditional_od_count"})
        z["embedded_pct"],z["embedded_rank"]=b.betweenness_pct,b["rank"]
        z["rank_gain"]=z.traditional_rank-z.embedded_rank
        z["betweenness_change_pp"]=z.embedded_pct-z.traditional_pct
        z["open_original_pct"]=original.betweenness_pct
        z["open_original_rank"]=original["rank"]
        z["rank_gain_vs_open"]=z.open_original_rank-z.embedded_rank
        z["betweenness_change_vs_open_pp"]=z.embedded_pct-z.open_original_pct
        z["shiptype"],z["step"],z["radius_km"]=ship,step,radius
        diffs.append(z.reset_index())
    changes=pd.concat(diffs,ignore_index=True)
    changes.to_csv(OUT/"country_rank_changes.csv",index=False)
    sensitivity=[]
    for ship,g in results.groupby("shiptype"):
        base=g[g.step.eq(5)&g.network.eq("embedded_distance")&g.radius_km.eq(100)].set_index("iso3")
        for radius in RADII:
            other=g[g.step.eq(5)&g.network.eq("embedded_distance")&g.radius_km.eq(radius)].set_index("iso3").loc[base.index]
            top=set(base[base.betweenness_pct.gt(0)].sort_values("rank").head(20).index)
            top_other=set(other[other.betweenness_pct.gt(0)].sort_values("rank").head(20).index)
            sensitivity.append(dict(shiptype=ship,radius_km=radius,spearman_rho=float(spearmanr(base["rank"],other["rank"]).statistic),
                top20_overlap=len(top&top_other),top20_base=len(top),
                median_absolute_rank_change=float(np.median(np.abs(base["rank"]-other["rank"]))),
                max_absolute_rank_change=int(np.max(np.abs(base["rank"]-other["rank"])))))
    pd.DataFrame(sensitivity).to_csv(OUT/"radius_rank_sensitivity.csv",index=False)
    # A complete reproducible trail from each demand row to nearby transit countries.
    records=[]
    for row in states[states.network.isin(["traditional","embedded_distance"])].itertuples():
        if row.route_id<0:
            continue
        values=distances[row.route_id]
        for i in np.flatnonzero(values<=200):
            if ids[i] in [row.origin_iso3,row.dest_iso3]:
                continue
            records.append(dict(shiptype=row.shiptype,step=row.step,network=row.network,od_id=row.od_id,
                route_id=row.route_id,transit_country=ids[i],minimum_coastal_distance_km=float(values[i]),
                trade_weight_usd=row.trade_weight_usd,within_primary_100km=bool(values[i]<=100)))
    pd.DataFrame(records).to_parquet(OUT/"od_transit_country_attribution.parquet",index=False)
    print("Country scores, rankings and sensitivity saved",flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    world,mapping=countries()
    catalog,states=catalogue(mapping)
    distances,ids=coast_distance(world,catalog)
    aggregate(states,distances,ids,world)
    (OUT/"method_provenance.json").write_text(json.dumps(dict(status="computed",primary_radius_km=100,
        sensitivity_radii_km=RADII,coastal_sampling_spacing_km=SPACING,route_sampling_spacing_km=SPACING,
        sampling_distance_upper_bias_bound_km=2.,earth_radius_km=EARTH,
        shortest_path_policy="Primary: physical distance, no Arctic cost premium; 5% premium counterfactual retained as sensitivity",
        scenario_selection="Exactly the four vessel-specific nested 0--5 sequences retained in main Figure 3, not a new global optimum",
        metric="100 * sum(OD trade value * any coastal-proximity encounter excluding endpoint countries) / fixed total OD trade",
        normalization="Not a partition: one shortest route can contribute to multiple transit countries; totals need not sum to 100%",
        rank_tie_rule="Competition rank after rounding scores to 10 decimal places in percentage units; zero scores share the final rank",
        shortest_path_ties="One deterministic stored shortest feasible candidate per OD; equal-length alternatives are not enumerated or fractionally split",
        coverage="Observed represented, trade-covered vessel-month ODs, not all 1,390 ports or all theoretical country pairs",
        country_mapping="Natural Earth parent-country statistical grouping, exported; no assertion of legal jurisdiction",
        land_boundaries="Only the coastline of the union of land polygons; internal country borders excluded",
        sources={str(COAST_SOURCE.relative_to(ROOT)):sha(COAST_SOURCE),"route_bank_registry.csv":sha(OUT/"route_bank_registry.csv")},
        territorial_water_data_used=False,downloaded_maritime_boundaries_excluded=True,
        antimeridian_map_cut_segments_excluded_from_coast=True,disconnected_coast_components_never_joined=True,
        rank_universe="ISO 3166-1 alpha-3 coastal statistical units after Natural Earth parent grouping; non-ISO reef banks and buffer/de facto units excluded without assigning them to other countries",
        geometry_groups=len(ids),countries=int(pd.read_csv(OUT/"country_rank_universe.csv").included_in_rank_universe.sum()),
        distinct_geometric_path_feature_sets=len(catalog["route_features"]),od_route_states=len(states)),indent=2))


if __name__=="__main__":
    main()
