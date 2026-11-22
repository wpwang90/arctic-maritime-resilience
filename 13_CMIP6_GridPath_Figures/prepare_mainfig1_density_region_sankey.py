"""Observed file-month support and mass-conserving region/gate/region flows."""
from pathlib import Path
import json
import shutil
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from rebuild_mainfig1_latest_network import ROOT, NET, OUT, sha

REGIONS = ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"]
EXCLUDED_ISO3 = {"TWN"}
NONE = "No listed chokepoint"


def unique_month_cell_support(lon, lat, month):
    """One archive member contributes once per month/cell, including revisits."""
    lon, lat, month = np.asarray(lon), np.asarray(lat), np.asarray(month)
    keep = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon)<=180) & (np.abs(lat)<=90) & (month>=1) & (month<=12)
    x=np.floor(np.mod(lon[keep]+180,360)).astype(int)
    y=np.floor(np.clip(lat[keep],-89.999999,89.999999)+90).astype(int)
    return np.unique(np.c_[month[keep],y,x],axis=0)


def backup():
    archive=OUT/"revisions/before_density_region_sankey"
    marker=archive/"archive_manifest.json"
    if marker.exists():
        return
    archive.mkdir(parents=True,exist_ok=True)
    files=list((OUT/"figures").glob("MainFig1*")) + [OUT/"README.md",OUT/"Captions_and_methods.md",OUT/"panel_c_flow.csv",ROOT/"plot_mainfig1_latest_network.py"]
    entries=[]
    for f in files:
        if f.exists():
            shutil.copy2(f,archive/f.name)
            entries.append(dict(file=str(f),sha256=sha(f)))
    marker.write_text(json.dumps(entries,indent=2))


def density():
    path=ROOT/"data/ethan_representative_ais_grid_density_1deg.csv"
    d=pd.read_csv(path)
    meta=json.loads((ROOT/"data/ethan_representative_trajectory_metadata.json").read_text())
    assert sha(Path(meta["source_zip"]))==meta["source_sha256"]
    assert not d.duplicated(["shiptype","month","lon_index","lat_index"]).any()
    assert d.month.between(1,12).all() and d.trajectory_support.gt(0).all()
    assert (d.trajectory_support<=d.ais_points).all()
    assert np.equal(d.trajectory_support,np.floor(d.trajectory_support)).all()
    result=d.groupby(["lon_index","lat_index"],as_index=False).trajectory_support.sum()
    result["D_g"]=np.log10(1+result.trajectory_support)
    result["lon"]=result.lon_index-179.5;result["lat"]=result.lat_index-89.5
    result.to_csv(OUT/"panel_a_file_month_support_1deg.csv",index=False)
    grid=np.zeros((180,360),np.int64)
    grid[result.lat_index,result.lon_index]=result.trajectory_support
    np.savez_compressed(OUT/"panel_a_file_month_support_1deg.npz",S_g=grid,D_g=np.log10(1+grid),
        bounds=np.array([-180,-90,180,90]),resolution_degrees=1.0,crs="EPSG:4326",row_order="south_to_north")
    test=unique_month_cell_support([1.1,1.2,1.1,1.1],[10.1]*4,[1,1,1,2])
    assert len(test)==2
    dates=pd.read_parquet(ROOT/"data/ethan_representative_trajectory_summary.parquet",columns=["start_timestamp","end_timestamp"])
    assert dates.start_timestamp.astype(str).str.startswith("2024").all()
    assert dates.end_timestamp.astype(str).str.startswith("2024").all()
    report=dict(source_csv=str(path),source_csv_sha256=sha(path),source_archive_sha256=meta["source_sha256"],
        archive_members=meta["archive_members_processed"],resolution_degrees=1,
        positive_cells=len(result),total_file_month_cell_support=int(grid.sum()),maximum_S_g=int(grid.max()),
        maximum_D_g=float(np.log10(1+grid.max())),source_start_and_end_dates_in_2024=True,
        repeated_visit_test="PASS",definition="count unique (archive_member, actual_calendar_month, cell), then sum ship types and months",
        scope="All archive members with valid observed positions, including files that fail the stricter complete-OD geometry screen",
        exclusions="No modelled points, no interpolated points, no activity weighting and no clipping of high support",
        colour_scope="Deep-water category only; values are 1-degree statistics sampled on the existing fine categorical mask")
    (OUT/"panel_a_density_audit.json").write_text(json.dumps(report,indent=2))
    print("DENSITY",report,flush=True)


def country_mapping():
    folder=ROOT/"data/public/naturalearth_10m_countries"
    folder.mkdir(exist_ok=True)
    mapping={};sources=[]
    for layer in ["countries","map_units"]:
        stem="ne_10m_admin_0_"+layer
        archive=folder/(stem+".zip")
        url="https://naturalearth.s3.amazonaws.com/10m_cultural/"+stem+".zip"
        if not archive.exists():
            response=requests.get(url,timeout=90);response.raise_for_status()
            archive.write_bytes(response.content)
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if Path(member).name!=member:
                    raise ValueError("Unexpected archive path")
            if not (folder/(stem+".shp")).exists():
                z.extractall(folder)
        frame=gpd.read_file(folder/(stem+".shp"))
        sources.append(dict(url=url,archive_sha256=sha(archive)))
        for field in ["ISO_A3","ISO_A3_EH","GU_A3","ADM0_A3"]:
            for row in frame.itertuples():
                code=str(getattr(row,field,""));region=row.CONTINENT;region_field="CONTINENT"
                # Some island entries use an ocean category; retain their supplied UN region.
                if region not in REGIONS and row.REGION_UN in REGIONS:
                    region=row.REGION_UN;region_field="REGION_UN"
                if code!="-99" and len(code)==3 and region in REGIONS:
                    mapping.setdefault(code,dict(iso3=code,region=region,country=row.ADMIN,
                        source_field=field,region_field=region_field,source_layer=layer))
    table=pd.DataFrame(mapping.values()).sort_values("iso3")
    table.to_csv(OUT/"panel_c_country_region_mapping.csv",index=False)
    (OUT/"panel_c_region_mapping_source.json").write_text(json.dumps(dict(sources=sources,
        documentation="https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/",
        classification="country/territory CONTINENT field, with map-unit and REGION_UN island fallbacks; not commodity production/consumption or port-level continental point-in-polygon"),indent=2))
    return table.set_index("iso3").region


def allocate_gate_weights(weight, mask, keys):
    used=[key for i,key in enumerate(keys) if int(mask)&(1<<i)]
    used=used or [NONE]
    return [(key,float(weight)/len(used)) for key in used]


def sankey():
    mapping=country_mapping()
    d=pd.read_parquet(OUT/"baseline_all.parquet")
    ports=pd.read_csv(NET/"canonical_port_registry.csv").set_index("key")
    for side in ["origin","dest"]:
        d[f"{side}_canonical_iso3"]=d[f"{side}_port_id"].map(ports.iso3)
        d[f"{side}_region"]=d[f"{side}_canonical_iso3"].map(mapping)
    excluded=d.origin_canonical_iso3.isin(EXCLUDED_ISO3)|d.dest_canonical_iso3.isin(EXCLUDED_ISO3)
    missing=d.origin_region.isna()|d.dest_region.isna()
    same=d.origin_region.eq(d.dest_region)
    d["selection_status"]=np.select([excluded,missing,same],
        ["excluded_display_region","unidentified_endpoint_region","same_continent"],default="included")
    d[["od_id","shiptype","start_month","activity_weight","origin_canonical_iso3","dest_canonical_iso3",
       "origin_region","dest_region","baseline_source","selection_status"]].to_parquet(OUT/"panel_c_sample_selection.parquet",index=False)
    eligible=d[d.selection_status.eq("included")].copy()
    registry=pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
    keys=registry.key.tolist()
    records=[]
    for row in eligible.itertuples():
        for gate,weight in allocate_gate_weights(row.activity_weight,row.baseline_mask,keys):
            records.append(dict(od_id=row.od_id,origin=row.origin_region,gate=gate,dest=row.dest_region,
                allocated_weight=weight,shiptype=row.shiptype,month=row.start_month,baseline_source=row.baseline_source))
    allocated=pd.DataFrame(records)
    total=float(eligible.activity_weight.sum())
    assert np.isclose(total,allocated.allocated_weight.sum(),rtol=1e-12)
    restored=allocated.groupby("od_id").allocated_weight.sum().sort_index()
    np.testing.assert_allclose(restored,eligible.set_index("od_id").activity_weight.sort_index(),rtol=1e-12)
    flows=allocated.groupby(["origin","gate","dest"],as_index=False).allocated_weight.sum()
    flows["percent"]=100*flows.allocated_weight/total
    flows.to_csv(OUT/"panel_c_region_gate_flows.csv",index=False)
    allocated.to_parquet(OUT/"panel_c_allocated_od_weights.parquet",index=False)
    gates=registry[["number","key","name"]].rename(columns={"key":"gate"})
    gates=pd.concat([gates,pd.DataFrame([dict(number=0,gate=NONE,name=NONE)])],ignore_index=True)
    gates["allocated_weight"]=gates.gate.map(flows.groupby("gate").allocated_weight.sum()).fillna(0)
    gates["percent"]=100*gates.allocated_weight/total
    gates.to_csv(OUT/"panel_c_gate_shares.csv",index=False)
    stages=[]
    for side in ["origin","dest"]:
        series=flows.groupby(side).allocated_weight.sum().reindex(REGIONS,fill_value=0)
        for region,weight in series.items():
            stages.append(dict(stage=side,region=region,allocated_weight=float(weight),percent=100*weight/total))
    pd.DataFrame(stages).to_csv(OUT/"panel_c_endpoint_region_shares.csv",index=False)
    np.testing.assert_allclose(flows.groupby("origin").allocated_weight.sum().sort_index(),eligible.groupby("origin_region").activity_weight.sum().sort_index())
    np.testing.assert_allclose(flows.groupby("dest").allocated_weight.sum().sort_index(),eligible.groupby("dest_region").activity_weight.sum().sort_index())
    split=allocate_gate_weights(100,(1<<keys.index("Suez"))|(1<<keys.index("Bab_el_Mandeb")),keys)
    assert sorted(weight for _,weight in split)==[50,50]
    assert allocate_gate_weights(100,0,keys)==[(NONE,100)]
    report=dict(input_rows=len(d),included_intercontinental_rows=len(eligible),included_activity_weight=total,
        sample_selection=d.groupby("selection_status").agg(rows=("od_id","size"),weight=("activity_weight","sum")).to_dict("index"),
        excluded_display_iso3=sorted(EXCLUDED_ISO3),gates=24,no_gate_category=NONE,
        visible_flow_triples=len(flows),positive_gate_categories=int(gates.allocated_weight.gt(0).sum()),
        gate_percent_sum=float(gates.percent.sum()),Suez_percent=float(gates.loc[gates.gate.eq("Suez"),"percent"].iloc[0]),
        source_mix=eligible.groupby("baseline_source").activity_weight.sum().to_dict(),
        od_weight_conservation="PASS",both_endpoint_marginals="PASS",two_gate_100_split="50 + 50 PASS",no_gate_test="PASS",
        input_baseline_sha256=sha(OUT/"baseline_all.parquet"),panel_b_sha256=sha(OUT/"panel_b_exposure.csv"),
        denominator="Selected cross-continent, known-endpoint, display-eligible latest canonical OD/type/month activity; not world trade",
        endpoint_meaning="Shipping origin/destination country-based region; not ultimate production or final consumption",
        relation_to_closures="Baseline descriptive accounting, not a closure outcome or the previous two-stage counterfactual Sankey")
    (OUT/"panel_c_region_sankey_audit.json").write_text(json.dumps(report,indent=2))
    print("SANKEY",report,flush=True)


if __name__=="__main__":
    backup()
    density()
    from prepare_mainfig1_trade_weights import prepare_trade_sankey
    prepare_trade_sankey()
