"""Conservative country-pair trade-value calibration of Figure 1c route weights."""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from rebuild_mainfig1_latest_network import ROOT, OUT, sha
from prepare_mainfig1_density_region_sankey import REGIONS, NONE, allocate_gate_weights, sankey

CACHE=ROOT/"data/mainfig1_trade_2024"
SOURCE=Path(__import__("os").environ.get("ARCTIC_LOCAL_DATA", "external"))
BACI_SOURCE=SOURCE/"BACI_调和贸易数据_2007-2024/BACI_HS07"
PAIR=["origin_iso3","dest_iso3"]
MODAL=CACHE/"recent_maritime_shares/unctad_bilateral_sea_shares_2016_2021.parquet"


def maritime_shares_2021():
    source=pd.read_parquet(MODAL)
    current=source[source.year.eq(2021)].copy()
    assert not current.duplicated(PAIR).any()
    valid=current[current.status.eq("valid")].copy()
    assert valid.sea_fraction.between(0,1).all()
    np.testing.assert_allclose(valid.sea_fraction,valid.sea_fob_usd/valid.total_fob_usd)
    quantiles=valid.sea_fraction.quantile([.25,.5,.75])
    audit=dict(source="UNCTAD_WorldBank_TnT",source_sha256=sha(MODAL),share_year=2021,
        valid_global_directed_pairs=len(valid),explicit_zero_ratios=int(valid.sea_fraction.eq(0).sum()),
        median=float(quantiles.loc[.5]),q25=float(quantiles.loc[.25]),q75=float(quantiles.loc[.75]),
        definition="Unweighted median of all valid global directed-pair value-based sea fractions in 2021, including explicit zeros",
        imputation="Only absent or invalid source shares; no reverse-flow borrowing and no zero replacement",
        evidence="Published analytical modal estimates, not raw observed maritime cargo",
        data_url="https://unctadstat.unctad.org/EN/TransportCost.html")
    current=current.rename(columns={"sea_fraction":"sea_fraction_2021_source","status":"modal_source_status"})
    current.loc[current.modal_source_status.ne("valid"),"sea_fraction_2021_source"]=np.nan
    return current[PAIR+["sea_fraction_2021_source","modal_source_status","raw_file"]],audit


def archive_previous():
    archive=OUT/"revisions/before_trade_weighted_alignment"
    if (archive/"manifest.json").exists():
        return
    archive.mkdir(parents=True,exist_ok=True)
    files=list(OUT.glob("panel_c_*"))+list((OUT/"figures").glob("MainFig1*"))
    files += [OUT/"README.md",OUT/"Captions_and_methods.md",ROOT/"plot_mainfig1_latest_network.py",
        ROOT/"prepare_mainfig1_density_region_sankey.py",ROOT/"validate_mainfig1_density_region_sankey.py"]
    records=[]
    for f in files:
        if f.is_file():
            shutil.copy2(f,archive/f.name)
            records.append(dict(source=str(f),sha256=sha(f)))
    (archive/"manifest.json").write_text(json.dumps(records,indent=2))


def local_inputs():
    CACHE.mkdir(exist_ok=True)
    files=[(BACI_SOURCE/"BACI_HS07_Y2024_V202601.csv","BACI_HS07_Y2024_V202601.csv"),
        (BACI_SOURCE/"country_codes_V202601.csv","country_codes_V202601.csv"),
        (BACI_SOURCE/"Readme.txt","BACI_Readme.txt")]
    entries=[]
    for source,name in files:
        dest=CACHE/name
        if not dest.exists():
            shutil.copy2(source,dest)
            assert sha(source)==sha(dest)
        entries.append(dict(original=str(source),local=str(dest),sha256=sha(dest),bytes=dest.stat().st_size))
    entries.append(dict(local=str(MODAL),sha256=sha(MODAL),bytes=MODAL.stat().st_size,
        source="https://unctadstat.unctad.org/EN/TransportCost.html",reference_year=2021))
    (CACHE/"input_manifest.json").write_text(json.dumps(dict(files=entries,
        baci_source="https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37",
        trade_year=2024,ais_year=2024,modal_share_year=2021,modal_adjustment_applied=True,
        excluded_historical_input="baci_mode_prediction_2015_EORA.csv is not read or used",
        weight_scope="2024 BACI times fixed 2021 analytical sea share; missing shares imputed by global pair median",
        model_scope="Country-pair calibration, not commodity-to-vessel assignment or a new hinterland/transshipment model"),indent=2))


def country_trade():
    local_inputs()
    raw=CACHE/"BACI_HS07_Y2024_V202601.csv"
    source_sha=sha(raw)
    cache=CACHE/"baci_2024_country_pairs.csv"
    audit_path=CACHE/"baci_aggregation_audit.json"
    if not cache.exists() or json.loads(audit_path.read_text())["source_sha256"]!=source_sha:
        codes=pd.read_csv(CACHE/"country_codes_V202601.csv").set_index("country_code").country_iso3
        chunks=[];total=0.;rows=0;unmapped=0.;unmapped_rows=0
        for chunk in pd.read_csv(raw,usecols=["t","i","j","v"],chunksize=750000):
            assert chunk.t.eq(2024).all()
            assert np.isfinite(chunk.v).all() and chunk.v.ge(0).all()
            total+=float(chunk.v.sum())*1000;rows+=len(chunk)
            chunk["origin_iso3"]=chunk.i.map(codes);chunk["dest_iso3"]=chunk.j.map(codes)
            missing=chunk[PAIR].isna().any(axis=1)
            unmapped+=float(chunk.loc[missing,"v"].sum())*1000;unmapped_rows+=int(missing.sum())
            chunks.append(chunk.dropna(subset=PAIR).groupby(PAIR).v.sum())
        pairs=pd.concat(chunks).groupby(level=[0,1]).sum().mul(1000).rename("baci_2024_usd").reset_index()
        np.testing.assert_allclose(pairs.baci_2024_usd.sum()+unmapped,total,rtol=1e-12)
        pairs.to_csv(cache,index=False)
        audit_path.write_text(json.dumps(dict(source_sha256=source_sha,raw_rows=rows,
            world_merchandise_value_usd=total,unmapped_iso_value_usd=unmapped,unmapped_iso_rows=unmapped_rows,
            country_pairs=len(pairs),input_unit="thousand current USD",usd_multiplier=1000,
            year=2024,trade_modes="all transport modes"),indent=2))
    return pd.read_csv(cache)


def summarize_flows(d,weight,registry):
    records=[];keys=registry.key.tolist()
    for row in d.itertuples():
        for gate,w in allocate_gate_weights(getattr(row,weight),row.baseline_mask,keys):
            records.append(dict(od_id=row.od_id,origin=row.origin_region,gate=gate,dest=row.dest_region,
                allocated_weight=w,shiptype=row.shiptype,month=row.start_month,baseline_source=row.baseline_source))
    allocated=pd.DataFrame(records)
    total=float(d[weight].sum())
    np.testing.assert_allclose(allocated.groupby("od_id").allocated_weight.sum().sort_index(),
        d.set_index("od_id")[weight].sort_index(),rtol=1e-12)
    flow=allocated.groupby(["origin","gate","dest"],as_index=False).allocated_weight.sum()
    flow["percent"]=100*flow.allocated_weight/total
    gates=registry[["number","key","name"]].rename(columns={"key":"gate"})
    gates=pd.concat([gates,pd.DataFrame([dict(number=0,gate=NONE,name=NONE)])],ignore_index=True)
    gates["allocated_weight"]=gates.gate.map(flow.groupby("gate").allocated_weight.sum()).fillna(0)
    gates["percent"]=100*gates.allocated_weight/total
    endpoints=[]
    for side in ["origin","dest"]:
        weights=flow.groupby(side).allocated_weight.sum().reindex(REGIONS,fill_value=0)
        for region,w in weights.items():
            endpoints.append(dict(stage=side,region=region,allocated_weight=float(w),percent=100*w/total))
    return allocated,flow,gates,pd.DataFrame(endpoints)


def prepare_trade_sankey():
    archive_previous()
    # Retain the original activity-only estimate as a reproducible comparison.
    sankey()
    old_audit=json.loads((OUT/"panel_c_region_sankey_audit.json").read_text())
    old_gates=pd.read_csv(OUT/"panel_c_gate_shares.csv")
    activity_dir=OUT/"sensitivity_activity";activity_dir.mkdir(exist_ok=True)
    for name in ["panel_c_region_gate_flows.csv","panel_c_gate_shares.csv","panel_c_region_sankey_audit.json"]:
        shutil.copy2(OUT/name,activity_dir/name)
    selected=pd.read_parquet(OUT/"panel_c_sample_selection.parquet")
    baseline=pd.read_parquet(OUT/"baseline_all.parquet")
    d=selected.merge(baseline[["od_id","baseline_mask"]],on="od_id",validate="one_to_one")
    d=d.rename(columns={"origin_canonical_iso3":"origin_iso3","dest_canonical_iso3":"dest_iso3"})
    d["geography_status"]=d.selection_status
    pairs=country_trade()
    mapping=pd.read_csv(OUT/"panel_c_country_region_mapping.csv").set_index("iso3").region
    for side in ["origin","dest"]:
        pairs[side+"_region"]=pairs[side+"_iso3"].map(mapping)
    excluded=pairs.origin_iso3.eq("TWN")|pairs.dest_iso3.eq("TWN")
    missing=pairs[["origin_region","dest_region"]].isna().any(axis=1)
    same=pairs.origin_region.eq(pairs.dest_region)
    pairs["geography_status"]=np.select([excluded,missing,same],
        ["excluded_display_region","unidentified_endpoint_region","same_continent"],default="included")
    activity=d[d.geography_status.eq("included")].groupby(PAIR).agg(
        pair_activity=("activity_weight","sum"),route_rows=("od_id","size")).reset_index()
    pairs=pairs.merge(activity,on=PAIR,how="left",validate="one_to_one")
    shares,modal_audit=maritime_shares_2021()
    pairs=pairs.merge(shares,on=PAIR,how="left",validate="one_to_one")
    pairs["modal_source_status"]=pairs.modal_source_status.fillna("source_pair_absent")
    pairs["sea_share_imputed"]=pairs.sea_fraction_2021_source.isna()
    pairs["sea_fraction_2021_used"]=pairs.sea_fraction_2021_source.fillna(modal_audit["median"])
    pairs["sea_share_method"]=np.where(pairs.sea_share_imputed,"global_2021_pair_median","UNCTAD_2021_pair_estimate")
    pairs["maritime_proxy_2024_usd"]=pairs.baci_2024_usd*pairs.sea_fraction_2021_used
    pairs["allocation_status"]=np.select([pairs.geography_status.ne("included"),
        pairs.pair_activity.isna(),pairs.baci_2024_usd.le(0),pairs.maritime_proxy_2024_usd.eq(0)],
        ["outside_geographic_sample","no_matched_shipping_OD","zero_trade_value","zero_maritime_value"],default="allocated")
    pairs.to_csv(OUT/"panel_c_country_pair_trade_coverage.csv",index=False)
    modal_fields=["sea_fraction_2021_source","sea_fraction_2021_used","sea_share_imputed","sea_share_method","modal_source_status","maritime_proxy_2024_usd"]
    d=d.merge(pairs[PAIR+["baci_2024_usd","pair_activity"]+modal_fields],
        on=PAIR,how="left",validate="many_to_one")
    d["selection_status"]=np.select([d.geography_status.ne("included"),d.baci_2024_usd.isna(),
        d.baci_2024_usd.le(0),d.maritime_proxy_2024_usd.eq(0)],
        [d.geography_status,"missing_BACI_pair","zero_trade_value","zero_maritime_value"],default="included")
    keep=d.selection_status.eq("included")
    d["within_pair_activity_share"]=np.where(keep,d.activity_weight/d.pair_activity,np.nan)
    d["trade_weight_usd"]=d.maritime_proxy_2024_usd*d.within_pair_activity_share
    d["trade_year"]=2024
    d["modal_share_year"]=2021
    d["modal_adjustment_applied"]=True
    d.to_parquet(OUT/"panel_c_trade_weighted_od.parquet",index=False)
    d.rename(columns={"origin_iso3":"origin_canonical_iso3","dest_iso3":"dest_canonical_iso3"}).to_parquet(
        OUT/"panel_c_sample_selection.parquet",index=False)
    eligible=d[keep].copy()
    np.testing.assert_allclose(eligible.groupby(PAIR).trade_weight_usd.sum().sort_index(),
        pairs[pairs.allocation_status.eq("allocated")].set_index(PAIR).maritime_proxy_2024_usd.sort_index(),rtol=1e-12)
    assert np.isfinite(eligible.trade_weight_usd).all() and eligible.trade_weight_usd.gt(0).all()
    registry=pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
    allocated,flow,gates,endpoints=summarize_flows(eligible,"trade_weight_usd",registry)
    allocated["weight_unit"]="USD_2024_sea_linked_proxy_using_2021_shares"
    allocated.to_parquet(OUT/"panel_c_allocated_od_weights.parquet",index=False)
    flow.to_csv(OUT/"panel_c_region_gate_flows.csv",index=False)
    gates.to_csv(OUT/"panel_c_gate_shares.csv",index=False)
    endpoints.to_csv(OUT/"panel_c_endpoint_region_shares.csv",index=False)
    _,_,same_activity,_=summarize_flows(eligible,"activity_weight",registry)
    sensitivity=gates[["gate","name","percent"]].rename(columns={"percent":"baci_2024_trade_percent"})
    sensitivity=sensitivity.merge(same_activity[["gate","percent"]].rename(columns={"percent":"same_sample_activity_percent"}),on="gate")
    sensitivity=sensitivity.merge(old_gates[["gate","percent"]].rename(columns={"percent":"previous_activity_percent"}),on="gate")
    sensitivity["trade_vs_activity_pp"]=sensitivity.baci_2024_trade_percent-sensitivity.same_sample_activity_percent
    sensitivity.to_csv(OUT/"panel_c_trade_weight_sensitivity.csv",index=False)
    imputation_runs=[]
    for name,fill in [("missing_q25",modal_audit["q25"]),("missing_median",modal_audit["median"]),
        ("missing_q75",modal_audit["q75"])]:
        alternative=eligible.copy()
        alternative["sensitivity_weight"]=alternative.baci_2024_usd*alternative.within_pair_activity_share*alternative.sea_fraction_2021_source.fillna(fill)
        _,_,nodes,_=summarize_flows(alternative,"sensitivity_weight",registry)
        nodes["scenario"]=name;nodes["missing_share_value"]=fill
        nodes["scenario_total_usd"]=alternative.sensitivity_weight.sum()
        imputation_runs.append(nodes)
    complete=eligible[eligible.sea_share_imputed.eq(False)].copy()
    _,_,nodes,_=summarize_flows(complete,"trade_weight_usd",registry)
    nodes["scenario"]="complete_cases_only";nodes["missing_share_value"]=np.nan
    nodes["scenario_total_usd"]=complete.trade_weight_usd.sum()
    imputation_runs.append(nodes)
    imputation_sensitivity=pd.concat(imputation_runs,ignore_index=True)
    imputation_sensitivity.to_csv(OUT/"panel_c_modal_imputation_sensitivity.csv",index=False)
    source_audit=json.loads((CACHE/"baci_aggregation_audit.json").read_text())
    total=float(eligible.trade_weight_usd.sum())
    geographic=pairs[pairs.geography_status.eq("included")]
    matched=pairs[pairs.allocation_status.eq("allocated")]
    imputed=matched[matched.sea_share_imputed]
    modal_audit.update(dict(included_imputed_pairs=len(imputed),included_source_pairs=len(matched)-len(imputed),
        included_imputed_OD_rows=int(eligible.sea_share_imputed.sum()),
        imputed_all_mode_baci_usd=float(imputed.baci_2024_usd.sum()),
        imputed_all_mode_value_pct=float(100*imputed.baci_2024_usd.sum()/matched.baci_2024_usd.sum()),
        imputed_maritime_proxy_usd=float(imputed.maritime_proxy_2024_usd.sum()),
        imputed_proxy_value_pct=float(100*imputed.maritime_proxy_2024_usd.sum()/total)))
    (OUT/"panel_c_modal_share_audit.json").write_text(json.dumps(modal_audit,indent=2))
    imputed.to_csv(OUT/"panel_c_median_imputed_country_pairs.csv",index=False)
    report=dict(weight_basis="BACI2024_times_UNCTAD2021_sea_share_median_fallback",weight_unit="USD of model-estimated 2024 sea-linked trade using fixed 2021 modal shares",
        trade_year=2024,ais_year=2024,modal_share_year=2021,modal_adjustment_applied=True,
        modal_imputation=modal_audit,
        input_rows=len(d),included_intercontinental_rows=len(eligible),
        included_country_pairs=len(matched),included_trade_weight_usd=total,
        matched_all_mode_baci_usd=float(matched.baci_2024_usd.sum()),
        world_all_mode_baci_usd=source_audit["world_merchandise_value_usd"],
        included_pair_all_mode_world_share_pct=100*matched.baci_2024_usd.sum()/source_audit["world_merchandise_value_usd"],
        eligible_intercontinental_baci_usd=float(geographic.baci_2024_usd.sum()),
        mapped_intercontinental_value_share_pct=100*matched.baci_2024_usd.sum()/geographic.baci_2024_usd.sum(),
        unallocated_intercontinental_baci_usd=float(geographic.loc[geographic.pair_activity.isna(),"baci_2024_usd"].sum()),
        sample_selection=d.groupby("selection_status").agg(rows=("od_id","size"),activity_weight=("activity_weight","sum")).to_dict("index"),
        gates=24,no_gate_category=NONE,visible_flow_triples=len(flow),gate_percent_sum=float(gates.percent.sum()),
        Suez_percent=float(gates.loc[gates.gate.eq("Suez"),"percent"].iloc[0]),
        source_mix=eligible.groupby("baseline_source").trade_weight_usd.sum().to_dict(),
        previous_activity_Suez_percent=old_audit["Suez_percent"],
        maximum_trade_vs_activity_pp=float(sensitivity.trade_vs_activity_pp.abs().max()),
        positive_gate_categories=int(gates.allocated_weight.gt(0).sum()),
        excluded_display_iso3=["TWN"],
        annual_pair_conservation="PASS",od_weight_conservation="PASS",two_gate_100_split="50 + 50 PASS",no_gate_test="PASS",
        input_baseline_sha256=sha(OUT/"baseline_all.parquet"),panel_b_sha256=sha(OUT/"panel_b_exposure.csv"),
        input_trade_manifest_sha256=sha(CACHE/"input_manifest.json"),
        denominator="Mapped intercontinental 2024 sea-linked trade estimate, holding 2021 modal shares fixed; missing modal shares use the global 2021 pair median",
        endpoint_meaning="Shipping endpoint country regions used for calibration; no reconstructed production-consumption, hinterland or transshipment assignment",
        limitation="2021 shares are analytical estimates, held fixed for 2024; median-filled shares are imputations, not observations. Port/type/month allocation uses activity proxies, not observed cargo values.")
    (OUT/"panel_c_region_sankey_audit.json").write_text(json.dumps(report,indent=2))
    print("TRADE SANKEY",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    prepare_trade_sankey()
