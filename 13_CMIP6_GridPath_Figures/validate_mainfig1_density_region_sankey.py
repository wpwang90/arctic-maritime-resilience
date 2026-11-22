"""Focused, independently regrouped checks for the current Figure 1a/c revision."""
import json

import numpy as np
import pandas as pd

from prepare_mainfig1_density_region_sankey import OUT, ROOT, NONE, allocate_gate_weights
from rebuild_mainfig1_latest_network import sha


def main():
    grid=np.load(OUT/"panel_a_file_month_support_1deg.npz")
    raw=pd.read_csv(ROOT/"data/ethan_representative_ais_grid_density_1deg.csv")
    expected=np.zeros((180,360),dtype=np.int64)
    np.add.at(expected,(raw.lat_index,raw.lon_index),raw.trajectory_support.to_numpy())
    np.testing.assert_array_equal(grid["S_g"],expected)
    np.testing.assert_allclose(grid["D_g"],np.log10(1+expected))
    assert int(grid["resolution_degrees"])==1
    map_checks=[]
    for path in sorted((OUT/"maps").glob("*_deep_support.npz")):
        local=np.load(path)
        categories=np.load(path.with_name(path.name.replace("_deep_support","")))
        np.testing.assert_array_equal(local["deep_water_mask"],categories["display_class"]==1)
        mask=local["deep_water_mask"]
        np.testing.assert_allclose(local["D_g"][mask],np.log10(1+local["S_g"][mask]))
        assert np.isnan(local["D_g"][~mask]).all()
        colour=json.loads(path.with_name(path.name.replace("_deep_support.npz","_colour_audit.json")).read_text())
        assert colour["non_deep_RGBA_unchanged"]
        map_checks.append(path.name)
    assert len(map_checks)==4

    baseline=pd.read_parquet(OUT/"baseline_all.parquet").set_index("od_id")
    sample=pd.read_parquet(OUT/"panel_c_sample_selection.parquet").set_index("od_id")
    included=sample[sample.selection_status.eq("included")]
    assert sample.index.is_unique and included.index.is_unique
    assert included.origin_region.ne(included.dest_region).all()
    assert included[["origin_region","dest_region"]].notna().all().all()
    assert not included.origin_canonical_iso3.eq("TWN").any()
    assert not included.dest_canonical_iso3.eq("TWN").any()
    assert np.isfinite(baseline.activity_weight).all() and baseline.activity_weight.gt(0).all()
    registry=pd.read_csv(ROOT/"data/chokepoints24/registry.csv")
    keys=registry.key.tolist()
    assert len(keys)==24
    selected=baseline.loc[included.index]
    selected=selected.assign(trade_weight_usd=included.trade_weight_usd)
    assert selected.trade_weight_usd.gt(0).all()
    assert selected.baseline_mask.between(0,(1<<24)-1).all()
    allocated=pd.read_parquet(OUT/"panel_c_allocated_od_weights.parquet")
    assert not allocated.duplicated(["od_id","gate"]).any()
    np.testing.assert_allclose(allocated.groupby("od_id").allocated_weight.sum().sort_index(),
        selected.trade_weight_usd.sort_index(),rtol=1e-12)
    # Reconstruct node totals with a separate bit-count allocation, not the helper under test.
    expected_gates={key:0.0 for key in keys+[NONE]}
    for row in selected.itertuples():
        indices=[i for i in range(24) if int(row.baseline_mask)&(1<<i)]
        if not indices:
            expected_gates[NONE]+=row.trade_weight_usd
        for i in indices:
            expected_gates[keys[i]]+=row.trade_weight_usd/len(indices)
    gates=pd.read_csv(OUT/"panel_c_gate_shares.csv").set_index("gate")
    assert set(gates.index)==set(keys+[NONE])
    for key,value in expected_gates.items():
        np.testing.assert_allclose(gates.loc[key,"allocated_weight"],value,rtol=1e-12)
    np.testing.assert_allclose(gates.percent.sum(),100,rtol=1e-12)
    for side in ["origin","dest"]:
        np.testing.assert_allclose(allocated.groupby(side).allocated_weight.sum().sort_index(),
            included.groupby(side+"_region").trade_weight_usd.sum().sort_index())
    pairs=pd.read_csv(OUT/"panel_c_country_pair_trade_coverage.csv")
    pair_keys=["origin_iso3","dest_iso3"]
    allocated_pairs=included.rename(columns={"origin_canonical_iso3":"origin_iso3","dest_canonical_iso3":"dest_iso3"})
    np.testing.assert_allclose(allocated_pairs.groupby(pair_keys).trade_weight_usd.sum().sort_index(),
        pairs[pairs.allocation_status.eq("allocated")].set_index(pair_keys).maritime_proxy_2024_usd.sort_index(),rtol=1e-12)
    np.testing.assert_allclose(included.trade_weight_usd,
        included.baci_2024_usd*included.sea_fraction_2021_used*included.activity_weight/included.pair_activity,rtol=1e-12)
    assert included.trade_year.eq(2024).all()
    assert included.modal_adjustment_applied.all() and included.modal_share_year.eq(2021).all()
    assert "sea_fraction_2015" not in sample.columns
    modal=pd.read_parquet(ROOT/"data/mainfig1_trade_2024/recent_maritime_shares/unctad_bilateral_sea_shares_2016_2021.parquet")
    valid=modal[modal.year.eq(2021)&modal.status.eq("valid")]
    median=valid.sea_fraction.median()
    np.testing.assert_array_equal(pairs.sea_share_imputed,pairs.sea_fraction_2021_source.isna())
    np.testing.assert_allclose(pairs.sea_fraction_2021_used,pairs.sea_fraction_2021_source.fillna(median))
    explicit_zero=pairs.sea_fraction_2021_source.eq(0)
    assert pairs.loc[explicit_zero,"sea_fraction_2021_used"].eq(0).all()
    assert not pairs.loc[explicit_zero,"sea_share_imputed"].any()
    inferred=included.sea_share_imputed.eq(True)
    np.testing.assert_allclose(included.loc[inferred,"sea_fraction_2021_used"],median)
    assert included.loc[inferred,"sea_fraction_2021_source"].isna().all()
    np.testing.assert_allclose(included.loc[~inferred,"sea_fraction_2021_used"],
        included.loc[~inferred,"sea_fraction_2021_source"])
    assert included.sea_fraction_2021_used.between(0,1).all()
    joined=allocated_pairs[allocated_pairs.sea_share_imputed.eq(False)].merge(valid[pair_keys+["sea_fraction"]],
        on=pair_keys,validate="many_to_one")
    assert len(joined)==int((~inferred).sum())
    np.testing.assert_allclose(joined.sea_fraction_2021_used,joined.sea_fraction)
    all_pairs=pairs[pairs.allocation_status.eq("allocated")]
    imputation_audit=json.loads((OUT/"panel_c_modal_share_audit.json").read_text())
    assert int(all_pairs.sea_share_imputed.sum())==imputation_audit["included_imputed_pairs"]
    sensitivity=pd.read_csv(OUT/"panel_c_modal_imputation_sensitivity.csv")
    np.testing.assert_allclose(sensitivity.groupby("scenario").percent.sum(),100)
    central=sensitivity[sensitivity.scenario.eq("missing_median")].set_index("gate")
    np.testing.assert_allclose(central.percent.sort_index(),gates.percent.sort_index())
    np.testing.assert_allclose(allocated_pairs.groupby(pair_keys).within_pair_activity_share.sum(),1,rtol=1e-12)
    mask=(1<<keys.index("Suez"))|(1<<keys.index("Bab_el_Mandeb"))
    assert [w for _,w in allocate_gate_weights(100,mask,keys)]==[50,50]
    assert allocate_gate_weights(100,0,keys)==[(NONE,100)]
    audit=json.loads((OUT/"panel_c_region_sankey_audit.json").read_text())
    assert audit["trade_year"]==audit["ais_year"]==2024
    assert audit["modal_share_year"]==2021 and audit["modal_adjustment_applied"]
    assert sha(OUT/"baseline_all.parquet")==audit["input_baseline_sha256"]
    assert sha(OUT/"panel_b_exposure.csv")==audit["panel_b_sha256"]
    layouts=list((OUT/"figures").glob("MainFig1*layout_audit.json"))
    assert len(layouts)==3
    for path in layouts:
        assert not json.loads(path.read_text())["texts_outside_canvas"],path
    main_layout=json.loads((OUT/"figures/MainFig1_AISfirst_24Chokepoints_layout_audit.json").read_text())
    assert main_layout["main_rows_same_left_and_right"]
    report=dict(status="PASS",support_regrouping="PASS",log_transform="PASS",
        deep_mask_and_unchanged_palette_checks=map_checks,allocated_gate_totals="PASS",
        record_and_endpoint_conservation="PASS",annual_bilateral_trade_conservation="PASS",
        trade_value_allocation_formula="PASS",baseline_and_panel_b_unchanged="PASS",
        trade_and_AIS_2024_alignment="PASS",modal_share_2021_provenance="PASS",
        median_imputation_explicit_and_recomputed="PASS",imputed_country_pairs=int(all_pairs.sea_share_imputed.sum()),
        explicit_zero_shares_preserved="PASS",
        median_used=float(median),quartile_sensitivity_conservation="PASS",obsolete_2015_share_not_used="PASS",
        main_row_alignment="PASS",
        layout_text_inside_canvas="PASS",included_records=len(included),
        selected_trade_weight_usd=float(included.trade_weight_usd.sum()),
        Suez_percent=float(gates.loc["Suez","percent"]),
        interpretation="Numerical and rendering checks, not external navigation or economic validation")
    (OUT/"density_region_sankey_validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
