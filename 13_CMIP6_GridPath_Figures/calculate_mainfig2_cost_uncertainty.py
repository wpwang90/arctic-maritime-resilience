"""Fixed-cost uncertainty from real climate members and OD-cluster resampling.

Boxes show joint scenario/resampling spread, not fitted real-price uncertainty
and not confidence intervals for the global shipping population.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from rebuild_mainfig2_latest_network import ROOT, OUT, ROUTES, SHIPS, sha
from recompute_physical_arctic_embedding import core_ice_masks
from repair_cmip6_monthly_routes import MODELS

SEED = 20261002
REPLICATES_PER_MEMBER = 100
PREMIUMS = np.arange(0, 101, 10, dtype=float)


def climate_masks(data):
    result = np.zeros((len(data), len(ROUTES)), dtype=np.uint64)
    cores = core_ice_masks()
    for (ship, month), group in data.groupby(["shiptype", "month"]):
        cat = np.load(OUT/"routing"/f"{ship}_{month:02d}_paths.npz")
        for ri, route in enumerate(ROUTES):
            if (route, int(month)) not in cores:
                continue
            valid = np.isfinite(group[f"{route}_km"])
            g = group[valid]
            p = g[f"{route}_entry"].to_numpy(int)
            q = g[f"{route}_exit"].to_numpy(int)
            result[g.index, ri] = cat["model_year_mask"][p] & cat["model_year_mask"][q] & cores[(route, int(month))]
    return result


def aggregate_selected(cluster, weight, lengths, without, ncluster):
    result = np.empty((len(PREMIUMS), ncluster), float)
    for ci, premium in enumerate(PREMIUMS):
        selected = np.isfinite(lengths) & ((1+premium/100)*lengths < without-1e-7)
        result[ci] = np.bincount(cluster, weights=weight*selected, minlength=ncluster)
    return result


def main():
    data = pd.read_parquet(OUT/"all_candidates_and_choices.parquet")
    data = data[data.scenario_index > 0].reset_index(drop=True)
    masks = climate_masks(data)
    draws, member_rows, summaries, cluster_rows = [], [], [], []
    rng = np.random.default_rng(SEED)
    for ship in SHIPS:
        g = data[data.shiptype == ship]
        weight = g.activity_weight.to_numpy(float)
        # One undirected port-pair cluster keeps directions, months and all
        # closure experiments together. Repeated exposure is not an IID sample.
        pairs = [tuple(sorted((a,b))) for a,b in zip(g.origin_port_id, g.dest_port_id)]
        unique = sorted(set(pairs))
        mapping = {pair:i for i,pair in enumerate(unique)}
        cluster = np.array([mapping[pair] for pair in pairs], int)
        count = len(unique)
        den = np.bincount(cluster, weights=weight, minlength=count)
        distances = g[[f"{r}_km" for r in ROUTES]].to_numpy(float)
        without = g.without_km.to_numpy(float)
        member_masks = masks[g.index]
        deterministic_num = aggregate_selected(cluster, weight, distances.min(axis=1), without, count)
        deterministic = 100*deterministic_num.sum(axis=1)/den.sum()
        joint = np.empty((60*REPLICATES_PER_MEMBER, len(PREMIUMS)))
        od_only = np.empty_like(joint)
        climate_only = np.empty((60, len(PREMIUMS)))
        for member in range(60):
            usable = ((member_masks >> np.uint64(member)) & np.uint64(1)).astype(bool)
            best = np.where(usable, distances, np.inf).min(axis=1)
            numerator = aggregate_selected(cluster, weight, best, without, count)
            climate_only[member] = 100*numerator.sum(axis=1)/den.sum()
            resample = rng.multinomial(count, np.full(count,1/count), size=REPLICATES_PER_MEMBER)
            # Direct reductions avoid platform BLAS floating-status artefacts.
            draw_den = np.einsum("ij,j->i", resample, den, optimize=False)
            assert (draw_den > 0).all()
            start = member*REPLICATES_PER_MEMBER
            end = start+REPLICATES_PER_MEMBER
            joint[start:end] = 100*np.einsum("ij,kj->ik", resample, numerator, optimize=False)/draw_den[:,None]
            od_only[start:end] = 100*np.einsum("ij,kj->ik", resample, deterministic_num, optimize=False)/draw_den[:,None]
            for ci, premium in enumerate(PREMIUMS):
                member_rows.append(dict(shiptype=ship, model=MODELS[member//10], year=2091+member%10,
                    extra_unit_distance_cost_pct=premium, selected_pct=climate_only[member,ci]))
                draws.append(pd.DataFrame(dict(shiptype=ship, model=MODELS[member//10], year=2091+member%10,
                    member_index=member, draw_id=np.arange(start,end), extra_unit_distance_cost_pct=premium,
                    joint_selected_pct=joint[start:end,ci], od_bootstrap_only_pct=od_only[start:end,ci],
                    resampled_exposed_weight=draw_den)))
        assert (np.diff(joint,axis=1) <= 1e-8).all()
        assert (np.diff(climate_only,axis=1) <= 1e-8).all()
        assert np.isfinite(joint).all() and (joint>=0).all() and (joint<=100).all()
        assert np.all(climate_only <= deterministic[None,:]+1e-8)
        for ci,premium in enumerate(PREMIUMS):
            record=dict(shiptype=ship,extra_unit_distance_cost_pct=premium,n_draws=len(joint),
                n_model_years=60,n_unique_models=6,n_od_clusters=count,deterministic_screen_pct=deterministic[ci])
            for name,arr in (("joint",joint),("climate_only",climate_only),("od_only",od_only)):
                for quantile,value in zip((5,25,50,75,95),np.percentile(arr[:,ci],[5,25,50,75,95])):
                    record[f"{name}_p{quantile}"]=float(value)
            summaries.append(record)
        for index,pair in enumerate(unique):
            cluster_rows.append(dict(shiptype=ship,port_a=pair[0],port_b=pair[1],exposed_activity_weight=den[index],
                od_scenario_rows=int((cluster==index).sum())))
        print(ship, "clusters",count,"joint draws",len(joint),"zero-cost median",np.median(joint[:,0]),flush=True)
    pd.concat(draws,ignore_index=True).to_parquet(OUT/"panel_d_uncertainty_draws.parquet",index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/"panel_d_uncertainty_summary.csv",index=False)
    pd.DataFrame(member_rows).to_csv(OUT/"panel_d_climate_member_sensitivity.csv",index=False)
    pd.DataFrame(cluster_rows).to_csv(OUT/"panel_d_bootstrap_clusters.csv",index=False)
    old = pd.read_csv(OUT/"panel_d_break_even_sensitivity.csv")
    match = summary.merge(old[old.weight.eq("activity")],on=["shiptype","extra_unit_distance_cost_pct"],validate="one_to_one")
    assert np.allclose(match.deterministic_screen_pct,match.selected_pct,atol=1e-8)
    audit = dict(status="PASS",seed=SEED,cost_levels=PREMIUMS.tolist(),shiptypes=SHIPS,
        draws_per_shiptype_cost=60*REPLICATES_PER_MEMBER,climate_members=60,unique_climate_models=6,
        bootstrap_draws_per_member=REPLICATES_PER_MEMBER,climate_members_balanced=True,
        clustering="unordered canonical physical port pair, retaining both directions, all months and closure scenarios",
        fixed_cost_not_sampled=True,cost_unit="additional complete-voyage unit-distance cost fraction",
        routes_reselected_for_each_member=True,deterministic_source_result_reproduced=True,
        every_draw_monotone=True,boxes="25th-75th percentiles; median; 5th-95th percentile whiskers; outliers omitted",
        interpretation="joint CMIP6 member and OD-cluster resampling spread, not a calibrated population confidence interval",
        limitations=["Conditional on the existing >=30/60 fixed route candidate bank; no member-specific re-routing of access geometry",
            "Climate models are equally weighted plausible scenarios, not independent random observations",
            "Missing AIS coverage, common network error, legal/draft constraints and fuel/tariff calibration are not captured",
            "Only SSP585 and 2091-2100; observed 2024 activity held as the demand reference"],
        provenance={"all_candidates_sha256":sha(OUT/"all_candidates_and_choices.parquet"),
            "monthly_feasibility_sha256":sha(ROOT/"data/cmip6_monthly_repair/model_year_month_feasibility.csv"),
            "script_sha256":sha(__file__)})
    (OUT/"panel_d_uncertainty_validation.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":
    main()
