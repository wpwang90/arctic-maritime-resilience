"""Separate admission-rule sensitivity from candidate-search sensitivity."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT / "44_CMIP6_Monthly_SSP_Audit"
sys.path.insert(0, str(ROOT))
import repair_cmip6_monthly_routes as routing

RESULTS = HERE / "results"
ROUTES = ["NWP", "NEP", "NSR", "TSR"]
THRESHOLDS = [30, 50, 70, 90]
REQUIRED = {t:60*t//100 for t in THRESHOLDS}
SCREENS = ["ice_capable", "low_ice_sensitivity", "observed_thickness_sensitivity"]
KEYS = ["period", "experiment", "screen", "route", "month"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    source_paths = [PREVIOUS / "results/monthly_route_support.csv", PREVIOUS / "results/conservative_monthly_route_support.csv",
                    ROOT / "37_FiveClosures_BoxScatter/figures/MainFig2.pdf"]
    protected = {str(p):digest(p) for p in source_paths}
    summary = pd.concat([pd.read_csv(p) for p in source_paths[:2]], ignore_index=True)
    assert len(summary) == 576 and not summary.duplicated(KEYS).any()
    fixed = []
    for threshold in THRESHOLDS:
        item = summary.copy()
        item["threshold_pct"] = threshold
        item["required_model_years"] = REQUIRED[threshold]
        item["admitted"] = item.common_candidate_exists & item.supported_model_years.ge(REQUIRED[threshold])
        item["search_node_threshold_pct"] = 50
        fixed.append(item)
    fixed = pd.concat(fixed, ignore_index=True)
    fixed.to_csv(RESULTS / "fixed_candidate_admission.csv", index=False)
    candidates, path_rows, member_rows, pools = [], [], [], []
    for period in [2050, 2100]:
        for ssp in ["ssp245", "ssp585"]:
            for screen in SCREENS:
                for route in ROUTES:
                    grid = dict(np.load(routing.OUT / f"geometry_{route}.npz"))
                    n = len(grid["points"])
                    parts, knowns, labels = [], [], []
                    for model in routing.MODELS:
                        with np.load(PREVIOUS / "cache" / f"{model}_{ssp}_{period}.npz") as src:
                            parts.append(np.unpackbits(src[route+"_"+screen], axis=1, count=n).astype(bool))
                            known_key = route+"_"+screen+"_known"
                            if known_key not in src:
                                known_key = route+"_known"
                            knowns.append(np.unpackbits(src[known_key], axis=1, count=n).astype(bool))
                            labels.extend((model,int(y),int(m)) for y,m in zip(src["year"],src["month"]))
                    values, known = np.concatenate(parts), np.concatenate(knowns)
                    del parts, knowns
                    for month in range(1, 13):
                        ix = np.array([i for i,(_,_,m) in enumerate(labels) if m == month])
                        v, k = values[ix], known[ix]
                        assert len(ix) == 60
                        counts = v.sum(axis=0)
                        probability = counts/60
                        metadata = dict(period=period, experiment=ssp, screen=screen, route=route, month=month)
                        candidate_set = []
                        for threshold in THRESHOLDS:
                            path = routing.solve_path(grid, counts >= REQUIRED[threshold], probability)
                            supported = v[:,path].all(axis=1) if path else np.zeros(60, bool)
                            coverage = k[:,path].all(axis=1) if path else np.zeros(60, bool)
                            if path:
                                xy = grid["points"][path]
                                km = float(routing.haversine_vec(xy[:-1,0],xy[:-1,1],xy[1:,0],xy[1:,1]).sum())
                                path_id = hashlib.sha256(np.array(path,dtype=np.int32).tobytes()).hexdigest()[:20]
                                candidate_set.append(dict(search_threshold=threshold, nodes=path, support=int(supported.sum()),
                                                          path_km=km, path_id=path_id))
                                for order,node in enumerate(path):
                                    lon,lat = grid["points"][node]
                                    path_rows.append(dict(metadata, search_threshold_pct=threshold, order=order,node=int(node),
                                                          lon=lon,lat=lat,path_id=path_id))
                            else:
                                km, path_id = np.nan, ""
                            candidates.append(dict(metadata, threshold_pct=threshold, required_model_years=REQUIRED[threshold],
                                candidate_exists=bool(path), supported_model_years=int(supported.sum()),
                                common_path_support_pct=100*float(supported.mean()), known_model_years=int(coverage.sum()),
                                admitted=bool(path) and supported.sum()>=REQUIRED[threshold], candidate_path_km=km,path_id=path_id))
                            for i,global_i in enumerate(ix):
                                model,year,_ = labels[global_i]
                                member_rows.append(dict(metadata, search_threshold_pct=threshold,model=model,year=year,
                                    candidate_exists=bool(path),path_known=bool(coverage[i]),path_passable=bool(supported[i])))
                        unique = {c["path_id"]:c for c in candidate_set}
                        for threshold in THRESHOLDS:
                            qualifying = [c for c in unique.values() if c["support"] >= REQUIRED[threshold]]
                            best = min(qualifying,key=lambda c:(c["path_km"],-c["support"])) if qualifying else None
                            pools.append(dict(metadata,threshold_pct=threshold,required_model_years=REQUIRED[threshold],
                                admitted=bool(best),unique_candidates=len(unique),
                                maximum_supported_model_years=max((c["support"] for c in unique.values()),default=0),
                                selected_support_pct=100*best["support"]/60 if best else np.nan,
                                selected_path_km=best["path_km"] if best else np.nan,
                                selected_search_threshold_pct=best["search_threshold"] if best else np.nan,
                                selected_path_id=best["path_id"] if best else ""))
                    print("Searched",period,ssp,screen,route,flush=True)
    rerouted = pd.DataFrame(candidates)
    pooled = pd.DataFrame(pools)
    paths = pd.DataFrame(path_rows)
    members = pd.DataFrame(member_rows)
    rerouted.to_csv(RESULTS / "rerouted_candidate_admission.csv",index=False)
    pooled.to_csv(RESULTS / "pooled_candidate_admission.csv",index=False)
    paths.to_parquet(RESULTS / "threshold_candidate_paths.parquet",index=False)
    members.to_csv(RESULTS / "threshold_model_year_evidence.csv",index=False)
    summaries = []
    for method,frame in [("fixed",fixed),("rerouted",rerouted),("candidate_pool",pooled)]:
        grouped = frame.groupby(KEYS[:-1]+["threshold_pct"]).agg(admitted_months=("admitted","sum"))
        month_list = frame[frame.admitted].groupby(KEYS[:-1]+["threshold_pct"]).month.apply(lambda s:",".join(map(str,sorted(s))))
        grouped["admitted_calendar_months"] = month_list.reindex(grouped.index).fillna("")
        grouped = grouped.reset_index()
        grouped["method"] = method
        summaries.append(grouped)
    annual = pd.concat(summaries,ignore_index=True)
    annual.to_csv(RESULTS / "annual_admission_summary.csv",index=False)
    merged = fixed.merge(pooled, on=KEYS+["threshold_pct"],suffixes=("_fixed","_pool"),validate="one_to_one")
    assert not (merged.admitted_fixed & ~merged.admitted_pool).any()
    merged["pool_adds_admission"] = merged.admitted_pool & ~merged.admitted_fixed
    merged[merged.pool_adds_admission].to_csv(RESULTS / "additional_months_found_by_candidate_search.csv",index=False)
    fifty = rerouted[rerouted.threshold_pct.eq(50)].merge(summary,on=KEYS,suffixes=("_new","_old"),validate="one_to_one")
    assert (fifty.supported_model_years_new == fifty.supported_model_years_old).all()
    assert (fifty.admitted_new == fifty.admitted_old).all()
    for name, frame in [("fixed",fixed),("candidate_pool",pooled)]:
        nested = frame.pivot(index=KEYS,columns="threshold_pct",values="admitted").astype(int)
        assert (np.diff(nested[THRESHOLDS].values,axis=1) <= 0).all(),name
    assert len(rerouted)==len(pooled)==len(fixed)==2304 and len(members)==138240
    # Single heuristic searches can choose different paths; record rather than hide inversions.
    rr = rerouted.pivot(index=KEYS,columns="threshold_pct",values="admitted").astype(int)
    inversions = rr[(np.diff(rr[THRESHOLDS].values,axis=1)>0).any(axis=1)]
    inversions.to_csv(RESULTS / "single_search_nonmonotonic_cases.csv")
    assert protected == {p:digest(p) for p in protected}
    report = dict(status="computed_pending_geometry_validation",threshold_percentages=THRESHOLDS,
        required_model_years=REQUIRED,models=routing.MODELS,model_years_per_month=60,
        climate_scenarios=["ssp245","ssp585"],windows={"2050":[2041,2050],"2100":[2091,2100]},
        screens=SCREENS,physical_route_month_cases=576,decision_rows_per_method=2304,
        individual_candidate_evaluations=len(members),searched_candidates=int(rerouted.candidate_exists.sum()),
        exact_50pct_reproduction_cases=len(fifty),fixed_and_pool_admission_nested=True,
        pooled_additional_admissions=int(merged.pool_adds_admission.sum()),single_search_inversion_cases=len(inversions),
        protected_sha256=protected,
        interpretation="Sensitivity of physical core admission only. No new probability calibration or network/economic recalculation.",
        pool_definition="Union of up to four candidate paths searched at 30/50/70/90 percent node support; shortest qualifying physical path selected. Finite search, not exhaustive feasibility.")
    (RESULTS / "calculation_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "protected_sha256"},indent=2),flush=True)
    print(annual.query("screen == 'ice_capable' and method == 'fixed'").pivot(index=["period","experiment","route"],columns="threshold_pct",values="admitted_months").to_string(),flush=True)


if __name__ == "__main__":
    main()
