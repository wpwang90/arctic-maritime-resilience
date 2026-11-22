"""Conservative rerouting without filling missing thickness at positive SIC."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from compute_seasonality import (CACHE, RESULTS, INVENTORY, PERIODS, SSPS, ROUTES,
                                routing, normalized_concentration, solve_case)

SCREEN = "observed_thickness_sensitivity"
grids = {r: dict(np.load(routing.OUT / f"geometry_{r}.npz")) for r in ROUTES}
for period, (first, last) in PERIODS.items():
    for experiment in SSPS:
        for source in pd.read_csv(INVENTORY).query("experiment == @experiment").itertuples(index=False):
            path = CACHE / f"{source.model}_{experiment}_{period}.npz"
            with np.load(path) as z:
                content = {k:z[k] for k in z.files}
            if ROUTES[0]+"_"+SCREEN in content:
                continue
            s, h = routing.open_data(source.siconc), routing.open_data(source.thickness)
            sv = s.siconc.sel(time=slice(str(first), str(last)))
            hv = h[source.thickness_variable].sel(time=slice(str(first), str(last)))
            sic = normalized_concentration(sv.to_numpy().reshape(120, -1), sv.attrs["units"])
            raw_h = hv.to_numpy().reshape(120, -1)
            thick = raw_h / (900*np.maximum(sic/100, .01)) if source.thickness_variable == "simass" else raw_h
            original_thick = np.where(np.isfinite(sic) & (sic >= 0) & (sic <= 15) & ~np.isfinite(thick), 0, thick)
            conservative_thick = np.where(sic == 0, 0, thick)
            lon, lat = routing.coordinate_arrays(s)
            ocean = np.flatnonzero((np.isfinite(sic) & np.isfinite(original_thick)).any(axis=0) & np.isfinite(lon) & (lat >= 50))
            tree = cKDTree(routing.xyz(np.c_[lon[ocean], lat[ocean]]))
            for route, grid in grids.items():
                chord, near = tree.query(routing.xyz(grid["points"]))
                col = ocean[near]
                distance = 2*routing.EARTH_KM*np.arcsin(np.minimum(chord/2, 1))
                known = (np.isfinite(sic[:, col]) & np.isfinite(conservative_thick[:, col]) &
                         (sic[:, col] >= 0) & (sic[:, col] <= 100) & (conservative_thick[:, col] >= 0) &
                         (distance <= 150)[None, :])
                passed = known & (sic[:, col] <= 80) & (conservative_thick[:, col] <= 1.2)
                original = np.unpackbits(content[route+"_ice_capable"], axis=1, count=len(col)).astype(bool)
                # Exactly zero SIC means no ice thickness is required, not an imputation at nonzero SIC.
                assert not (passed & ~original).any()
                content[route+"_"+SCREEN] = np.packbits(passed, axis=1)
                content[route+"_"+SCREEN+"_known"] = np.packbits(known, axis=1)
            np.savez_compressed(path, **content)
            s.close(); h.close()
            print("Mapped conservative", source.model, experiment, period, flush=True)

members, summary, paths = [], [], []
for period in PERIODS:
    for experiment in SSPS:
        for route, grid in grids.items():
            r, s, p = solve_case(experiment, period, SCREEN, route, grid)
            members += r; summary += s; paths += p
pd.DataFrame(members).to_csv(RESULTS / "conservative_model_year_month_feasibility.csv", index=False)
pd.DataFrame(summary).to_csv(RESULTS / "conservative_monthly_route_support.csv", index=False)
pd.DataFrame(paths).to_parquet(RESULTS / "conservative_candidate_core_paths.parquet", index=False)
print(pd.DataFrame(summary).pivot(index=["period", "experiment", "route"], columns="month", values="common_path_support_pct").round(1), flush=True)
