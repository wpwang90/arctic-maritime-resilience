#!/usr/bin/env python3
"""Embed four CMIP6 Arctic routes in the 2024 observed AIS grid network.

Traditional access and egress legs are solved on vessel-type/month-specific
AIS grid graphs.  Edge costs retain physical distance while mildly favouring
edges supported by more representative trajectories.  Dijkstra paths are
loop-free by construction.  The Arctic middle segment is taken verbatim from
the existing monthly CMIP6 PC6 grid-path output for NWP, NEP, NSR and TSR.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SUMMARY_PARQUET = DATA / "ethan_representative_trajectory_summary.parquet"
EDGE_PARQUET = DATA / "ethan_representative_ais_grid_edges_1deg.parquet"
CMIP_PATHS = DATA / "monthly_cmip6_four_route_grid_paths.csv"
CMIP_SUMMARY = DATA / "monthly_cmip6_four_route_summary.csv"
CMIP_FEASIBLE_MONTHS = DATA / "cmip6_grid_feasible_months.csv"
OECD_MARGINS = DATA / "oecd_itic_route_country_pair_margins_2019plus.csv"

EARTH_RADIUS_KM = 6371.0088
ANALYSIS_MONTHS = [7, 8, 9, 10, 11]
ROUTES = ["NWP", "NEP", "NSR", "TSR"]

# Dimensionless cost parameters.  These are model assumptions, not observed
# data, and are retained as separate columns for later Monte Carlo sampling.
SHIP_ICE_FACTOR = {"container": 1.00, "bulk": 1.18, "tanker": 1.25, "lng": 0.85}
ROUTE_RISK_PREMIUM = {"NWP": 0.040, "NEP": 0.025, "NSR": 0.020, "TSR": 0.050}
ROUTE_ESCORT_TOLL = {"NWP": 0.025, "NEP": 0.030, "NSR": 0.035, "TSR": 0.015}

CHOKEPOINTS = {
    "Bering": (-168.9, 65.8, 2.2),
    "Panama": (-79.6, 9.1, 1.7),
    "Suez": (32.5, 30.3, 1.6),
    "Bab_el_Mandeb": (43.3, 12.6, 1.7),
    "Hormuz": (56.3, 26.5, 1.8),
    "Malacca": (101.2, 2.8, 2.3),
    "Gibraltar": (-5.6, 35.9, 1.6),
    "Dover": (1.5, 51.0, 1.4),
    "Danish": (12.0, 56.0, 1.8),
    "Turkish": (29.0, 41.1, 1.5),
    "Cape_Good_Hope": (18.5, -34.4, 2.5),
}


def haversine_vec(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return float(haversine_vec(np.array([lon1]), np.array([lat1]), np.array([lon2]), np.array([lat2]))[0])


def node_from_lonlat(lon: float, lat: float) -> int:
    x = int(np.clip(np.floor((lon + 180.0) % 360.0), 0, 359))
    y = int(np.clip(np.floor(np.clip(lat, -89.999999, 89.999999) + 90.0), 0, 179))
    return y * 360 + x


def node_lonlat(node: int) -> tuple[float, float]:
    y, x = divmod(int(node), 360)
    return -179.5 + x, -89.5 + y


def unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    return np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )


class AISGraph:
    def __init__(self, edges: pd.DataFrame):
        n1 = edges["y1"].to_numpy(np.int32) * 360 + edges["x1"].to_numpy(np.int32)
        n2 = edges["y2"].to_numpy(np.int32) * 360 + edges["x2"].to_numpy(np.int32)
        lon1 = -179.5 + edges["x1"].to_numpy(float)
        lat1 = -89.5 + edges["y1"].to_numpy(float)
        lon2 = -179.5 + edges["x2"].to_numpy(float)
        lat2 = -89.5 + edges["y2"].to_numpy(float)
        physical = haversine_vec(lon1, lat1, lon2, lat2)
        support = edges["trajectory_support"].to_numpy(float)
        # A 0.18 support penalty selects heavily used cells without allowing a
        # long popular detour to overwhelm physical distance.
        modal_cost = physical * (1.0 + 0.18 / np.log1p(np.maximum(support, 1.0)))
        row = np.concatenate([n1, n2])
        col = np.concatenate([n2, n1])
        weight = np.concatenate([modal_cost, modal_cost])
        self.graph: csr_matrix = coo_matrix((weight, (row, col)), shape=(64800, 64800)).tocsr()
        self.graph.sum_duplicates()
        self.active = np.unique(np.concatenate([n1, n2]))
        active_xy = np.asarray([node_lonlat(n) for n in self.active], dtype=float)
        self.tree = cKDTree(unit_xyz(active_xy[:, 0], active_xy[:, 1]))
        self._source_cache: dict[int, np.ndarray] = {}

    def snap(self, lon: float, lat: float) -> tuple[int, float]:
        raw = node_from_lonlat(lon, lat)
        pos = np.searchsorted(self.active, raw)
        if pos < len(self.active) and self.active[pos] == raw:
            return raw, haversine(lon, lat, *node_lonlat(raw))
        _, index = self.tree.query(unit_xyz(np.asarray([lon]), np.asarray([lat]))[0], k=1)
        node = int(self.active[int(index)])
        return node, haversine(lon, lat, *node_lonlat(node))

    def distances_from(self, node: int) -> np.ndarray:
        node = int(node)
        if node not in self._source_cache:
            self._source_cache[node] = dijkstra(self.graph, directed=False, indices=node, return_predecessors=False)
        return self._source_cache[node]

    def distance(self, source: int, target: int) -> float:
        return float(self.distances_from(source)[int(target)])

    def path(self, source: int, target: int) -> list[int]:
        dist, pred = dijkstra(self.graph, directed=False, indices=int(source), return_predecessors=True)
        if not np.isfinite(dist[int(target)]):
            return []
        path = [int(target)]
        cursor = int(target)
        while cursor != int(source):
            cursor = int(pred[cursor])
            if cursor < 0:
                return []
            path.append(cursor)
        path.reverse()
        return path


@lru_cache(maxsize=64)
def load_graph(shiptype: str, month: int) -> AISGraph:
    con = duckdb.connect()
    edges = con.execute(
        """
        SELECT x1,y1,x2,y2,trajectory_support
        FROM read_parquet(?)
        WHERE shiptype=? AND month=?
        """,
        [str(EDGE_PARQUET), shiptype, int(month)],
    ).df()
    if edges.empty:
        raise RuntimeError(f"No AIS grid edges for {shiptype} month {month}")
    return AISGraph(edges)


def union_strings(values: pd.Series) -> str:
    tokens: set[str] = set()
    for value in values.dropna().astype(str):
        tokens.update(v for v in value.split(";") if v)
    return ";".join(sorted(tokens))


def representative_monthly_od() -> pd.DataFrame:
    con = duckdb.connect()
    summary = con.execute("SELECT * FROM read_parquet(?)", [str(SUMMARY_PARQUET)]).df()
    summary = summary[summary["start_month"].isin(ANALYSIS_MONTHS)].copy()
    summary = summary[summary["origin_iso3"].ne("TWN") & summary["dest_iso3"].ne("TWN")].copy()
    key = ["shiptype", "start_month", "origin_key", "dest_key"]
    weights = (
        summary.groupby(key, as_index=False)
        .agg(
            activity_weight=("activity_weight", "sum"),
            call_count=("call_count", "sum"),
            unique_vessels=("unique_vessels", "sum"),
            member_count=("member", "size"),
            all_observed_chokepoints=("chokepoints", union_strings),
        )
    )
    representatives = (
        summary.sort_values(
            ["direct_quality", "activity_weight", "raw_points"], ascending=[False, False, False]
        )
        .drop_duplicates(key)
        .drop(columns=["activity_weight", "call_count", "unique_vessels"], errors="ignore")
    )
    out = representatives.merge(weights, on=key, how="inner")
    out["od_month_id"] = (
        out["shiptype"]
        + "|"
        + out["start_month"].astype(str)
        + "|"
        + out["origin_key"]
        + "|"
        + out["dest_key"]
    )
    out["port_great_circle_km"] = haversine_vec(
        out["origin_port_lon"].to_numpy(float),
        out["origin_port_lat"].to_numpy(float),
        out["dest_port_lon"].to_numpy(float),
        out["dest_port_lat"].to_numpy(float),
    )
    return out


def margin_lookup() -> tuple[dict[tuple[str, str], float], float]:
    margins = pd.read_csv(OECD_MARGINS)
    lookup: dict[tuple[str, str], float] = {}
    for row in margins.itertuples(index=False):
        value = float(row.median_margin_pct)
        if np.isfinite(value):
            lookup[(str(row.importer_iso3), str(row.exporter_iso3))] = value
    fallback = float(pd.to_numeric(margins["median_margin_pct"], errors="coerce").median())
    return lookup, fallback


def chokepoints_on_nodes(nodes: list[int]) -> str:
    if not nodes:
        return ""
    pts = np.asarray([node_lonlat(n) for n in nodes], dtype=float)
    exposed: list[str] = []
    for name, (lon, lat, radius) in CHOKEPOINTS.items():
        dist = haversine_vec(pts[:, 0], pts[:, 1], np.full(len(pts), lon), np.full(len(pts), lat))
        if np.any(dist <= radius * 111.2):
            exposed.append(name)
    return ";".join(exposed)


def prepare_candidate_baselines(od: pd.DataFrame) -> pd.DataFrame:
    candidate = od[
        (od["origin_port_lat"] > 15.0)
        & (od["dest_port_lat"] > 15.0)
        & (od["port_great_circle_km"] >= 2000.0)
        & od["origin_iso3"].notna()
        & od["dest_iso3"].notna()
        & od["origin_iso3"].ne(od["dest_iso3"])
    ].copy()
    rows: list[dict] = []
    for (shiptype, month), group in candidate.groupby(["shiptype", "start_month"], sort=True):
        graph = load_graph(str(shiptype), int(month))
        for row in group.itertuples(index=False):
            source, source_snap = graph.snap(float(row.origin_port_lon), float(row.origin_port_lat))
            target, target_snap = graph.snap(float(row.dest_port_lon), float(row.dest_port_lat))
            if bool(row.direct_quality):
                observed = float(max(row.loop_erased_distance_km, row.port_great_circle_km * 1.02))
                source_name = "direct_AIS_loop_erased"
                exposure = str(row.all_observed_chokepoints or "")
                geometry = ""
            else:
                modal = graph.distance(source, target)
                if np.isfinite(modal):
                    observed = float(modal + source_snap + target_snap)
                    source_name = "AIS_grid_modal_reconstructed"
                    nodes = graph.path(source, target)
                    exposure = chokepoints_on_nodes(nodes)
                    geometry = json.dumps([node_lonlat(n) for n in nodes], separators=(",", ":"))
                else:
                    observed = float(max(row.port_great_circle_km * 1.18, min(row.loop_erased_distance_km, row.raw_valid_distance_km)))
                    source_name = "AIS_endpoint_detour_fallback"
                    exposure = str(row.all_observed_chokepoints or "")
                    geometry = ""
            rows.append(
                {
                    **row._asdict(),
                    "source_node": source,
                    "target_node": target,
                    "source_snap_km": source_snap,
                    "target_snap_km": target_snap,
                    "traditional_distance_km": observed,
                    "traditional_path_source": source_name,
                    "traditional_chokepoints": exposure,
                    "traditional_grid_geometry": geometry,
                }
            )
    return pd.DataFrame(rows)


def route_month_metadata() -> tuple[pd.DataFrame, dict[tuple[str, int], dict]]:
    paths = pd.read_csv(CMIP_PATHS)
    summary = pd.read_csv(CMIP_SUMMARY)
    feasible = pd.read_csv(CMIP_FEASIBLE_MONTHS)
    feasible = feasible[
        feasible["scenario"].eq("SSP5-8.5")
        & feasible["period"].eq(2100)
        & feasible["class"].eq("PC6")
    ].copy()
    feasible["grid_key"] = list(
        zip(feasible["lon"].round().astype(int), feasible["lat"].round().astype(int))
    )
    feasible_lookup = feasible.groupby("grid_key")["median_months"].median().to_dict()
    metadata: dict[tuple[str, int], dict] = {}
    for (route, month), group in paths.groupby(["route", "month"]):
        group = group.sort_values("order")
        rec = summary[(summary["route"] == route) & (summary["month"] == month)].iloc[0]
        mean_months = float(rec["path_mean_months"]) if pd.notna(rec["path_mean_months"]) else np.nan
        if not np.isfinite(mean_months):
            sampled = [
                feasible_lookup.get((round(float(x)), round(float(y))), np.nan)
                for x, y in group[["lon", "lat"]].to_numpy(float)
            ]
            mean_months = float(np.nanmean(sampled))
        continuous = "fallback" not in str(rec["path_source"]).lower() and np.isfinite(float(rec["path_mean_months"]))
        metadata[(str(route), int(month))] = {
            "gateway_a_lon": float(group.iloc[0]["lon"]),
            "gateway_a_lat": float(group.iloc[0]["lat"]),
            "gateway_b_lon": float(group.iloc[-1]["lon"]),
            "gateway_b_lat": float(group.iloc[-1]["lat"]),
            "arctic_middle_km": float(rec["path_km"]),
            "path_mean_months": mean_months,
            "threshold_used": float(rec["threshold_used"]),
            "path_source": str(rec["path_source"]),
            "cmip6_continuous": bool(continuous),
        }
    return paths, metadata


def build_route_choices(candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, route_meta = route_month_metadata()
    margins, fallback_margin = margin_lookup()
    records: list[dict] = []
    for (shiptype, month), group in candidate.groupby(["shiptype", "start_month"], sort=True):
        graph = load_graph(str(shiptype), int(month))
        gate_cache: dict[tuple[float, float], tuple[int, float, np.ndarray]] = {}
        for route in ROUTES:
            meta = route_meta[(route, int(month))]
            for side in ["a", "b"]:
                lon = meta[f"gateway_{side}_lon"]
                lat = meta[f"gateway_{side}_lat"]
                node, snap = graph.snap(lon, lat)
                gate_cache[(lon, lat)] = (node, snap, graph.distances_from(node))
        for row in group.itertuples(index=False):
            margin = margins.get((str(row.dest_iso3), str(row.origin_iso3)))
            margin_source = "OECD_ITIC_importer_exporter_2019plus"
            if margin is None:
                margin = margins.get((str(row.origin_iso3), str(row.dest_iso3)))
                margin_source = "OECD_ITIC_reverse_pair_2019plus"
            if margin is None:
                margin = fallback_margin
                margin_source = "OECD_ITIC_global_median_2019plus"
            baseline_cost = float(row.traditional_distance_km) * (1.0 + float(margin) / 100.0)
            for route in ROUTES:
                meta = route_meta[(route, int(month))]
                a = gate_cache[(meta["gateway_a_lon"], meta["gateway_a_lat"])]
                b = gate_cache[(meta["gateway_b_lon"], meta["gateway_b_lat"])]
                option_ab = a[2][int(row.source_node)] + b[2][int(row.target_node)] + a[1] + b[1]
                option_ba = b[2][int(row.source_node)] + a[2][int(row.target_node)] + a[1] + b[1]
                access = float(min(option_ab, option_ba))
                orientation = "A_to_B" if option_ab <= option_ba else "B_to_A"
                if not np.isfinite(access):
                    continue
                embedded_km = access + meta["arctic_middle_km"]
                ice_penalty = SHIP_ICE_FACTOR[str(shiptype)] * (
                    0.04 + 0.18 * (12.0 - meta["path_mean_months"]) / 12.0
                )
                risk = ROUTE_RISK_PREMIUM[route]
                escort_toll = ROUTE_ESCORT_TOLL[route]
                arctic_cost = embedded_km * (1.0 + float(margin) / 100.0 + ice_penalty + risk + escort_toll)
                records.append(
                    {
                        **row._asdict(),
                        "route": route,
                        "orientation": orientation,
                        **meta,
                        "ais_modal_access_km": access,
                        "arctic_embedded_distance_km": embedded_km,
                        "distance_saving_pct": 100.0 * (float(row.traditional_distance_km) - embedded_km) / float(row.traditional_distance_km),
                        "oecd_margin_pct": float(margin),
                        "oecd_margin_source": margin_source,
                        "ice_penalty_index": ice_penalty,
                        "route_risk_premium_index": risk,
                        "escort_toll_index": escort_toll,
                        "baseline_generalized_cost": baseline_cost,
                        "arctic_generalized_cost": arctic_cost,
                        "generalized_cost_saving_pct": 100.0 * (baseline_cost - arctic_cost) / baseline_cost,
                    }
                )
    alternatives = pd.DataFrame(records)
    alternatives["eligible_choice_cost"] = alternatives["arctic_generalized_cost"].where(
        alternatives["cmip6_continuous"], np.inf
    )
    alternatives["rank"] = alternatives.groupby("od_month_id")["eligible_choice_cost"].rank(method="first")
    best = alternatives[alternatives["rank"].eq(1)].copy()
    best["arctic_selected"] = (
        best["cmip6_continuous"]
        & (best["arctic_generalized_cost"] < best["baseline_generalized_cost"])
    )
    best["reconnection_score"] = (
        best["activity_weight"].clip(lower=0.0)
        * best["generalized_cost_saving_pct"].clip(lower=0.0)
        * best["distance_saving_pct"].clip(lower=0.0)
    )
    return alternatives, best


def main() -> None:
    od = representative_monthly_od()
    candidate = prepare_candidate_baselines(od)
    alternatives, best = build_route_choices(candidate)
    candidate.to_csv(DATA / "ethan_ais_arctic_candidate_baselines.csv", index=False)
    alternatives.to_csv(DATA / "ethan_ais_four_route_alternatives.csv", index=False)
    best.to_csv(DATA / "ethan_ais_best_arctic_route_by_od_month.csv", index=False)

    summary = (
        best.groupby(["shiptype", "start_month", "route"], as_index=False)
        .agg(
            candidate_od=("od_month_id", "size"),
            selected_od=("arctic_selected", "sum"),
            activity_weight=("activity_weight", "sum"),
            selected_activity_weight=(
                "activity_weight",
                lambda x: float(x[best.loc[x.index, "arctic_selected"]].sum()),
            ),
            median_distance_saving_pct=("distance_saving_pct", "median"),
            median_cost_saving_pct=("generalized_cost_saving_pct", "median"),
        )
    )
    summary["selected_od_pct"] = 100.0 * summary["selected_od"] / summary["candidate_od"]
    summary["selected_activity_pct"] = 100.0 * summary["selected_activity_weight"] / summary["activity_weight"]
    summary.to_csv(DATA / "ethan_ais_arctic_route_choice_summary.csv", index=False)
    print("OD-month candidates:", len(candidate))
    print("Selected Arctic OD-months:", int(best["arctic_selected"].sum()))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
