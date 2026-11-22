"""Add a dependency projection of MainFig1c without editing its inputs/outputs."""

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "25_MainFig1_LatestNetwork"
FIGURES = HERE / "figures"
DATA = HERE / "source_data"
REGIONS = ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"]
COLORS = {"Asia": "#0077BB", "Europe": "#009988", "North America": "#EE7733",
          "South America": "#AA4499", "Africa": "#CC6677", "Oceania": "#8D9A3D"}
NONE = "No listed chokepoint"
EDGE_COLORS = {"origin": "#0077BB", "dest": "#D55E00"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_paths():
    files = [SOURCE / name for name in (
        "panel_c_trade_weighted_od.parquet", "panel_c_allocated_od_weights.parquet",
        "panel_c_region_gate_flows.csv", "panel_c_gate_shares.csv",
        "panel_c_endpoint_region_shares.csv", "panel_c_region_sankey_audit.json",
        "baseline_all.parquet")]
    files += list((SOURCE / "figures").glob("MainFig1c*"))
    files += list((ROOT / "37_FiveClosures_BoxScatter/figures").glob("MainFig[1-5].*"))
    files += list((ROOT / "38_Overleaf_Package/Overleaf_Arctic_Trade").rglob("*"))
    return [p for p in files if p.is_file()]


def calculate():
    d = pd.read_parquet(SOURCE / "panel_c_trade_weighted_od.parquet")
    d = d.loc[d.selection_status.eq("included")].copy()
    registry = pd.read_csv(ROOT / "data/chokepoints24/registry.csv")
    audit0 = json.loads((SOURCE / "panel_c_region_sankey_audit.json").read_text())
    assert len(registry) == 24
    assert registry.number.tolist() == list(range(1, 25))
    assert d.od_id.is_unique and d.trade_weight_usd.gt(0).all()
    assert np.isfinite(d.trade_weight_usd).all()
    assert set(d.origin_region) == set(REGIONS) == set(d.dest_region)
    assert len(d) == audit0["included_intercontinental_rows"]
    total = d.trade_weight_usd.sum()
    np.testing.assert_allclose(total, audit0["included_trade_weight_usd"], rtol=1e-12)
    masks = d.baseline_mask.to_numpy(dtype=np.int64)
    assert ((masks >= 0) & (masks < (1 << 24))).all()
    incidence = (masks[:, None] & (1 << np.arange(24))) != 0
    degree = incidence.sum(axis=1)
    weights = d.trade_weight_usd.to_numpy()
    allocated = weights / np.maximum(degree, 1)
    observed = d.baseline_source.eq("observed_AIS_polyline").to_numpy()
    imputed = d.sea_share_imputed.eq(True).to_numpy()
    records, regions = [], []
    for side in ("origin", "dest"):
        for region in REGIONS:
            pick = d[f"{side}_region"].eq(region).to_numpy()
            denominator = weights[pick].sum()
            no_gate = weights[pick & (degree == 0)].sum()
            regions.append(dict(side=side, region=region, trade_weight_usd=denominator,
                                record_count=int(pick.sum()), no_gate_value_usd=no_gate,
                                no_gate_percent=100 * no_gate / denominator,
                                any_gate_percent=100 * (1 - no_gate / denominator),
                                mean_gate_count_trade_weighted=np.dot(weights[pick], degree[pick]) / denominator))
            for g, row in registry.iterrows():
                yes = pick & incidence[:, g]
                exposed = weights[yes].sum()
                records.append(dict(side=side, region=region, gate=row.key, gate_number=int(row.number),
                                    gate_name=row["name"], exposed_value_usd=exposed,
                                    region_value_usd=denominator, dependency_percent=100 * exposed / denominator,
                                    equal_split_value_usd=allocated[yes].sum(),
                                    equal_split_region_percent=100 * allocated[yes].sum() / denominator,
                                    route_records=int(yes.sum()),
                                    observed_geometry_value_usd=weights[yes & observed].sum(),
                                    network_geometry_value_usd=weights[yes & ~observed].sum(),
                                    imputed_modal_share_value_usd=weights[yes & imputed].sum()))
    edges = pd.DataFrame(records)
    nodes = pd.DataFrame(regions)
    assert len(edges) == 2 * 6 * 24
    assert not edges.duplicated(["side", "region", "gate"]).any()
    assert edges.dependency_percent.between(0, 100).all()
    assert (edges.equal_split_value_usd <= edges.exposed_value_usd + 1e-3).all()
    np.testing.assert_allclose(edges.observed_geometry_value_usd + edges.network_geometry_value_usd,
                               edges.exposed_value_usd, rtol=1e-12)
    for side in ("origin", "dest"):
        subset = edges[edges.side.eq(side)]
        np.testing.assert_allclose(nodes[nodes.side.eq(side)].trade_weight_usd.sum(), total, rtol=1e-12)
        np.testing.assert_allclose(subset.equal_split_value_usd.sum() + weights[degree == 0].sum(), total, rtol=1e-12)
    for field in ("exposed_value_usd", "equal_split_value_usd"):
        np.testing.assert_allclose(edges[edges.side.eq("origin")].groupby("gate")[field].sum().sort_index(),
                                   edges[edges.side.eq("dest")].groupby("gate")[field].sum().sort_index(), rtol=1e-12)

    # Reproduce the original conservation-preserving Sankey before using unsplit exposure.
    original = pd.read_csv(SOURCE / "panel_c_gate_shares.csv").set_index("gate")
    split_gate = edges[edges.side.eq("origin")].groupby("gate").equal_split_value_usd.sum()
    np.testing.assert_allclose(split_gate.sort_index(), original.loc[split_gate.index, "allocated_weight"].sort_index(), rtol=1e-12)
    np.testing.assert_allclose(weights[degree == 0].sum(), original.loc[NONE, "allocated_weight"], rtol=1e-12)
    original_flow = pd.read_csv(SOURCE / "panel_c_region_gate_flows.csv")
    expanded = []
    for g, gate in enumerate(registry.key):
        chunk = d.loc[incidence[:, g], ["origin_region", "dest_region"]].copy()
        chunk.columns = ["origin", "dest"]
        chunk["gate"] = gate
        chunk["allocated_weight"] = allocated[incidence[:, g]]
        expanded.append(chunk)
    chunk = d.loc[degree == 0, ["origin_region", "dest_region"]].copy()
    chunk.columns = ["origin", "dest"]
    chunk["gate"] = NONE
    chunk["allocated_weight"] = weights[degree == 0]
    expanded.append(chunk)
    reconstructed = pd.concat(expanded).groupby(["origin", "gate", "dest"]).allocated_weight.sum().sort_index()
    ref = original_flow.set_index(["origin", "gate", "dest"]).allocated_weight.sort_index()
    assert reconstructed.index.equals(ref.index)
    np.testing.assert_allclose(reconstructed, ref, rtol=1e-12)

    gate_totals = registry.rename(columns={"key": "gate", "name": "gate_name"})[["number", "gate", "gate_name"]].merge(
        edges[edges.side.eq("origin")].groupby("gate").agg(exposed_value_usd=("exposed_value_usd", "sum"),
            equal_split_value_usd=("equal_split_value_usd", "sum")), on="gate", validate="one_to_one")
    gate_totals["sample_exposure_percent"] = 100 * gate_totals.exposed_value_usd / total
    gate_totals["sankey_allocation_percent"] = 100 * gate_totals.equal_split_value_usd / total
    gate_totals = gate_totals.sort_values("exposed_value_usd", ascending=False)
    gate_totals["plot_order"] = np.arange(1, 25)
    edges["dependency_rank_in_region"] = edges.groupby(["side", "region"]).dependency_percent.rank(method="first", ascending=False).astype(int)
    edges["highlighted_top3"] = edges.dependency_rank_in_region.le(3) & edges.dependency_percent.gt(0)
    edges["source_node"] = np.where(edges.side.eq("origin"), "region:" + edges.region, "gate:" + edges.gate)
    edges["target_node"] = np.where(edges.side.eq("origin"), "gate:" + edges.gate, "region:" + edges.region)
    edges.to_csv(DATA / "region_chokepoint_dependencies.csv", index=False)
    nodes.to_csv(DATA / "regional_denominators.csv", index=False)
    gate_totals.to_csv(DATA / "chokepoint_totals_and_sankey_reconciliation.csv", index=False)
    importance = []
    region_importance = nodes.groupby("region").trade_weight_usd.mean().sort_values(ascending=False)
    for rank, (region, value) in enumerate(region_importance.items(), 1):
        importance.append(dict(node="region:" + region, node_class="region", label=region,
                               importance_usd=value, metric="mean_origin_destination_value", rank_within_class=rank))
    for rank, row in enumerate(gate_totals.itertuples(), 1):
        importance.append(dict(node="gate:" + row.gate, node_class="chokepoint", label=row.gate_name,
                               importance_usd=row.exposed_value_usd, metric="full_incidence_trade_value", rank_within_class=rank))
    importance = pd.DataFrame(importance)
    importance["marker_area_pt2"] = 18 + .30 * importance.importance_usd / 1e9
    importance.to_csv(DATA / "node_importance.csv", index=False)
    for side in ("origin", "dest"):
        edges[edges.side.eq(side)].pivot(index="gate_name", columns="region", values="dependency_percent").reindex(
            index=gate_totals.gate_name, columns=REGIONS).to_csv(DATA / f"{side}_dependency_percent_matrix.csv")

    # Retain type/month distinctions in source data; no additional sample is introduced.
    monthly = []
    for (vessel, month), sub in d.groupby(["shiptype", "start_month"]):
        for side in ("origin", "dest"):
            for region, subset in sub.groupby(f"{side}_region"):
                denom = subset.trade_weight_usd.sum()
                for bit, row in registry.iterrows():
                    exposed = subset.loc[(subset.baseline_mask.astype(np.int64) & (1 << bit)).ne(0), "trade_weight_usd"].sum()
                    monthly.append(dict(shiptype=vessel, month=int(month), side=side, region=region, gate=row.key,
                                        region_value_usd=denom, exposed_value_usd=exposed, dependency_percent=100 * exposed / denom))
    pd.DataFrame(monthly).to_csv(DATA / "type_month_region_chokepoint_dependencies.csv", index=False)
    stats = dict(status="numeric_checks_passed_visual_review_pending", sample_records=len(d),
                 directed_country_pairs=int(d[["origin_iso3", "dest_iso3"]].drop_duplicates().shape[0]),
                 regions=REGIONS, chokepoints=24, trade_value_usd=float(total),
                 no_listed_gate_value_usd=float(weights[degree == 0].sum()),
                 no_listed_gate_percent=float(100 * weights[degree == 0].sum() / total),
                 multi_gate_value_percent=float(100 * weights[degree > 1].sum() / total),
                 mean_gate_count_trade_weighted=float(np.dot(weights, degree) / total),
                 observed_geometry_percent=float(100 * weights[observed].sum() / total),
                 imputed_modal_share_percent=float(100 * weights[imputed].sum() / total),
                 graph=dict(type="single_directed_bipartite", region_nodes=6, gate_nodes=24,
                            positive_directed_edges=int(edges.exposed_value_usd.gt(0).sum()),
                            reciprocal_pairs=int((edges.pivot(index=["region", "gate"], columns="side", values="exposed_value_usd").gt(0).all(axis=1)).sum())),
                 positive_links_by_direction=edges.groupby("side").exposed_value_usd.apply(lambda x: int(x.gt(0).sum())).to_dict(),
                 maximum_dependency_percent=float(edges.dependency_percent.max()),
                 layout="ranked curved bipartite; descending economic importance within each node class",
                 region_order=region_importance.index.tolist(),
                 direction_colours=EDGE_COLORS,
                 original_sankey_flow_triples_reproduced=len(ref),
                 maximum_sankey_reconstruction_error_usd=float(np.max(np.abs(reconstructed.values - ref.values))),
                 sources={"trade_year": 2024, "modal_share_year": 2021, "AIS_year": 2024,
                          "weighting": "BACI2024 x UNCTAD2021 sea share; missing source shares use flagged median; activity allocation within country pair"},
                 definition="full record weight per encountered gate / all included regional record weights; not additive over gates",
                 version={"python": platform.python_version(), "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "numpy": np.__version__},
                 top3_dependencies=edges[edges.highlighted_top3][["side", "region", "gate", "dependency_percent"]].to_dict("records"))
    return edges, nodes, gate_totals, stats


def plot(edges, nodes, gates):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(9.2, 11.4), facecolor="white")
    ax = fig.add_axes((0.035, 0.14, 0.94, 0.81))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    line_width = lambda p: 0.12 + 0.105 * p
    gate_y = dict(zip(gates.gate, np.linspace(0.94, 0.055, 24)))
    region_importance = nodes.groupby("region").trade_weight_usd.mean().sort_values(ascending=False)
    region_y = dict(zip(region_importance.index, np.linspace(0.90, 0.10, 6)))
    region_x = {r: .33 - .07 * np.sqrt(1 - ((y-.50)/.47)**2) for r, y in region_y.items()}
    gate_x = {g: .65 + .07 * np.sqrt(max(0, 1 - ((y-.50)/.46)**2)) for g, y in gate_y.items()}
    area = lambda value: 18 + .30 * value / 1e9
    ax.text(.28, 1.01, "Shipping regions", ha="right", va="bottom", fontsize=12, fontweight="bold")
    ax.text(.68, 1.01, "Chokepoints", ha="left", va="bottom", fontsize=12, fontweight="bold")
    selected = edges[edges.exposed_value_usd.gt(0)].sort_values(["highlighted_top3", "dependency_percent"])
    for row in selected.itertuples():
        y0, y1 = region_y[row.region], gate_y[row.gate]
        x0, x1 = region_x[row.region], gate_x[row.gate]
        outgoing = row.side == "origin"
        sign = 1 if outgoing else -1
        span = x1-x0
        pts = np.array([(x0 + .006, y0 + sign * .003), (x0 + span*.37, y0 + sign * .030),
                        (x0 + span*.63, y1 + sign * .018), (x1 - .006, y1 + sign * .003)])
        if not outgoing:
            pts = pts[::-1]
        path = MplPath(pts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        alpha = .76 if row.highlighted_top3 else .09
        color = EDGE_COLORS[row.side]
        ax.add_patch(PathPatch(path, fc="none", ec=color, lw=line_width(row.dependency_percent),
                               ls="-" if outgoing else (0, (2.8, 2.1)), alpha=alpha, capstyle="round",
                               zorder=3 if row.highlighted_top3 else 1))
        # Mid-edge arrowheads avoid merging many heads at a region or gate node.
        def bezier(t):
            return (1-t)**3 * pts[0] + 3*(1-t)**2*t*pts[1] + 3*(1-t)*t**2*pts[2] + t**3*pts[3]
        arrow_t = .62 + .12 * (row.dependency_rank_in_region - 1) if row.highlighted_top3 else .74
        ax.add_patch(FancyArrowPatch(bezier(arrow_t - .06), bezier(arrow_t), arrowstyle="-|>,head_length=0.8,head_width=0.45",
                                    mutation_scale=11.5 if row.highlighted_top3 else 4,
                                    lw=.4, color=color, alpha=alpha,
                                    path_effects=[pe.Stroke(linewidth=1.1, foreground="white"), pe.Normal()] if row.highlighted_top3 else [],
                                    shrinkA=0, shrinkB=0, zorder=4 if row.highlighted_top3 else 2))
    for region, y in region_y.items():
        info = nodes[nodes.region.eq(region)].set_index("side")
        x = region_x[region]
        ax.scatter([x], [y], s=area(region_importance.loc[region]), color="#677681", edgecolors="white", linewidths=1, zorder=6)
        ax.text(x-.03, y + .023, region, ha="right", va="center", fontsize=11.5, fontweight="bold", color="#25343C")
        ax.text(x-.03, y - .004, f"Origin: US${info.loc['origin', 'trade_weight_usd']/1e9:,.1f}bn",
                ha="right", va="center", fontsize=9.5, color="#56646C")
        ax.text(x-.03, y - .028, f"Dest.: US${info.loc['dest', 'trade_weight_usd']/1e9:,.1f}bn",
                ha="right", va="center", fontsize=9.5, color="#56646C")
    for row in gates.itertuples():
        y = gate_y[row.gate]
        x = gate_x[row.gate]
        ax.scatter([x], [y], s=area(row.exposed_value_usd), facecolors="#F1F4F5", edgecolors="#677681", linewidths=.8, zorder=6)
        ax.text(x+.026, y, row.gate_name, ha="left", va="center", fontsize=10, color="#25343C")
    handles = [Line2D([0], [0], color="#59656B", lw=line_width(v), alpha=.8, label=f"{v}%") for v in (5, 20, 50)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.03, .06), ncol=3, frameon=False,
               title="Regional dependence (line width)", handlelength=2.4, fontsize=9.5, title_fontsize=9.5,
               labelspacing=1, columnspacing=2.2)
    legend = fig.add_axes((.55, .042, .42, .09))
    legend.axis("off")
    for y, origin in ((.77, True), (.42, False)):
        x0, x1 = (.02, .18) if origin else (.18, .02)
        color = EDGE_COLORS["origin" if origin else "dest"]
        legend.plot([.02, .18], [y, y], color=color, lw=1.7, ls="-" if origin else (0, (2.8, 2.1)))
        legend.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", color=color, mutation_scale=10, lw=0))
        legend.text(.22, y, "Region to gate (origin)" if origin else "Gate to region (destination)", va="center", fontsize=10)
    legend.text(.02, .07, "Top 3 per region and direction highlighted", va="center", fontsize=9, color="#59656B")
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 1)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#F1F4F5", markeredgecolor="#677681",
                      markersize=np.sqrt(area(v*1e9)), label=f"{v:,}") for v in (100, 500, 1000)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.03, .004), ncol=3, frameon=False,
               title="Node importance (US$bn)", fontsize=9, title_fontsize=9.5, handletextpad=1, columnspacing=2.3)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"SuppFig9_Region_Chokepoint_Bipartite.{ext}", dpi=240, facecolor="white")
    fig.savefig(FIGURES / "SuppFig9_preview.png", dpi=110, facecolor="white")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths()}
    edges, nodes, gates, audit = calculate()
    plot(edges, nodes, gates)
    assert all(sha(ROOT / p) == digest for p, digest in before.items()), "Protected source changed"
    audit["protected_sources_unchanged"] = True
    audit["protected_source_sha256"] = before
    audit["figure_sha256"] = {p.name: sha(p) for p in FIGURES.iterdir() if p.is_file()}
    (HERE / "analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("protected_source_sha256", "top3_dependencies")}, indent=2))


if __name__ == "__main__":
    main()
