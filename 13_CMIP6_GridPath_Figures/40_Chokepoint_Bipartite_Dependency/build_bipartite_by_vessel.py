"""Standalone five-panel Supplementary Figure 9; no manuscript integration."""

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

from build_bipartite import EDGE_COLORS, REGIONS, ROOT, SOURCE, protected_paths, sha


HERE = Path(__file__).resolve().parent
OUT = HERE / "five_panels"
COHORTS = [("all", "All vessel types"), ("container", "Container"),
           ("bulk", "Dry bulk"), ("tanker", "Tanker"), ("lng", "LNG")]
EDGE_WIDTH = lambda p: .10 + .046 * p
NODE_AREA = lambda usd: 12 + .14 * usd / 1e9


def aggregate(d, cohort, registry):
    masks = d.baseline_mask.to_numpy(dtype=np.int64)
    incidence = (masks[:, None] & (1 << np.arange(24))) != 0
    degree = incidence.sum(axis=1)
    w = d.trade_weight_usd.to_numpy()
    divided = w / np.maximum(1, degree)
    observed = d.baseline_source.eq("observed_AIS_polyline").to_numpy()
    imputed = d.sea_share_imputed.eq(True).to_numpy()
    edges, nodes = [], []
    for side in ("origin", "dest"):
        for region in REGIONS:
            pick = d[f"{side}_region"].eq(region).to_numpy()
            denominator = w[pick].sum()
            nodes.append(dict(cohort=cohort, side=side, region=region, value_usd=denominator,
                              records=int(pick.sum()), no_gate_value_usd=w[pick & (degree == 0)].sum()))
            for bit, gate in registry.iterrows():
                yes = pick & incidence[:, bit]
                exposed = w[yes].sum()
                edges.append(dict(cohort=cohort, side=side, region=region, gate=gate.key,
                                  gate_name=gate["name"], exposed_value_usd=exposed,
                                  region_value_usd=denominator,
                                  dependency_percent=100 * exposed / denominator if denominator > 0 else np.nan,
                                  equal_split_value_usd=divided[yes].sum(), route_records=int(yes.sum()),
                                  observed_geometry_value_usd=w[yes & observed].sum(),
                                  network_geometry_value_usd=w[yes & ~observed].sum(),
                                  imputed_modal_value_usd=w[yes & imputed].sum()))
    e, n = pd.DataFrame(edges), pd.DataFrame(nodes)
    e["rank_in_region_direction"] = e.groupby(["side", "region"]).dependency_percent.rank(method="first", ascending=False)
    e["highlighted_top3"] = e.rank_in_region_direction.le(3) & e.exposed_value_usd.gt(0)
    e["source_node"] = np.where(e.side.eq("origin"), "region:" + e.region, "gate:" + e.gate)
    e["target_node"] = np.where(e.side.eq("origin"), "gate:" + e.gate, "region:" + e.region)
    gates = registry.rename(columns={"key": "gate", "name": "gate_name"})[["number", "gate", "gate_name"]].merge(
        e[e.side.eq("origin")].groupby("gate").exposed_value_usd.sum().rename("importance_usd"),
        on="gate", validate="one_to_one")
    gates["cohort"] = cohort
    assert len(e) == 288 and not e.duplicated(["side", "region", "gate"]).any()
    assert e.dependency_percent.dropna().between(0, 100 + 1e-9).all()
    assert e.loc[e.region_value_usd.eq(0), "dependency_percent"].isna().all()
    for side in ("origin", "dest"):
        np.testing.assert_allclose(n[n.side.eq(side)].value_usd.sum(), w.sum(), rtol=1e-12)
        np.testing.assert_allclose(e[e.side.eq(side)].equal_split_value_usd.sum() + w[degree == 0].sum(),
                                   w.sum(), rtol=1e-12)
    np.testing.assert_allclose(e[e.side.eq("origin")].groupby("gate").exposed_value_usd.sum().sort_index(),
                               e[e.side.eq("dest")].groupby("gate").exposed_value_usd.sum().sort_index(), rtol=1e-12)
    summary = dict(cohort=cohort, records=len(d), allocated_value_usd=float(w.sum()),
                   country_pairs=int(d[["origin_iso3", "dest_iso3"]].drop_duplicates().shape[0]),
                   months=sorted(d.start_month.unique().astype(int).tolist()),
                   region_nodes=6, gate_nodes=24, positive_directed_edges=int(e.exposed_value_usd.gt(0).sum()),
                   maximum_dependency_percent=float(e.dependency_percent.max()),
                   no_gate_percent=float(100 * w[degree == 0].sum() / w.sum()),
                   actual_AIS_geometry_value_percent=float(100 * w[observed].sum() / w.sum()),
                   median_modal_imputation_value_percent=float(100 * w[imputed].sum() / w.sum()))
    return e, n, gates, summary


def draw_panel(fig, rect, edges, nodes, gates, region_order, gate_order, letter, name, summary, lead=False):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    node_font = 10.0 if lead else 8.8
    money_font = 8.2 if lead else 7.3
    gate_font = 9.5 if lead else 8.0
    title_y = rect[1] + rect[3] + .025
    fig.text(rect[0] + .008, title_y, letter, fontsize=16, fontweight="bold", va="baseline")
    fig.text(rect[0] + .030, title_y, name, fontsize=13, fontweight="bold", va="baseline")
    fig.text(rect[0] + .030, title_y-.014, f"n = {summary['records']:,}; allocated value = US${summary['allocated_value_usd']/1e9:,.1f}bn",
             fontsize=8.5, color="#59666F", va="baseline")
    ry = dict(zip(region_order, np.linspace(.89, .09, 6)))
    gy = dict(zip(gate_order, np.linspace(.925, .045, 24)))
    rx = {r: .33 - .07*np.sqrt(1-((y-.5)/.47)**2) for r, y in ry.items()}
    gx = {g: .65 + .07*np.sqrt(max(0, 1-((y-.5)/.46)**2)) for g, y in gy.items()}
    ax.text(.28, .987, "Shipping regions", ha="right", va="top", fontsize=node_font, fontweight="bold")
    ax.text(.68, .987, "Chokepoints", ha="left", va="top", fontsize=node_font, fontweight="bold")
    for edge in edges[edges.exposed_value_usd.gt(0)].sort_values(["highlighted_top3", "dependency_percent"]).itertuples():
        outgoing = edge.side == "origin"
        sign = 1 if outgoing else -1
        x0, x1, y0, y1 = rx[edge.region], gx[edge.gate], ry[edge.region], gy[edge.gate]
        span = x1-x0
        pts = np.array([(x0+.006, y0+sign*.003), (x0+span*.37, y0+sign*.029),
                        (x0+span*.63, y1+sign*.017), (x1-.006, y1+sign*.003)])
        if not outgoing:
            pts = pts[::-1]
        alpha = .75 if edge.highlighted_top3 else .085
        color = EDGE_COLORS[edge.side]
        path = MplPath(pts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, fc="none", ec=color, lw=EDGE_WIDTH(edge.dependency_percent),
                              ls="-" if outgoing else (0, (2.8, 2.1)), alpha=alpha, capstyle="round",
                              zorder=3 if edge.highlighted_top3 else 1))
        def bezier(t):
            return (1-t)**3*pts[0] + 3*(1-t)**2*t*pts[1] + 3*(1-t)*t*t*pts[2] + t**3*pts[3]
        t = .62 + .12*(edge.rank_in_region_direction-1) if edge.highlighted_top3 else .74
        ax.add_patch(FancyArrowPatch(bezier(t-.065), bezier(t), arrowstyle="-|>,head_length=.8,head_width=.45",
                                    mutation_scale=8.0 if edge.highlighted_top3 else 3,
                                    lw=.3, color=color, alpha=alpha, shrinkA=0, shrinkB=0,
                                    path_effects=[pe.Stroke(linewidth=.85, foreground="white"), pe.Normal()] if edge.highlighted_top3 else [],
                                    zorder=4 if edge.highlighted_top3 else 2))
    region_importance = nodes.groupby("region").value_usd.mean()
    for region in region_order:
        x, y = rx[region], ry[region]
        nd = nodes[nodes.region.eq(region)].set_index("side")
        ax.scatter([x], [y], s=NODE_AREA(region_importance[region]), fc="#677681", ec="white", lw=.8, zorder=6)
        ax.text(x-.027, y+.022, region, ha="right", va="center", fontsize=node_font, fontweight="bold", color="#25343C")
        ax.text(x-.027, y-.006, f"O: {nd.loc['origin','value_usd']/1e9:,.1f}bn", ha="right", va="center", fontsize=money_font, color="#59666F")
        ax.text(x-.027, y-.032, f"D: {nd.loc['dest','value_usd']/1e9:,.1f}bn", ha="right", va="center", fontsize=money_font, color="#59666F")
    for g in gates.set_index("gate").loc[gate_order].itertuples():
        x, y = gx[g.Index], gy[g.Index]
        ax.scatter([x], [y], s=NODE_AREA(g.importance_usd), fc="#F1F4F5", ec="#677681", lw=.65, zorder=6)
        ax.text(x+.025, y, g.gate_name, ha="left", va="center", fontsize=gate_font, color="#25343C")
    return ax


def plot(cohorts, orders):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(21, 15.2), facecolor="white")
    fig.text(.032, .973, "Supplementary Figure 9", fontsize=17, fontweight="bold")
    rects = [(0.018, .13, .308, .78), (.354, .548, .304, .362),
             (.684, .548, .304, .362), (.354, .13, .304, .362), (.684, .13, .304, .362)]
    axes = []
    for (cohort, name), letter, rect in zip(COHORTS, "abcde", rects):
        edges, nodes, gates, summary = cohorts[cohort]
        region_order = orders[cohort]["region_order"]
        gate_order = orders[cohort]["gate_order"]
        axes.append(draw_panel(fig, rect, edges, nodes, gates, region_order, gate_order,
                               letter, name, summary, lead=cohort=="all"))
    # All panels use common physical line widths and node areas, not per-panel rescaling.
    legend = fig.add_axes((.035, .025, .30, .065))
    legend.axis("off")
    for y, side in ((.76, "origin"), (.32, "dest")):
        origin = side == "origin"
        start, end = (.02, .17) if origin else (.17, .02)
        legend.plot([.02, .17], [y,y], color=EDGE_COLORS[side], lw=2.2, ls="-" if origin else (0, (2.8,2.1)))
        legend.add_patch(FancyArrowPatch((start,y), (end,y), arrowstyle="-|>", mutation_scale=13, color=EDGE_COLORS[side], lw=0))
        legend.text(.21, y, "Region to chokepoint (origin)" if origin else "Chokepoint to region (destination)", fontsize=11, va="center")
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 1)
    lines = [Line2D([0],[0],color="#677681",lw=EDGE_WIDTH(v),label=f"{v}%") for v in (5,25,50,100)]
    fig.legend(handles=lines, loc="lower left", bbox_to_anchor=(.36,.035), ncol=4, frameon=False,
               title="Regional dependence (line width)", fontsize=10.5, title_fontsize=11,
               handlelength=2.5, columnspacing=1.5, labelspacing=1.1)
    circles = [Line2D([0],[0],marker="o",color="none",markerfacecolor="#F1F4F5",markeredgecolor="#677681",
                          markersize=np.sqrt(NODE_AREA(v*1e9)),label=f"{v:,}") for v in (0,100,500,1000)]
    fig.legend(handles=circles, loc="lower left", bbox_to_anchor=(.705,.035), ncol=4, frameon=False,
               title="Node importance (US$bn)", fontsize=10.5, title_fontsize=11,
               columnspacing=1.25, handletextpad=.5, labelspacing=1.1)
    fig.text(.51,.014,"O / D: origin / destination value (US$bn).  Top 3 dependencies per region and direction highlighted.",
             ha="center", fontsize=10, color="#59666F")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.bbox
    outside = []
    for ax in axes:
        for label in ax.texts:
            t = label.get_window_extent(renderer)
            if t.x0 < bbox.x0 or t.x1 > bbox.x1 or t.y0 < bbox.y0 or t.y1 > bbox.y1:
                outside.append(label.get_text())
    assert not outside, outside
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"SuppFig9_Bipartite_All_and_FourVesselTypes.{suffix}", dpi=240, facecolor="white")
    fig.savefig(OUT / "preview.png", dpi=110, facecolor="white")
    # Separate high-resolution panels make the dense comparison easy to inspect.
    for (cohort, _), ax in zip(COHORTS, axes):
        b = ax.get_position()
        region = matplotlib.transforms.Bbox.from_extents(b.x0*21, max(0,b.y0-.012)*15.2,
                  min(1,b.x1+.008)*21, min(1,b.y1+.044)*15.2)
        fig.savefig(OUT / f"panel_{cohort}.png", dpi=200, bbox_inches=region, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "source_data").mkdir(exist_ok=True)
    keep = protected_paths() + [HERE / "build_bipartite.py"] + list((HERE / "figures").glob("*"))
    before = {str(p.relative_to(ROOT)):sha(p) for p in keep if p.is_file()}
    prior_data = {p.name:sha(p) for p in (OUT / "source_data").glob("*.csv") if p.name != "node_plot_order.csv"}
    d = pd.read_parquet(SOURCE / "panel_c_trade_weighted_od.parquet")
    d = d[d.selection_status.eq("included")].copy()
    assert d.od_id.is_unique
    assert set(d.shiptype) == {"container", "bulk", "tanker", "lng"}
    registry = pd.read_csv(ROOT / "data/chokepoints24/registry.csv")
    assert registry.number.tolist() == list(range(1,25))
    cohorts = {key:aggregate(d if key=="all" else d[d.shiptype.eq(key)], key, registry) for key,_ in COHORTS}
    all_edges, all_nodes, all_gates, all_summary = cohorts["all"]
    keys = ["side", "region", "gate"]
    numeric = ["exposed_value_usd", "equal_split_value_usd", "route_records", "observed_geometry_value_usd",
               "network_geometry_value_usd", "imputed_modal_value_usd"]
    totals = pd.concat([cohorts[k][0] for k,_ in COHORTS[1:]]).groupby(keys)[numeric].sum().sort_index()
    reference = all_edges.set_index(keys)[numeric].sort_index()
    np.testing.assert_allclose(totals, reference, rtol=1e-12, atol=1e-4)
    np.testing.assert_allclose(pd.concat([cohorts[k][1] for k,_ in COHORTS[1:]]).groupby(["side","region"]).value_usd.sum().sort_index(),
                               all_nodes.set_index(["side","region"]).value_usd.sort_index(), rtol=1e-12)
    prior = pd.read_csv(HERE / "source_data/region_chokepoint_dependencies.csv").set_index(keys).sort_index()
    np.testing.assert_allclose(all_edges.set_index(keys).sort_index().dependency_percent, prior.dependency_percent, rtol=1e-12)
    assert sum(cohorts[k][3]["records"] for k,_ in COHORTS[1:]) == all_summary["records"]
    orders, rank_rows = {}, []
    for cohort, _ in COHORTS:
        _, nodes, gates, _ = cohorts[cohort]
        region_values = nodes.groupby("region").value_usd.mean().sort_values(ascending=False, kind="stable")
        gate_values = gates.sort_values(["importance_usd", "number"], ascending=[False, True])
        assert np.all(np.diff(region_values.values) <= 0)
        assert np.all(np.diff(gate_values.importance_usd.values) <= 0)
        orders[cohort] = dict(region_order=region_values.index.tolist(), gate_order=gate_values.gate.tolist())
        for rank, (region, value) in enumerate(region_values.items(), 1):
            rank_rows.append(dict(cohort=cohort, node_class="region", node=region, plot_rank=rank, importance_usd=value))
        for rank, gate in enumerate(gate_values.itertuples(), 1):
            rank_rows.append(dict(cohort=cohort, node_class="chokepoint", node=gate.gate, plot_rank=rank, importance_usd=gate.importance_usd))
    pd.DataFrame(rank_rows).to_csv(OUT / "source_data/node_plot_order.csv", index=False)
    frames = [pd.concat([cohorts[k][i] for k,_ in COHORTS]) for i in range(3)]
    for name, frame in zip(("directed_dependencies", "regional_denominators", "gate_importance"), frames):
        frame.to_csv(OUT / "source_data" / f"{name}.csv", index=False)
    pd.DataFrame([cohorts[k][3] for k,_ in COHORTS]).to_csv(OUT / "source_data/cohort_summary.csv", index=False)
    assert all(sha(OUT / "source_data" / name) == digest for name, digest in prior_data.items()), "Numeric source data changed"
    plot(cohorts, orders)
    assert all(sha(ROOT/p)==h for p,h in before.items())
    audit = dict(status="numeric_checks_passed_visual_review_pending", figure_number=9,
                 panels={letter:dict(vessel_type=k,**cohorts[k][3],**orders[k]) for letter,(k,_) in zip("abcde",COHORTS)},
                 layout="Each panel independently ranks both node classes by its own cohort's importance, descending from top to bottom",
                 node_and_edge_scales_shared=True, previous_numeric_source_data_unchanged=True,
                 dependency_denominator="Within the named vessel cohort, region and endpoint direction; no-gate records included",
                 trade_allocation="Original MainFig1c weights retained; no independent rescaling of each vessel type to country-pair totals",
                 all_type_reconstructed_by_four_vessel_types=True, all_type_matches_previous_single_panel=True,
                 relative_shares_are_not_additive=True, origin_and_destination_accounts_not_summed_as_new_trade=True,
                 line_width_pt="0.10 + 0.046 * dependence_percent", marker_area_pt2="12 + 0.14 * importance_in_billion_USD",
                 node_importance="Region: mean origin/destination value; gate: unsplit incident value, counted once per record",
                 publication_scope="Standalone figure only; main manuscript, main figures and appendix integration left unchanged",
                 protected_files_unchanged=True, protected_sha256=before,
                 input_sha256=sha(SOURCE / "panel_c_trade_weighted_od.parquet"),
                 direction_colors=EDGE_COLORS, python=platform.python_version())
    (OUT / "audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"panels":audit["panels"], "protected_files_unchanged":True},indent=2))


if __name__ == "__main__":
    main()
