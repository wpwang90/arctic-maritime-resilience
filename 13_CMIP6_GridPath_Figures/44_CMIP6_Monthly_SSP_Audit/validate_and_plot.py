"""Independent geometry checks and comparable monthly-support figures."""
import calendar
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from compute_seasonality import HERE, ROOT, RESULTS, CACHE, ROUTES, PERIODS, SSPS, routing, sha

FIGURES = HERE / "figures"
KEYS = ["experiment", "period", "screen", "route", "month"]


def validate():
    summary = pd.read_csv(RESULTS / "monthly_route_support.csv")
    members = pd.read_csv(RESULTS / "model_year_month_feasibility.csv")
    paths = pd.read_parquet(RESULTS / "candidate_core_paths.parquet")
    assert len(summary) == 384 and not summary.duplicated(KEYS).any()
    assert len(members) == 384 * 60
    assert (summary.known_network_coverage_pct == 100).all()
    if (RESULTS / "conservative_monthly_route_support.csv").exists():
        summary = pd.concat([summary, pd.read_csv(RESULTS / "conservative_monthly_route_support.csv")], ignore_index=True)
        members = pd.concat([members, pd.read_csv(RESULTS / "conservative_model_year_month_feasibility.csv")], ignore_index=True)
        paths = pd.concat([paths, pd.read_parquet(RESULTS / "conservative_candidate_core_paths.parquet")], ignore_index=True)
        assert len(summary) == 576 and len(members) == 576*60
    grouped = members.groupby(KEYS).common_path_passable.agg(["sum", "size"])
    joined = summary.set_index(KEYS).join(grouped)
    assert (joined["size"] == 60).all()
    assert (joined["sum"] == joined.supported_model_years).all()
    assert (joined.admitted == (joined["sum"] >= 30)).all()
    assert np.allclose(summary.common_path_support_pct, summary.supported_model_years / .6)
    assert (summary.common_candidate_exists == (summary.candidate_path_nodes > 0)).all()
    wide = members.pivot(index=["experiment", "period", "route", "month", "model", "year"],
                         columns="screen", values="member_network_connected")
    assert not (wide.low_ice_sensitivity & ~wide.ice_capable).any()
    if "observed_thickness_sensitivity" in wide:
        assert not (wide.observed_thickness_sensitivity & ~wide.ice_capable).any()
    grids = {r: dict(np.load(routing.OUT / f"geometry_{r}.npz")) for r in ROUTES}
    edges = {r: set(zip(g["row"].tolist(), g["col"].tolist())) for r, g in grids.items()}
    land = routing.prepare_land()
    geometry = []
    for key, p in paths.groupby(KEYS, sort=False):
        p = p.sort_values("order")
        route = key[3]
        grid = grids[route]
        ids = p.node.to_numpy()
        xy = p[["lon", "lat"]].to_numpy()
        land_hits = int(routing.crosses_land(land, xy[:-1], xy[1:]).sum())
        repeated = len(ids) - len(set(ids))
        missing_edges = sum((a, b) not in edges[route] and (b, a) not in edges[route]
                            for a, b in zip(ids[:-1], ids[1:]))
        assert land_hits == repeated == missing_edges == 0
        assert set(ids[[0, -1]]) == set(grid["endpoints"])
        assert np.allclose(xy, grid["points"][ids])
        if route == "TSR":
            assert xy[:, 1].max() >= 85
        geometry.append(dict(zip(KEYS, key), nodes=len(ids), land_crossing_segments=land_hits,
                             repeated_nodes=repeated, invalid_edges=missing_edges,
                             max_latitude=float(xy[:, 1].max())))
    pd.DataFrame(geometry).to_csv(RESULTS / "independent_geometry_checks.csv", index=False)
    winter = pd.read_csv(RESULTS / "winter_original_core_raw_field_audit.csv")
    admitted = winter[winter.passable]
    assert admitted.path_known.all() and (admitted.unknown_nodes == 0).all()
    assert (admitted.max_siconc_pct <= 80).all() and (admitted.max_thickness_m <= 1.2).all()
    counts = winter.groupby(["route", "month"]).passable.sum()
    for key, n in counts.items():
        s = summary[summary.screen.eq("ice_capable") & summary.period.eq(2100) &
                    summary.experiment.eq("ssp585") & summary.route.eq(key[0]) & summary.month.eq(key[1])]
        assert len(s) == 1 and int(s.iloc[0].supported_model_years) == n
    audit = json.loads((RESULTS / "calculation_audit.json").read_text())
    assert audit["mainfig2b_max_abs_difference_pp"] < 1e-10
    assert audit["raw_pass_mask_different_bits"] == 0
    assert all(sha(Path(p)) == digest for p, digest in audit["protected_input_sha256"].items())
    report = dict(status="PASS", scenario_route_month_screen_cells=len(summary),
                  member_route_month_checks=len(members), candidate_paths_checked=len(geometry),
                  candidate_segments_checked=int(sum(g["nodes"] - 1 for g in geometry)),
                  land_crossings=0, repeated_nodes=0, invalid_edges=0,
                  mainfig2b_max_abs_difference_pp=audit["mainfig2b_max_abs_difference_pp"],
                  raw_mask_different_bits=0, all_months_have_60_model_years=True,
                  stricter_node_mask_connectivity_is_subset=True,
                  original_main_figures_unchanged=True,
                  limitation="Numerical reproduction and geometry audit, not observational validation of future navigability. Winter admission is sensitive to how missing thickness at near-zero positive SIC is treated.")
    (RESULTS / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return summary


def heatmap(ax, frame, panel=None):
    z = frame.pivot(index="route", columns="month", values="common_path_support_pct").loc[ROUTES, range(1, 13)]
    exists = frame.pivot(index="route", columns="month", values="common_candidate_exists").loc[ROUTES, range(1, 13)]
    colors = plt.get_cmap("YlGnBu")(np.linspace(0, 1, 256))
    colors[0] = matplotlib.colors.to_rgba("#EEEEEE")
    cmap = ListedColormap(colors)
    mesh = ax.pcolormesh(np.arange(13), np.arange(5), z, cmap=cmap, vmin=0, vmax=100,
                         edgecolors="white", linewidth=.9)
    for r in range(4):
        for c in range(12):
            val = float(z.iloc[r, c])
            if 0 < val < 50:
                ax.add_patch(Rectangle((c, r), 1, 1, fill=False, hatch="///", edgecolor="#bcc6ca", linewidth=0))
            label = f"{val:.1f}".rstrip("0").rstrip(".") if val else "0"
            if not exists.iloc[r, c]:
                label += r"$^{\dagger}$"
            ax.text(c+.5, r+.5, label, ha="center", va="center", fontsize=8.5,
                    color="white" if val >= 65 else "#17282f",
                    fontweight="bold" if val >= 50 else "normal")
    ax.set_xticks(np.arange(12)+.5, [calendar.month_abbr[m] for m in range(1, 13)], fontsize=9)
    ax.set_yticks(np.arange(4)+.5, ROUTES, fontsize=10)
    ax.set_ylim(4, 0)
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if panel:
        ax.text(-.085, 1.10, panel, transform=ax.transAxes, weight="bold", fontsize=14)
    return mesh


def plot_figures(summary):
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"Arial", "pdf.fonttype":42, "svg.fonttype":"none", "hatch.linewidth":.35})
    for screen, number in [("ice_capable", 11), ("low_ice_sensitivity", 12), ("observed_thickness_sensitivity", 13)]:
        if not summary.screen.eq(screen).any():
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.1))
        fig.subplots_adjust(left=.09, right=.985, bottom=.21, top=.88, hspace=.65, wspace=.18)
        for i, period in enumerate(PERIODS):
            for j, ssp in enumerate(SSPS):
                f = summary[summary.screen.eq(screen) & summary.period.eq(period) & summary.experiment.eq(ssp)]
                mesh = heatmap(axes[i, j], f, "abcd"[2*i+j])
                if i == 0:
                    axes[i, j].text(.5, 1.28, {"ssp245":"SSP2-4.5", "ssp585":"SSP5-8.5"}[ssp],
                                    transform=axes[i, j].transAxes, ha="center", fontsize=12)
            a, b = PERIODS[period]
            axes[i, 0].text(-.16, .5, f"{a}-{b}", rotation=90, transform=axes[i, 0].transAxes,
                            ha="center", va="center", fontsize=11)
        cax = fig.add_axes([.28, .12, .44, .028])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0, 20, 40, 60, 80, 100])
        cb.set_label("Common-path model-year support (%)", fontsize=10)
        cb.outline.set_visible(False)
        fig.legend(handles=[Patch(facecolor="#e3eee9", hatch="///", edgecolor="#abb8bd", label="0 < support < 50%: not admitted"),
                            Patch(facecolor="#eeeeee", edgecolor="none", label=r"0$^{\dagger}$: no majority-node candidate found")],
                   loc="lower center", bbox_to_anchor=(.5, .003), ncol=2, frameon=False, fontsize=9)
        stem = f"SupplementaryFig{number}_CMIP6_Monthly_{screen}"
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(FIGURES / f"{stem}.{ext}", dpi=240, facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(4, 1, figsize=(7.1, 8.2))
        fig.subplots_adjust(left=.09, right=.99, bottom=.20, top=.95, hspace=.76)
        for i, (period, ssp) in enumerate((p, s) for p in PERIODS for s in SSPS):
            f = summary[summary.screen.eq(screen) & summary.period.eq(period) & summary.experiment.eq(ssp)]
            mesh = heatmap(axes[i], f, "abcd"[i])
            label = {"ssp245":"SSP2-4.5", "ssp585":"SSP5-8.5"}[ssp]
            a, b = PERIODS[period]
            axes[i].text(.5, 1.17, f"{a}-{b} | {label}", transform=axes[i].transAxes,
                         ha="center", fontsize=10)
        cax = fig.add_axes([.21, .12, .62, .018])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0, 20, 40, 60, 80, 100])
        cb.set_label("Common-path model-year support (%)", fontsize=9)
        cb.outline.set_visible(False)
        fig.legend(handles=[Patch(facecolor="#e3eee9", hatch="///", edgecolor="#abb8bd", label="0 < support < 50%: not admitted"),
                            Patch(facecolor="#eeeeee", edgecolor="none", label=r"0$^{\dagger}$: no majority-node candidate found")],
                   loc="lower center", bbox_to_anchor=(.5, .001), ncol=1, frameon=False, fontsize=8)
        for ext in ["png", "pdf"]:
            fig.savefig(FIGURES / f"{stem}_Portrait.{ext}", dpi=240, facecolor="white")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.39, top=.97)
    f = summary[summary.screen.eq("ice_capable") & summary.period.eq(2100) & summary.experiment.eq("ssp585")]
    mesh = heatmap(ax, f)
    cax = fig.add_axes([.28, .13, .44, .045])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0, 20, 40, 60, 80, 100])
    cb.set_label("Common-path model-year support (%)", fontsize=10)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"MainFig2b_Audited_Readout.{ext}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    plot_figures(validate())
