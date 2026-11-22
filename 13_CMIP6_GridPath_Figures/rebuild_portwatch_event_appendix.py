"""Recompute historical event evidence from untouched IMF PortWatch daily data."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "35_PortWatch_Event_Appendix"
SOURCE = ROOT.parent / "10_Nature_RealData_Upgrade"
RAW = SOURCE / "data_public/portwatch/portwatch_key_chokepoints_2019_2025.csv"
OLD = SOURCE / "figures/portwatch_real_disruption_event_stats.csv"
SERVICE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0"
COLORS = {"Suez Canal": "#0077A9", "Bab el-Mandeb Strait": "#75539B",
          "Cape of Good Hope": "#E67B20", "Panama Canal": "#0077A9"}
METRICS = ["n_total", "capacity", "n_container", "n_dry_bulk", "n_tanker",
           "n_general_cargo", "n_roro"]
SERIES = [
    ("suez_gate", "Suez blockage", "Suez Canal", "2021-03-23", "2021-03-29", "2021-02-15", "2021-03-15", "Suez 2021 / canal"),
    ("suez_cape", "Suez blockage", "Cape of Good Hope", "2021-03-23", "2021-04-20", "2021-02-15", "2021-03-15", "Suez 2021 / Cape"),
    ("red_bab", "Red Sea crisis", "Bab el-Mandeb Strait", "2023-12-15", "2024-03-31", "2023-09-01", "2023-11-15", "Red Sea / Bab el-Mandeb"),
    ("red_suez", "Red Sea crisis", "Suez Canal", "2023-12-15", "2024-03-31", "2023-09-01", "2023-11-15", "Red Sea / Suez"),
    ("red_cape", "Red Sea crisis", "Cape of Good Hope", "2023-12-15", "2024-03-31", "2023-09-01", "2023-11-15", "Red Sea / Cape"),
    ("panama_gate", "Panama drought", "Panama Canal", "2023-08-01", "2024-02-29", "2022-08-01", "2023-02-28", "Panama drought / canal"),
]


def subset(daily, gate, start, end):
    data = daily.loc[daily.portname.eq(gate)].set_index("date").sort_index()
    dates = pd.date_range(start, end)
    result = data.reindex(dates)
    if result[METRICS].isna().any().any():
        raise ValueError(f"Missing daily data: {gate}, {start}, {end}")
    return result


def changes(daily, spec, base_start=None, base_end=None):
    key, event, gate, start, end, bs, be, label = spec
    bs, be = base_start or bs, base_end or be
    b = subset(daily, gate, bs, be)
    e = subset(daily, gate, start, end)
    rows = []
    for metric in METRICS:
        denominator, numerator = b[metric].mean(), e[metric].mean()
        rows.append(dict(series_id=key, event=event, chokepoint=gate, label=label,
                         event_start=start, event_end=end, baseline_start=str(bs)[:10],
                         baseline_end=str(be)[:10], baseline_days=len(b), event_days=len(e),
                         metric=metric, baseline_mean=float(denominator), event_mean=float(numerator),
                         change_pct=float(100 * (numerator / denominator - 1)) if denominator > 0 else np.nan))
    return rows


def calculate():
    OUT.mkdir(exist_ok=True)
    (OUT / "source_data").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    daily = pd.read_csv(RAW, parse_dates=["date"])
    assert not daily.duplicated(["portid", "date"]).any()
    assert (daily[METRICS] >= 0).all().all()
    assert (daily.n_total == daily.n_cargo + daily.n_tanker).all()
    assert (daily.n_cargo == daily[["n_container", "n_dry_bulk", "n_general_cargo", "n_roro"]].sum(axis=1)).all()
    capacity_columns = ["capacity_container", "capacity_dry_bulk", "capacity_general_cargo", "capacity_roro", "capacity_tanker"]
    capacity_rounding_residual = daily.capacity - daily[capacity_columns].sum(axis=1)
    # Published integer capacity columns differ by up to four reporting units.
    # Retain the reported total, rather than manufacturing an exact identity.
    assert capacity_rounding_residual.abs().max() <= 5
    results = pd.DataFrame([r for spec in SERIES for r in changes(daily, spec)])
    results.to_csv(OUT / "event_statistics.csv", index=False)
    sensitivity = []
    for spec in SERIES:
        start, end = pd.Timestamp(spec[3]), pd.Timestamp(spec[4])
        baselines = {"Legacy baseline": (spec[5], spec[6]),
                     "Preceding 28 days": (str((start-pd.Timedelta(days=28)).date()), str((start-pd.Timedelta(days=1)).date())),
                     "Preceding 60 days": (str((start-pd.Timedelta(days=60)).date()), str((start-pd.Timedelta(days=1)).date())),
                     "Previous calendar year": (str((start-pd.DateOffset(years=1)).date()), str((end-pd.DateOffset(years=1)).date()))}
        for rule, (bs, be) in baselines.items():
            for row in changes(daily, spec, bs, be):
                sensitivity.append(dict(row, baseline_rule=rule))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / "baseline_sensitivity.csv", index=False)
    old = pd.read_csv(OLD)
    old_errors = []
    for row in old.itertuples():
        found = results.loc[(results.chokepoint == row.chokepoint) & (results.event_start == row.event_start) & (results.event_end == row.event_end)]
        for metric, oldcol in [("n_total", "n_total_change_pct"), ("capacity", "capacity_change_pct")]:
            value = found.loc[found.metric == metric, "change_pct"].item()
            old_errors.append(abs(value - getattr(row, oldcol)))
    assert max(old_errors) < 1e-10
    retained = daily.loc[daily.portname.isin({s[2] for s in SERIES})].copy()
    retained.to_csv(OUT / "source_data/portwatch_daily_event_gates.csv", index=False)
    shutil.copy2(OLD, OUT / "source_data/legacy_event_windows_and_results.csv")
    meta = dict(raw_path=str(RAW), sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
                cached_rows=len(daily), cached_first_date=str(daily.date.min().date()),
                cached_last_date=str(daily.date.max().date()), source_service=SERVICE,
                version="Retained local historical snapshot; original retrieval date not recorded",
                date_of_source_metadata_check="2026-10-03", calibrated_tables_used=False,
                legacy_max_absolute_difference_pp=max(old_errors), no_missing_window_days=True,
                count_sum_checks=True, capacity_max_reporting_residual=float(capacity_rounding_residual.abs().max()),
                independent_lng_series_available=False,
                uncertainty="Alternative baselines, not confidence intervals",
                scope="Descriptive event-window check; no event-outcome fitting or causal attribution")
    try:
        for filename, url in [("official_service_metadata.json", SERVICE + "?f=pjson"),
                              ("official_item_metadata.json", "https://www.arcgis.com/sharing/rest/content/items/3da2b9ca97684916b75c4013f95d18ab?f=pjson")]:
            with urllib.request.urlopen(url, timeout=40) as response:
                content = json.load(response)
            (OUT / "source_data" / filename).write_text(json.dumps(content, indent=2))
        meta["official_metadata_verified"] = True
    except Exception as exc:
        meta["official_metadata_verified"] = False
        meta["metadata_error"] = str(exc)
    (OUT / "data_audit.json").write_text(json.dumps(meta, indent=2))
    return daily, results, sensitivity


def panel_label(ax, label):
    ax.text(-.10, 1.07, label, transform=ax.transAxes, fontsize=14, weight="bold")


def plot(daily, results, sensitivity):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 10, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(16.8, 9.6))
    grid = fig.add_gridspec(2, 3, left=.075, right=.985, bottom=.13, top=.955,
                           hspace=.42, wspace=.39, height_ratios=[1, 1.12])
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(3)]
    daily_rows = []
    settings = [("2021-02-01", "2021-05-15", ["suez_gate", "suez_cape"]),
                ("2023-09-01", "2024-05-15", ["red_bab", "red_suez", "red_cape"]),
                ("2023-06-01", "2024-04-30", ["panama_gate"])]
    for ax, (start, end, keys), label in zip(axes[:3], settings, "abc"):
        for key in keys:
            row = results.loc[(results.series_id == key) & results.metric.eq("n_total")].iloc[0]
            data = subset(daily, row.chokepoint, start, end)
            index = data.n_total / row.baseline_mean * 100
            smooth = index.rolling(7, min_periods=7).mean()
            col = COLORS[row.chokepoint]
            ax.plot(index.index, index, color=col, alpha=.15, lw=.6)
            ax.plot(smooth.index, smooth, color=col, lw=1.6, label=row.chokepoint)
            daily_rows.extend(dict(series_id=key, date=str(date.date()), n_total=int(data.loc[date, "n_total"]), index=float(value),
                                   trailing_7day_index=float(smooth.loc[date])) for date, value in index.items())
            if key == "panama_gate":
                # Calendar-date matching deliberately leaves 29 February unmatched.
                prior = []
                for date in data.index:
                    if date.month == 2 and date.day == 29:
                        prior.append(np.nan)
                    else:
                        prev = date - pd.DateOffset(years=1)
                        prior.append(subset(daily, row.chokepoint, prev, prev).n_total.iloc[0] / row.baseline_mean * 100)
                line = pd.Series(prior, index=data.index).rolling(7, min_periods=7).mean()
                ax.plot(line.index, line, ls="--", color="#70777D", lw=1.4, label="Previous-year dates")
        first = results.loc[(results.series_id == keys[0]) & results.metric.eq("n_total")].iloc[0]
        ax.axvspan(pd.Timestamp(first.event_start), pd.Timestamp(first.event_end), color="#AAB3B8", alpha=.19, lw=0)
        ax.axhline(100, color="#60666B", ls=":", lw=.85)
        ax.set_ylabel("Vessel-count index (baseline = 100)")
        ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
        ax.set_ylim(bottom=0)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.04), ncol=1, frameon=False, fontsize=8)
        ax.grid(axis="y", color="#E4E7E9", lw=.6)
        panel_label(ax, label)
    pd.DataFrame(daily_rows).to_csv(OUT / "plotted_daily_indices.csv", index=False)
    ids = [s[0] for s in SERIES]
    labels = ["Suez 2021\nCanal", "Suez 2021\nCape", "Red Sea\nBab el-Mandeb",
              "Red Sea\nSuez", "Red Sea\nCape", "Panama drought\nCanal"]
    y = np.arange(len(ids))
    ax = axes[3]
    for metric, color, shift, marker, name in [("n_total", "#0077A9", -.12, "o", "Vessel count"),
                                             ("capacity", "#248674", .12, "s", "Estimated trade volume")]:
        values = results.loc[results.metric.eq(metric)].set_index("series_id").loc[ids, "change_pct"].to_numpy()
        ax.scatter(values, y + shift, s=37, c=color, marker=marker, label=name, zorder=3)
        for xx, yy in zip(values, y + shift):
            ax.annotate(f"{xx:+.1f}", (xx, yy), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=color)
    ax.axvline(0, color="#7A8186", lw=.8)
    ax.set_yticks(y, labels)
    ax.set_ylim(5.7, -.7)
    ax.set_xlim(-100, 90)
    ax.set_xlabel("Change from stated baseline (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.16), frameon=False, fontsize=9)
    panel_label(ax, "d")
    ax = axes[4]
    metrics = ["n_container", "n_dry_bulk", "n_tanker"]
    matrix = results.pivot(index="series_id", columns="metric", values="change_pct").loc[ids, metrics].to_numpy()
    vmax = max(100, np.ceil(abs(matrix).max() / 20) * 20)
    im = ax.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    ax.set_xticks(range(3), ["Container", "Dry bulk", "Tanker"])
    ax.set_yticks(y, labels)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for iy in range(6):
        for ix in range(3):
            value = matrix[iy, ix]
            ax.text(ix, iy, f"{value:+.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > vmax * .60 else "#222222")
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + .03, .055, pos.width - .06, .017])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Vessel-count change (%)", fontsize=9)
    panel_label(ax, "e")
    ax = axes[5]
    variants = [("Legacy baseline", "o", "#20272B"), ("Preceding 28 days", "s", "#0077A9"),
                ("Preceding 60 days", "^", "#248674"), ("Previous calendar year", "D", "#BA6D28")]
    table = sensitivity.loc[sensitivity.metric.eq("n_total")]
    for i, key in enumerate(ids):
        vals = table.loc[table.series_id.eq(key), "change_pct"]
        ax.plot([vals.min(), vals.max()], [i, i], color="#A9B1B6", lw=1, zorder=1)
    for k, (rule, marker, color) in enumerate(variants):
        values = table.loc[table.baseline_rule.eq(rule)].set_index("series_id").loc[ids, "change_pct"]
        ax.scatter(values, y + (k-1.5)*.09, s=29, marker=marker, color=color,
                   label="Reference window" if rule == "Legacy baseline" else rule, zorder=3)
    ax.axvline(0, color="#7A8186", lw=.8)
    ax.set_yticks(y, labels)
    ax.set_ylim(5.7, -.7)
    ax.set_xlabel("Vessel-count change (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.16), frameon=False, fontsize=8, ncol=2, columnspacing=.7)
    panel_label(ax, "f")
    for ax in [axes[3], axes[5]]:
        ax.grid(axis="x", color="#E4E7E9", lw=.6)
    fig.savefig(OUT / "figures/SupplementaryFig8_PortWatch_Events.pdf")
    fig.savefig(OUT / "figures/SupplementaryFig8_PortWatch_Events.png", dpi=230)
    plt.close(fig)


if __name__ == "__main__":
    daily, results, sensitivity = calculate()
    plot(daily, results, sensitivity)
    print(results.loc[results.metric.eq("n_total"), ["series_id", "baseline_days", "event_days", "change_pct"]].to_string(index=False))
    print("Saved", OUT)
