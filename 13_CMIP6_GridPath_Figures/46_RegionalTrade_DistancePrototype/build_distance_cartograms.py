"""Distance-change cartograms with audited, comparable country-area targets."""
from pathlib import Path
import hashlib
import importlib
import json

import cartogram
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results/cartograms"
FIG = HERE / "figures/cartograms"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SOURCE = HERE / "results/country_origin_distances.csv"
SHIPS = ["all", "container", "bulk", "tanker", "lng"]
NAMES = dict(all="All vessel types", container="Container", bulk="Dry bulk", tanker="Tanker", lng="LNG")
GATES = ["Open network", "+ Bab el-Mandeb", "+ Makassar", "+ Malacca", "+ Panama", "+ Sunda"]
COLORS = ["#316d96", "#5796b6", "#a5cbd9", "#f8dfc4", "#efb279", "#dd884f", "#c55b3e", "#9d3d36", "#702e37"]
BOUNDS = [-50, -20, -10, 0, 10, 25, 50, 100, 200, 350]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(BOUNDS, len(COLORS))
BASE_COLOR = "#d6e2e8"
MOD = importlib.import_module("cartogram.cartogram")
MOD.NUM_THREADS = 1
METHOD_VERSION = "components_country_dissolve_v3"
REGION_COLORS = {"North America": "#d7ae68", "South America": "#72aa98", "Europe": "#809bc3", "Africa": "#b8bd7a", "Asia": "#ce8a92", "Oceania": "#89b9c8", "Antarctica": "#dddddd", "Seven seas (open ocean)": "#aaaaaa"}


class VectorCartogram(cartogram.Cartogram):
    """Vectorized evaluation of the library's unchanged DCN1985 vertex forces."""
    def _transform_vertices(self, vertices, features, reduction_factor):
        feature = np.asarray(features, float)
        result = np.empty_like(vertices)
        feature = feature[feature[:, 2] != 0]
        if not len(feature):
            return vertices.copy()
        centers, mass, radius = feature[:, :2], feature[:, 2], feature[:, 3]
        for start in range(0, len(vertices), 2048):
            part = vertices[start:start + 2048]
            delta = part[:, None, :] - centers[None, :, :]
            distance = np.linalg.norm(delta, axis=2)
            dr = distance / radius
            force = np.where(distance > radius, mass * radius / np.maximum(distance, 1e-20),
                             mass * dr * dr * (4 - 3 * dr))
            coefficient = reduction_factor * force / np.maximum(distance, 1e-20)
            result[start:start + len(part)] = part + np.sum(delta * coefficient[:, :, None], axis=1)
        return result


def verify_vectorization():
    rng = np.random.default_rng(14)
    features = [MOD.CartogramFeature(*row) for row in np.column_stack([rng.normal(size=(12, 2)), rng.normal(size=12) * .2, rng.uniform(.1, 2, 12)])]
    vertices = rng.normal(size=(80, 2))
    obj = VectorCartogram.__new__(VectorCartogram)
    expected = np.asarray([MOD.Cartogram._transform_vertex(obj, point, features, .67) for point in vertices])
    actual = obj._transform_vertices(vertices, features, .67)
    error = float(np.max(np.abs(expected - actual)))
    assert error < 1e-12
    return error


def base_world():
    world = gpd.read_file(ROOT / "data/public/naturalearth_10m_countries/ne_10m_admin_0_countries.shp")
    world["iso3"] = world.ISO_A3_EH.where(world.ISO_A3_EH.ne("-99"), world.ISO_A3)
    world["iso3"] = world.iso3.where(world.iso3.ne("-99"), world.ADM0_A3)
    world = world[["iso3", "NAME_EN", "CONTINENT", "geometry"]].to_crs(8857)
    geometry = shapely.make_valid(world.geometry.array)
    assert shapely.coverage_is_valid(geometry)
    geometry = shapely.coverage_simplify(geometry, 20000)
    assert shapely.coverage_is_valid(geometry)
    # Coordinates become plotting units, not a physical CRS after deformation.
    world.geometry = shapely.transform(geometry, lambda coordinates: coordinates / 1e6)
    world = world.set_crs(None, allow_override=True)
    world = world.dissolve(by="iso3", aggfunc="first").reset_index()
    world["reference_area"] = shapely.area(world.geometry.array)
    return world


def component_targets(data):
    rows = []
    for index, row in data.iterrows():
        parts = list(row.geometry.geoms) if row.geometry.geom_type == "MultiPolygon" else [row.geometry]
        areas = np.array([part.area for part in parts])
        major = np.flatnonzero(areas >= .002).tolist()
        if not major:
            major = [int(np.argmax(areas))]
        groups = {i: [parts[i]] for i in major}
        centers = np.array([[parts[i].centroid.x, parts[i].centroid.y] for i in major])
        for i, part in enumerate(parts):
            if i in groups:
                continue
            center = np.array([part.centroid.x, part.centroid.y])
            nearest = major[int(np.argmin(np.linalg.norm(centers - center, axis=1)))]
            groups[nearest].append(part)
        for components in groups.values():
            geometry = shapely.union_all(components)
            ratio = row.distance_ratio if np.isfinite(row.distance_ratio) else 1.
            rows.append(dict(parent_index=index, target_area=geometry.area * ratio, geometry=geometry))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)


def distort(world, country, ship, step, scenario):
    stem = f"{ship}_step{step}_{scenario}"
    cache = OUT / (stem + ".parquet")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    metadata_path = OUT / (stem + ".json")
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        converged_enough = metadata.get("trade_weighted_area_error_pct", 100) < .2 or metadata.get("iterations", 0) >= 100
        if metadata["source_sha256"] == source_hash and metadata.get("method_version") == METHOD_VERSION and converged_enough:
            return gpd.read_parquet(cache), metadata
    records = country[country.shiptype.eq(ship) & country.step.eq(step)].set_index("origin_iso3")
    data = world.copy()
    column = "baseline_km" if scenario == "baseline" else scenario + "_km"
    data["distance_km"] = data.iso3.map(records[column])
    data["baseline_km"] = data.iso3.map(records.baseline_km)
    data["matched_od_count"] = data.iso3.map(records.matched_od_count)
    data["matched_trade_pct"] = data.iso3.map(records.matched_trade_pct)
    data["matched_trade_usd"] = data.iso3.map(records.matched_trade_usd)
    data["distance_ratio"] = data.distance_km / data.baseline_km
    data["change_pct"] = 100 * (data.distance_ratio - 1)
    data["has_data"] = data.distance_ratio.notna()
    # Unobserved areas are neutral geometry anchors, never filled economic estimates.
    data["target_area"] = data.reference_area * data.distance_ratio.fillna(1)
    if scenario == "baseline" or np.allclose(data.distance_ratio.fillna(1), 1):
        result = data
        iterations, error, scale = 0, 0., 1.
    else:
        components = component_targets(data)
        cart = VectorCartogram(components, "target_area", max_iterations=100, max_average_error=.0005)
        iterations, error = int(cart.iteration), float(cart.average_error)
        transformed = gpd.GeoDataFrame(cart, geometry="geometry", crs=None)
        result = data.copy()
        for index, group in transformed.groupby("parent_index"):
            result.loc[index, "geometry"] = shapely.union_all(group.geometry.array)
        # Undo per-map total-area normalization so all states share an absolute scale.
        scale = float(np.sqrt(data.target_area.sum() / shapely.area(result.geometry.array).sum()))
        result.geometry = shapely.transform(result.geometry.array, lambda xy: xy * scale)
    result["realized_area"] = shapely.area(result.geometry.array)
    result["area_relative_error"] = (result.realized_area / result.target_area - 1).abs()
    assert shapely.is_valid(result.geometry.array).all()
    assert not result.geometry.is_empty.any()
    assert np.isfinite(shapely.get_coordinates(result.geometry.array)).all()
    union_area = float(shapely.area(shapely.union_all(result.geometry.array)))
    overlap_pct = 100 * max(0., float(result.realized_area.sum()) - union_area) / union_area
    observed = result[result.has_data]
    weighted_error = float(np.average(observed.area_relative_error, weights=observed.matched_trade_usd.fillna(0)))
    metadata = dict(shiptype=ship, step=step, scenario=scenario, source_sha256=source_hash, method_version=METHOD_VERSION,
                    iterations=iterations, library_average_error=error,
                    final_total_area_scale=scale,
                    trade_weighted_area_error_pct=100 * weighted_error,
                    median_country_area_error_pct=100 * float(observed.area_relative_error.median()),
                    max_country_area_error_pct=100 * float(observed.area_relative_error.max()),
                    overlap_area_pct=overlap_pct, all_geometries_valid=True,
                    target="reference Equal Earth area * (country weighted distance / country open-network distance)",
                    missing="Neutral area target (ratio=1) for geometry only; hatched as no data, no distance imputation.")
    result.to_parquet(cache)
    result.drop(columns="geometry").to_csv(OUT / (stem + "_targets.csv"), index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(stem, "iterations", iterations, "trade-weighted area error %", round(100 * weighted_error, 3), "overlap %", round(overlap_pct, 5), flush=True)
    return result, metadata


def plot_map(ax, base, shaped, bounds, labels=False, region_style=False, label_codes=None):
    ax.set_facecolor("white" if region_style else "#f2f6f8")
    base.boundary.plot(ax=ax, color="#acb9c1", linewidth=.35, linestyle=":", zorder=1)
    missing = ~shaped.has_data
    if missing.any():
        shaped[missing].plot(ax=ax, facecolor="#fafafa", edgecolor="#b5c0c6", linewidth=.25, hatch="///", zorder=2)
    observed = shaped[shaped.has_data]
    if region_style:
        observed.plot(ax=ax, color=observed.CONTINENT.map(REGION_COLORS), edgecolor="white", linewidth=.35, zorder=3)
        observed.boundary.plot(ax=ax, color="#677b84", linewidth=.18, zorder=4)
    else:
        observed.plot(ax=ax, column="change_pct", cmap=CMAP, norm=NORM, edgecolor="#617b86", linewidth=.27, zorder=3)
    zero = observed[observed.change_pct.abs().lt(1e-6)]
    if len(zero) and not region_style:
        zero.plot(ax=ax, facecolor=BASE_COLOR, edgecolor="#617b86", linewidth=.27, zorder=4)
    if labels:
        for iso in (label_codes or ["CHN", "USA", "RUS", "BRA", "AUS", "IND", "CAN"]):
            sub = observed[observed.iso3.eq(iso)]
            if not len(sub):
                continue
            row = sub.iloc[0]
            parts = list(row.geometry.geoms) if row.geometry.geom_type == "MultiPolygon" else [row.geometry]
            point = max(parts, key=lambda p: p.area).representative_point()
            label = f"{iso}\n{row.distance_km / 1000:.1f}k km"
            if region_style:
                label = f"{iso} {row.distance_km / 1000:.1f}k\n{row.change_pct:+.1f}%"
            if iso == "IND" and region_style:
                point = shapely.Point(point.x - .3, point.y - .25)
            text = ax.text(point.x, point.y, label, ha="center", va="center", fontsize=7, color="#243d4b", zorder=6)
            text.set_path_effects([pe.withStroke(linewidth=2, foreground="white", alpha=.94)])
    ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal", xticks=[], yticks=[])
    ax.set_xlabel(""); ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(.45); spine.set_color("#c0cbd0")
        if region_style:
            spine.set_visible(False)


def shared_bounds(frames):
    bb = np.array([f.total_bounds for f in frames])
    xmin, ymin = bb[:, :2].min(axis=0)
    xmax, ymax = bb[:, 2:].max(axis=0)
    pad = .02 * (xmax - xmin)
    return [xmin - pad, xmax + pad, ymin - pad, ymax + pad]


def common_legend(fig, y=.11, width=.57):
    cbax = fig.add_axes([(.99 - width) / 2, y, width, .023])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cbax, orientation="horizontal", spacing="uniform")
    cb.set_ticks(BOUNDS)
    cb.set_label("Trade-weighted distance change from the open conventional network (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(.3)
    handles = [Patch(facecolor=BASE_COLOR, edgecolor="#617b86", label="No distance change"),
               Patch(facecolor="#fafafa", hatch="///", edgecolor="#b5c0c6", label="No comparable sample")]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(.5, .012), fontsize=9)


def save(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"{stem}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)
    print("Saved", stem, flush=True)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": .25})
    vector_error = verify_vectorization()
    country = pd.read_csv(SOURCE)
    world = base_world()
    frames, metadata = {}, []
    for ship in SHIPS:
        choices = [(0, "baseline")] + [(k, scenario) for k in (range(6) if ship == "all" else [5]) for scenario in ["traditional", "with_arctic"]]
        for step, scenario in choices:
            shaped, audit = distort(world, country, ship, step, scenario)
            frames[(ship, step, scenario)] = shaped
            metadata.append(audit)
    # A single coordinate frame and scale for every display, including vessel-specific views.
    bounds = shared_bounds(list(frames.values()))
    for ship in SHIPS:
        fig = plt.figure(figsize=(19, 7.2))
        gs = fig.add_gridspec(1, 3, left=.015, right=.985, bottom=.25, top=.83, wspace=.035)
        cases = [(0, "baseline", "Open conventional network"), (5, "traditional", "Five closures: traditional rerouting"), (5, "with_arctic", "Five closures: Arctic option")]
        for j, (step, scenario, title) in enumerate(cases):
            ax = fig.add_subplot(gs[j])
            plot_map(ax, world, frames[(ship, step, scenario)], bounds, labels=True)
            ax.text(0, 1.08, chr(97 + j), transform=ax.transAxes, fontsize=15, weight="bold")
            ax.text(.047, 1.085, title, transform=ax.transAxes, fontsize=10.5)
        fig.text(.02, .95, NAMES[ship] + " | Trade-weighted distance cartograms", fontsize=14, weight="bold")
        fig.text(.02, .902, "Target country area / reference area = weighted route distance / open-network distance", fontsize=10, color="#49606c")
        common_legend(fig, .12)
        save(fig, f"Distance_Cartogram_Comparison_{ship}")
    fig = plt.figure(figsize=(19, 9.2))
    gs = fig.add_gridspec(2, 3, left=.015, right=.985, bottom=.18, top=.87, wspace=.035, hspace=.18, height_ratios=[2, 1.1])
    cases = [(0, "baseline", "Open conventional network"), (5, "traditional", "Five closures: traditional rerouting"), (5, "with_arctic", "Five closures: Arctic option")]
    european_boxes = []
    for step, scenario, _ in cases:
        frame = frames[("all", step, scenario)]
        for geometry in frame[frame.iso3.isin(["GBR", "FRA", "DEU", "NLD", "BEL", "ESP", "ITA", "POL", "DNK", "CHE", "AUT"])].geometry:
            parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
            european_boxes.append(max(parts, key=lambda p: p.area).bounds)
    bb = np.asarray(european_boxes)
    europe_bounds = [bb[:, 0].min() - .4, bb[:, 2].max() + .4, bb[:, 1].min() - .2, bb[:, 3].max() + .25]
    for j, (step, scenario, title) in enumerate(cases):
        ax = fig.add_subplot(gs[0, j])
        plot_map(ax, world, frames[("all", step, scenario)], bounds, labels=True, region_style=True)
        ax.text(0, 1.05, chr(97 + j), transform=ax.transAxes, fontsize=15, weight="bold")
        ax.text(.047, 1.055, title, transform=ax.transAxes, fontsize=10.5)
        inset = fig.add_subplot(gs[1, j])
        plot_map(inset, world, frames[("all", step, scenario)], europe_bounds, labels=True, region_style=True,
                 label_codes=["GBR", "FRA", "DEU", "ITA", "ESP"])
        inset.text(0, 1.04, "Europe detail", transform=inset.transAxes, fontsize=9, color="#49606c")
    fig.text(.02, .95, "Trade-weighted distance reshapes the shipping world", fontsize=15, weight="bold")
    fig.text(.02, .915, "Identical regional colours across states; country-area changes encode distance ratios, without visual amplification.", fontsize=10, color="#49606c")
    handles = [Patch(facecolor=REGION_COLORS[r], edgecolor="white", label=r) for r in ["North America", "South America", "Europe", "Africa", "Asia", "Oceania"]]
    handles.append(Patch(facecolor="#fafafa", edgecolor="#b5c0c6", hatch="///", label="No comparable sample"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .096), ncol=7, frameon=False, fontsize=9)
    fig.text(.5, .064, "Area ratio = trade-weighted route distance / open-network distance. Labels: 1,000 km and change from baseline.", ha="center", fontsize=9, color="#49606c")
    fig.text(.5, .032, "Layout reference: Dorling, PLOS Medicine (2007), Figs. 1-2. Geometry is computed independently from this study's data.", ha="center", fontsize=8, color="#63757e")
    save(fig, "Distance_Cartogram_ArticleStyle")
    fig = plt.figure(figsize=(15, 20.5))
    gs = fig.add_gridspec(6, 2, left=.12, right=.985, bottom=.12, top=.94, wspace=.035, hspace=.075)
    for k in range(6):
        for j, scenario in enumerate(["traditional", "with_arctic"]):
            ax = fig.add_subplot(gs[k, j])
            plot_map(ax, world, frames[("all", k, scenario)], bounds)
            if k == 0:
                ax.text(.5, 1.09, ["Traditional only", "Arctic option"][j], transform=ax.transAxes, ha="center", fontsize=12, weight="bold")
            if j == 0:
                ax.text(-.022, .57, f"{k} closure" + ("s" if k != 1 else ""), transform=ax.transAxes, ha="right", fontsize=10, weight="bold")
                ax.text(-.022, .37, GATES[k].replace("+ ", "+\n"), transform=ax.transAxes, ha="right", fontsize=9)
    common_legend(fig, .065, .66)
    save(fig, "Distance_Cartogram_ClosureProgression")
    audit = dict(method="Dougenik, Chrisman and Niemeyer (1985), python-cartogram 1.0.2",
                 algorithm_reference="https://python-cartogram.readthedocs.io/reference.html",
                 vectorization_max_absolute_error=vector_error, maps=metadata, common_bounds=bounds,
                 cautions="Area encodes relative mean route length, not physical land area, trade volume, economic losses, or distance between points on the deformed map. Existing sample and single closure sequence only.")
    (OUT / "cartogram_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
