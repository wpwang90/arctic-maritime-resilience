"""Reproducible categorical raster of the published static port network.

This is route coverage, not observed traffic or vessel eligibility. Source
geometries and port identities are never changed by this visualization.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import requests
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

ROOT = Path(__file__).resolve().parent
NET = ROOT / "data/physical_arctic_embedding/oxmartrans_1390"
OUT = ROOT / "figures/Port_network_1390_raster"
PUB = ROOT / "data/public/network_map_context"
COLORS = ["#ffffff00", "#2368a0", "#54b6c2", "#278457", "#c49630",
          "#df713f", "#943c83", "#8b949d"]
LABELS = ["No network path", "Deep-water sea (>200 m)", "Coastal / shelf sea",
          "River reach", "Lake / reservoir / Caspian", "Canal reach",
          "Lock-controlled reach", "Water type unresolved"]
INSETS = [
    ("b", "Great Lakes - St Lawrence", [-94, 40, -64, 51], .025),
    ("c", "Baltic - Saimaa - Onega", [14, 55, 37, 64], .025),
    ("d", "Caspian - Volga - Don", [38, 36, 56, 51], .025),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_data():
    PUB.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ["ne_10m_bathymetry_K_200", "ne_10m_rivers_lake_centerlines", "ne_10m_lakes"]:
        archive = PUB / (name + ".zip")
        url = "https://naturalearth.s3.amazonaws.com/10m_physical/" + archive.name
        if not archive.exists():
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            archive.write_bytes(response.content)
        if not (PUB / (name + ".shp")).exists():
            with ZipFile(archive) as z:
                z.extractall(PUB)
        files[name] = dict(url=url, sha256=sha(archive), file=str(archive))
    (OUT / "context_sources.json").write_text(json.dumps(files, indent=2))
    return files


def shortest_path_coverage(edges, ports, xy):
    """Count one deterministic shortest path for each unordered port pair."""
    cache = OUT / "all_pair_edge_coverage.npz"
    graph_hash = sha(NET / "physical_edges.parquet")
    if cache.exists():
        v = np.load(cache)
        if str(v["graph_sha256"]) == graph_hash:
            return v["pair_count"]
    n = len(xy)
    graph = coo_matrix((edges.km, (edges.a, edges.b)), shape=(n, n)).tocsr()
    ids = ports.node.to_numpy(int)
    keys = edges.a.to_numpy(np.int64) * n + edges.b.to_numpy(np.int64)
    sorter = np.argsort(keys)
    sorted_keys = keys[sorter]
    counts = np.zeros(len(edges), dtype=np.int64)
    reference = np.load(NET / "static_portpair_distance_matrix.npz")
    assert np.array_equal(reference["port_ids"], ports.key.to_numpy(str))
    expected_distance_sum = 0.0
    for i, source in enumerate(ids[:-1]):
        distance, pred = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
        np.testing.assert_allclose(distance[ids], reference["distance_km"][i], rtol=1e-11, atol=1e-6)
        targets = ids[i+1:]
        assert np.isfinite(distance[targets]).all()
        expected_distance_sum += distance[targets].sum()
        visited = np.zeros(n, dtype=bool)
        visited[source] = True
        order = []
        for target in targets:
            path = []
            node = int(target)
            while not visited[node]:
                visited[node] = True
                path.append(node)
                node = int(pred[node])
                assert node >= 0
            order.extend(reversed(path))
        accum = np.zeros(n, dtype=np.int64)
        np.add.at(accum, targets, 1)
        for node in reversed(order):
            accum[pred[node]] += accum[node]
        assert accum[source] == len(targets)
        order = np.asarray(order, dtype=int)
        parents = pred[order]
        used_keys = np.minimum(order, parents).astype(np.int64)*n + np.maximum(order, parents)
        loc = np.searchsorted(sorted_keys, used_keys)
        np.testing.assert_array_equal(sorted_keys[loc], used_keys)
        np.add.at(counts, sorter[loc], accum[order])
        if i % 100 == 0:
            print("Port-pair route union", i+1, "/", len(ids), "edges", int((counts > 0).sum()), flush=True)
    weighted_sum = float(np.dot(counts, edges.km))
    assert np.isclose(weighted_sum, expected_distance_sum, rtol=1e-10)
    sub = edges[counts > 0]
    network = coo_matrix((np.ones(len(sub)), (sub.a, sub.b)), shape=(n, n)).tocsr()
    _, components = connected_components(network, directed=False)
    assert len(np.unique(components[ids])) == 1
    np.savez_compressed(cache, pair_count=counts, graph_sha256=graph_hash)
    (OUT / "shortest_path_audit.json").write_text(json.dumps(dict(
        graph_sha256=graph_hash, port_count=len(ids), unordered_pairs=len(ids)*(len(ids)-1)//2,
        selected_edges=len(sub), available_edges=len(edges), all_ports_in_one_component=True,
        total_pair_distance_km=expected_distance_sum, edge_count_weighted_distance_km=weighted_sum,
        interpretation="One static shortest path per unordered pair; equal pairs, no trade weights; tie alternatives not enumerated"), indent=2))
    return counts


def is_lock(control):
    tags = control.get("tags", {})
    return tags.get("lock") == "yes" or tags.get("waterway") == "lock_gate"


def contains_any(geometries, points):
    result = np.zeros(len(points), dtype=bool)
    if not len(geometries):
        return result
    shapely.prepare(geometries)
    tree = shapely.STRtree(geometries)
    for start in range(0, len(points), 150000):
        subset = points[start:start+150000]
        pairs = tree.query(subset)
        valid = shapely.intersects(geometries[pairs[1]], subset[pairs[0]])
        result[start+np.unique(pairs[0, valid])] = True
    return result


def prepare_samples(edges, geometry, offsets):
    cache = OUT / "classified_route_samples.parquet"
    cache_meta = OUT / "sample_cache_manifest.json"
    signature = dict(version=3, graph_sha256=sha(NET / "physical_edges.parquet"),
                     geometry_sha256=sha(NET / "edge_geometry.npz"))
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text()) == signature:
        return pd.read_parquet(cache)
    chunks = []
    for row in edges.itertuples():
        p = geometry[offsets[row.geometry_id]:offsets[row.geometry_id+1]].copy()
        # Interpolate the existing polyline only; wrap across the dateline.
        p[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(p[:, 0])))
        a, b = p[:-1], p[1:]
        steps = np.maximum(1, np.ceil(np.max(np.abs(b-a), axis=1)/.0125).astype(int))
        idx = np.repeat(np.arange(len(a)), steps)
        local = np.arange(len(idx)) - np.repeat(np.cumsum(steps)-steps, steps)
        fraction = (local + .5) / steps[idx]
        q = a[idx] + (b-a)[idx] * fraction[:, None]
        q[:, 0] = (q[:, 0]+180) % 360 - 180
        chunks.append(np.c_[q, np.full(len(q), row.Index)])
    v = np.vstack(chunks)
    frame = pd.DataFrame(dict(lon=v[:, 0], lat=v[:, 1], edge_index=v[:, 2].astype(int)))
    point = shapely.points(v[:, :2])
    classes = np.full(len(v), 2, dtype=np.uint8)
    land = gpd.read_file(ROOT / "data/public/naturalearth_10m/ne_10m_land.shp")
    inland = contains_any(land.geometry.to_numpy(), point)
    classes[inland] = 7
    deep = gpd.read_file(PUB / "ne_10m_bathymetry_K_200.shp")
    classes[contains_any(deep.geometry.to_numpy(), point) & ~inland] = 1
    lakes = gpd.read_file(ROOT / "data/public/gshhg/GSHHS_shp/f/GSHHS_f_L2.shp")
    classes[contains_any(lakes.geometry.to_numpy(), point)] = 4

    # Coarse river matching is classification only, never a routing buffer.
    rivers = gpd.read_file(PUB / "ne_10m_rivers_lake_centerlines.shp")
    river_polys = shapely.buffer(rivers.geometry.to_numpy(), .02)
    classes[contains_any(river_polys, point) & (classes == 7)] = 3
    source_files = []
    for f in sorted((ROOT / "data/public/inland_navigation").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["class", "geometry"])
        shapes = shapely.from_wkb(d.geometry.to_numpy())
        source_files.append(dict(file=str(f), sha256=sha(f)))
        for label, code in [("lake", 4), ("reservoir", 4), ("river", 3), ("canal", 5)]:
            arr = shapes[d["class"].eq(label).to_numpy()]
            arr = shapely.make_valid(arr)
            line = shapely.get_type_id(arr) == 1
            arr[line] = shapely.buffer(arr[line], .00002)
            classes[contains_any(arr, point)] = code
        print("Classified water types", f.name, flush=True)

    # Source-tagged transport canals outrank broad water-surface categories.
    edge_class = np.zeros(int(edges.index.max())+1, dtype=np.uint8)
    lock_points, long_control_geometries, long_canal_geometries = [], [], []
    for row in edges.itertuples():
        passage = str(row.passage).lower()
        if passage in ["panama", "suez"]:
            edge_class[row.Index] = 5
        controls = json.loads(row.navigation_controls_json) if isinstance(row.navigation_controls_json, str) else []
        if any(c.get("tags", {}).get("waterway") == "canal" for c in controls):
            if row.km <= 2:
                edge_class[row.Index] = 5
            else:
                for c in controls:
                    if c.get("tags", {}).get("waterway") != "canal" or not c.get("points"):
                        continue
                    q = np.asarray(c["points"])
                    feature = shapely.Point(q[0]) if len(q) == 1 else shapely.LineString(q)
                    long_canal_geometries.append(feature.buffer(.002))
        if any(is_lock(c) for c in controls):
            p = geometry[offsets[row.geometry_id]:offsets[row.geometry_id+1]]
            if row.km <= 2:
                edge_class[row.Index] = 6
                q = p[len(p)//2]
                lock_points.append(dict(lon=float(q[0]), lat=float(q[1]), edge_index=int(row.Index)))
            else:
                # A lock on a long access edge does not make its full length a lock.
                for c in controls:
                    if not is_lock(c) or not c.get("points"):
                        continue
                    q = np.asarray(c["points"])
                    feature = shapely.Point(q[0]) if len(q) == 1 else shapely.LineString(q)
                    long_control_geometries.append(feature.buffer(.002))
                    center = np.mean(q, axis=0)
                    lock_points.append(dict(lon=float(center[0]), lat=float(center[1]), edge_index=int(row.Index)))
    override = edge_class[frame.edge_index.to_numpy()]
    classes[override > 0] = override[override > 0]
    if long_canal_geometries:
        hit = contains_any(np.array(long_canal_geometries, dtype=object), point)
        classes[hit & (classes != 6)] = 5
    if long_control_geometries:
        classes[contains_any(np.array(long_control_geometries, dtype=object), point)] = 6
    frame["class_code"] = classes
    frame.to_parquet(cache, index=False)
    cache_meta.write_text(json.dumps(signature, indent=2))
    pd.DataFrame(lock_points).to_csv(OUT / "documented_lock_reaches.csv", index=False)
    (OUT / "regional_classification_sources.json").write_text(json.dumps(source_files, indent=2))
    return frame


def raster(samples, bounds, step):
    x0, y0, x1, y1 = bounds
    width, height = int(np.ceil((x1-x0)/step)), int(np.ceil((y1-y0)/step))
    inside = samples.lon.ge(x0) & samples.lon.lt(x1) & samples.lat.ge(y0) & samples.lat.lt(y1)
    p = samples[inside]
    col = np.floor((p.lon.to_numpy()-x0)/step).astype(int)
    row = np.floor((p.lat.to_numpy()-y0)/step).astype(int)
    presence = np.zeros((7, height, width), dtype=bool)
    presence[p.class_code.to_numpy()-1, row, col] = True
    display = np.zeros((height, width), dtype=np.uint8)
    # Keep all bands; display gives small control features priority, not weight.
    for code in [7, 1, 2, 4, 3, 5, 6]:
        display[presence[code-1]] = code
    return display, presence


def save_grid(name, display, presence, bounds, step):
    np.savez_compressed(OUT / (name + ".npz"), display_class=display,
                        class_presence=presence, bounds=np.array(bounds), resolution_degrees=step,
                        class_names=np.array(LABELS[1:]), crs="EPSG:4326", row_order="south_to_north")
    # ESRI ASCII grid is portable and keeps the categorical integer values.
    with (OUT / (name + ".asc")).open("w") as stream:
        stream.write(f"ncols {display.shape[1]}\nnrows {display.shape[0]}\nxllcorner {bounds[0]}\nyllcorner {bounds[1]}\ncellsize {step}\nNODATA_value 0\n")
        np.savetxt(stream, display[::-1], fmt="%d")
    (OUT / (name + ".prj")).write_text('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')


def annotate(ax, label, xy, offset, color="#454d55"):
    ax.annotate(label, xy, xytext=offset, textcoords="offset points", fontsize=7.7,
                color=color, ha="left", va="center", zorder=7,
                bbox=dict(facecolor="white", alpha=.88, edgecolor="none", pad=1.5),
                arrowprops=dict(arrowstyle="-", color=color, lw=.55))


def draw_panel(ax, bounds, step, display, land, lakes, ports, locks, letter, global_map=False):
    x0, y0, x1, y1 = bounds
    ax.set_facecolor("#fbfcfd")
    box = shapely.box(x0, y0, x1, y1)
    land.clip(box).plot(ax=ax, color="#e9edef", edgecolor="#adb7bd", linewidth=.28, zorder=1)
    sub = lakes.clip(box)
    if len(sub):
        sub.plot(ax=ax, color="#fbfcfd", edgecolor="#bac5cc", linewidth=.25, zorder=2)
    ax.imshow(display, origin="lower", extent=[x0, x0+display.shape[1]*step, y0, y0+display.shape[0]*step],
              cmap=ListedColormap(COLORS), norm=BoundaryNorm(np.arange(-.5, 8, 1), 8),
              interpolation="nearest", zorder=3, rasterized=True, aspect="auto")
    p = ports[ports.water_lon.between(x0, x1) & ports.water_lat.between(y0, y1)]
    ax.scatter(p.water_lon, p.water_lat, s=2.3 if global_map else 10, c="#293640",
               edgecolors="white", linewidths=.15 if global_map else .3, zorder=5, alpha=.7)
    if not global_map:
        q = locks[locks.lon.between(x0, x1) & locks.lat.between(y0, y1)]
        ax.scatter(q.lon, q.lat, s=13, marker="s", c=COLORS[6], edgecolors="white", linewidths=.25, zorder=6)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("auto")
    ax.tick_params(labelsize=8, length=3, color="#8a949c")
    ax.grid(color="#d2d9de", linewidth=.35, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#aab4ba"); spine.set_linewidth(.6)
    ax.text(-.035, 1.035, letter, transform=ax.transAxes, fontsize=16, weight="bold")
    if global_map:
        ax.set_xticks(np.arange(-180, 181, 60), ["180 W", "120 W", "60 W", "0", "60 E", "120 E", "180 E"])
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], ["90 S", "60 S", "30 S", "0", "30 N", "60 N", "90 N"])
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)


def draw_figures(samples, ports):
    land = gpd.read_file(ROOT / "data/public/naturalearth_10m/ne_10m_land.shp")
    lakes = gpd.read_file(PUB / "ne_10m_lakes.shp")
    locks = pd.read_csv(OUT / "documented_lock_reaches.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    world_bounds = [-180, -90, 180, 90]
    display, presence = raster(samples, world_bounds, .25)
    save_grid("Global_network_025deg", display, presence, world_bounds, .25)
    stats = [{"class_code": i, "type": LABELS[i], "occupied_025deg_cells": int(presence[i-1].sum())} for i in range(1, 8)]
    pd.DataFrame(stats).to_csv(OUT / "grid_class_summary.csv", index=False)
    legend = [Patch(facecolor=COLORS[i], label=LABELS[i], edgecolor="none") for i in range(1, 8)]
    legend += [Line2D([], [], color="#293640", marker="o", ls="", markersize=3, label="Port entry")]
    fig = plt.figure(figsize=(17, 12.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.8, 1], left=.048, right=.985,
                          top=.962, bottom=.135, hspace=.29, wspace=.18)
    ax = fig.add_subplot(gs[0, :])
    draw_panel(ax, world_bounds, .25, display, land, lakes, ports, locks, "a", True)
    for letter, name, bounds, step in INSETS:
        ax.add_patch(Rectangle((bounds[0], bounds[1]), bounds[2]-bounds[0], bounds[3]-bounds[1],
                               fill=False, edgecolor="#59646b", lw=.7, ls="--", zorder=6))
        ax.text(bounds[0]+1, bounds[3]+2, letter, weight="bold", fontsize=10, zorder=7,
                bbox=dict(fc="white", ec="none", pad=.4))
    for label, xy, offset in [("Panama", (-79.7, 9.1), (-65, -28)), ("Suez", (32.4, 30.5), (-12, -43)),
                               ("Malacca", (103, 1.3), (8, -27)), ("Bering", (-168, 66), (6, 12))]:
        annotate(ax, label, xy, offset)
    for i, (letter, name, bounds, step) in enumerate(INSETS):
        panel = fig.add_subplot(gs[1, i])
        d, bands = raster(samples, bounds, step)
        save_grid("Region_" + letter + "_0025deg", d, bands, bounds, step)
        draw_panel(panel, bounds, step, d, land, lakes, ports, locks, letter)
        if letter == "b":
            annotate(panel, "Welland", (-79.21, 43.1), (-40, -32))
            annotate(panel, "St Lawrence locks", (-74.2, 45.15), (8, -20))
            annotate(panel, "Great Lakes", (-87, 46.5), (-45, 18))
        elif letter == "c":
            annotate(panel, "Sodertalje", (17.63, 59.19), (-35, -34))
            annotate(panel, "Saimaa lock chain", (28.5, 61), (-87, 24))
            annotate(panel, "Onega", (34.5, 61.5), (-25, 24))
        else:
            annotate(panel, "Volga-Don locks", (44.5, 48.5), (-72, 15))
            annotate(panel, "Don", (40.7, 47.5), (-20, -23))
            annotate(panel, "Volga", (47, 46.5), (12, 10))
            annotate(panel, "Caspian Sea", (50.8, 41), (-40, -20))
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.51, .065), ncol=4,
               frameon=False, fontsize=9, handlelength=1.4, columnspacing=2)
    fig.text(.048, .04, "1,390 source records | 1,386 unique ports | All 959,805 unordered port pairs | a: 0.25-degree cells; b-d: 0.025-degree cells", fontsize=9, color="#3e4c55")
    fig.text(.048, .019, "Static shortest-path coverage, not AIS intensity. Cells show route presence, not channel width. Depth and control classes do not certify vessel passage.", fontsize=8.5, color="#59666f")
    for suffix, dpi in [("png", 250), ("pdf", 250)]:
        fig.savefig(OUT / ("Port_network_1390_classified_raster." + suffix), dpi=dpi)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(18, 9.8))
    fig.subplots_adjust(left=.05, right=.98, top=.94, bottom=.18)
    draw_panel(ax, world_bounds, .25, display, land, lakes, ports, locks, "a", True)
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .055), ncol=4, frameon=False)
    fig.text(.05, .025, "All-port static route union; 0.25-degree cells. Depth >200 m is a generalized Natural Earth map class, not a navigational depth guarantee.", fontsize=9)
    fig.savefig(OUT / "Global_network_1390_raster.png", dpi=250)
    fig.savefig(OUT / "Global_network_1390_raster.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ports = pd.read_csv(NET / "canonical_port_registry.csv")
    if args.plot_only:
        draw_figures(pd.read_parquet(OUT / "classified_route_samples.parquet"), ports)
        return
    context_data()
    edges = pd.read_parquet(NET / "physical_edges.parquet")
    validation = json.loads((NET / "port_identity/final_graph_validation.json").read_text())
    assert validation["graph_sha256"] == sha(NET / "physical_edges.parquet")
    assert len(pd.read_csv(NET / "reference_to_main_port.csv")) == 1390
    assert ports.connected_to_global_water_component.all()
    xy = np.load(NET / "node_coordinates.npy")
    counts = shortest_path_coverage(edges, ports, xy)
    edges["static_port_pair_count"] = counts
    edges.to_parquet(OUT / "all_edge_pair_coverage.parquet", index=False)
    selected = edges[counts > 0]
    archive = np.load(NET / "edge_geometry.npz")
    samples = prepare_samples(selected, archive["geometry"], archive["offsets"])
    draw_figures(samples, ports)
    manifest = dict(graph_sha256=sha(NET / "physical_edges.parquet"),
                    geometry_sha256=sha(NET / "edge_geometry.npz"), source_records=1390,
                    unique_ports=len(ports), route_edges=len(selected),
                    classes={str(i): LABELS[i] for i in range(1, 8)},
                    routing_inputs_changed=False, observed_traffic_density=False,
                    unrestricted_vessel_passage=False, sample_step_max_degrees=.0125,
                    raster_display_priority=[6, 5, 3, 4, 2, 1, 7],
                    multitype_cells_preserved_in_npz=True)
    (OUT / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
