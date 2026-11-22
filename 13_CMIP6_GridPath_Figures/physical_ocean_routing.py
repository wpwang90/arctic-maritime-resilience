"""Land-checked maritime geometry; graph lengths are physical kilometres.

Observed AIS demand and observed trajectories are not rewritten. Public
maritime-network edges and new ocean-grid connectors are modelled geometry.
No distance cap is applied to port or Arctic gateway access.
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path
from functools import lru_cache

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.ops import nearest_points

from chokepoints24 import HERE, EARTH_KM, xyz
from analyze_ethan_trajectory_arctic_embedding import haversine_vec

OUT = HERE / "data/physical_arctic_embedding"
OUT.mkdir(parents=True, exist_ok=True)


def normalise(points):
    points = np.asarray(points, dtype=float).copy()
    lon=points[...,0]
    # Preserve in-range source vertices bit-for-bit at exact shoreline junctions.
    points[..., 0] = np.where((lon < -180) | (lon >= 180),(lon + 180) % 360 - 180,lon)
    return points


def length(points):
    p = np.asarray(points)
    return float(haversine_vec(p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]).sum()) if len(p) > 1 else 0.0


def densify(a, b, max_km=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    u, v = xyz(np.asarray([a, b]))
    angle = np.arccos(np.clip(u @ v, -1, 1))
    n = max(1, int(np.ceil(angle * EARTH_KM / max_km)))
    if angle < 1e-10:
        return np.asarray([a, b])
    t = np.linspace(0, 1, n + 1)[:, None]
    q = (np.sin((1 - t) * angle) * u + np.sin(t * angle) * v) / np.sin(angle)
    return normalise(np.c_[np.degrees(np.arctan2(q[:, 1], q[:, 0])), np.degrees(np.arcsin(q[:, 2]))])


def split_dateline(points):
    p = normalise(points)
    if not len(p):
        return []
    pieces, current = [], [p[0].tolist()]
    for a, b in zip(p[:-1], p[1:]):
        delta = (b[0] - a[0] + 180) % 360 - 180
        if abs(b[0] - a[0]) > 180:
            boundary = 180 if delta > 0 else -180
            lat = a[1] + (boundary - a[0]) / delta * (b[1] - a[1])
            current.append([boundary, float(lat)])
            pieces.append(current)
            current = [[-boundary, float(lat)], b.tolist()]
        else:
            current.append(b.tolist())
    pieces.append(current)
    return [part for part in pieces if len(part) > 1]


def exact_overlay_window(water, geometry):
    """Clip irrelevant vertices for overlay, without simplifying any shoreline."""
    if geometry.is_empty or shapely.get_num_coordinates(water)<200000:
        return water
    xmin,ymin,xmax,ymax=geometry.bounds
    wxmin,wymin,wxmax,wymax=water.bounds
    if (xmax-xmin)*(ymax-ymin) > .25*(wxmax-wxmin)*(wymax-wymin):
        return water
    # Padding is only for the clipping rectangle; it never enlarges water.
    return shapely.intersection(water,shapely.box(xmin-1e-9,ymin-1e-9,xmax+1e-9,ymax+1e-9))


def covers_components(water,geometry):
    """Exact coverage without GEOS 3.11's costly boundary-relate fallback."""
    if geometry.is_empty:
        return False
    if geometry.geom_type=="GeometryCollection":
        return all(covers_components(water,part) for part in geometry.geoms if not part.is_empty)
    if geometry.geom_type in ("Point","MultiPoint"):
        return bool(shapely.covers(water,geometry))
    xmin,ymin,xmax,ymax=geometry.bounds
    wxmin,wymin,wxmax,wymax=water.bounds
    if xmin<wxmin or ymin<wymin or xmax>wxmax or ymax>wymax:
        return False
    if shapely.contains_properly(water,geometry):
        return True
    # Rejection only: interior vertices cannot rule out crossing an island.
    vertices=shapely.points(shapely.get_coordinates(geometry))
    if not shapely.covers(water,vertices).all():
        return False
    return bool(shapely.is_empty(shapely.difference(geometry,exact_overlay_window(water,geometry))))


class WaterRouter:
    def __init__(self, refined=True, network_dir=OUT):
        refined_file = network_dir / "validated_land_mask.wkb"
        if not refined_file.exists():
            refined_file=OUT/"validated_land_mask.wkb"
        if refined and refined_file.exists():
            self.land = shapely.from_wkb(refined_file.read_bytes())
        else:
            land = gpd.read_file(HERE / "data/public/naturalearth_10m/ne_10m_land.shp")
            self.land = shapely.union_all(land.geometry.to_numpy())
        self._index_land()
        passages=OUT/"observed_canal_transits.json"
        self.observed_canal_water=shapely.Polygon()
        if passages.exists():
            lines=[shapely.LineString(normalise(r["points"])) for r in json.loads(passages.read_text())]
            # A millimetre-scale numeric tolerance preserves documented AIS
            # canal centre-lines, not a made-up navigable river corridor.
            self.observed_canal_water=shapely.union_all(lines).buffer(1e-8)
            shapely.prepare(self.observed_canal_water)
        manifest=network_dir/"regional_water_overrides.json"
        self.regional_water=[]
        if manifest.exists():
            self.regional_water=json.loads(manifest.read_text())["regions"]
            self.region_boxes=np.array([shapely.box(*r["bounds"]) for r in self.regional_water],dtype=object)
            self.region_tree=shapely.STRtree(self.region_boxes)

    @lru_cache(maxsize=64)
    def regional_mask(self,index):
        water=shapely.from_wkb(Path(self.regional_water[index]["water_file"]).read_bytes())
        if self.regional_water[index].get("navigation_passage_line"):
            assert water.geom_type=="LineString"
            water=shapely.LineString(normalise(np.asarray(water.coords)))
        shapely.prepare(water)
        return water

    def _index_land(self):
        if getattr(self, "_indexed_land", None) is self.land:
            return
        pending = list(shapely.get_parts(self.land))
        polygons = []
        while pending:
            part = pending.pop()
            if part.is_empty:
                continue
            if part.geom_type in ("GeometryCollection", "MultiPolygon"):
                pending.extend(shapely.get_parts(part))
            else:
                polygons.append(part)
        self.polygons = np.asarray(polygons, dtype=object)
        shapely.prepare(self.polygons)
        self.land_tree = shapely.STRtree(self.polygons)
        self._indexed_land = self.land

    def is_water(self, points, include_water_boundary=False):
        self._index_land()
        p = normalise(points)
        flat = p.reshape(-1, 2)
        geometries = shapely.points(flat)
        pairs = self.land_tree.query(geometries)
        inside = np.zeros(len(flat), dtype=bool)
        if pairs.shape[1]:
            hits = shapely.contains(self.polygons[pairs[1]], geometries[pairs[0]])
            inside[pairs[0, hits]] = True
        water=~inside
        if getattr(self,"regional_water",None):
            pairs=self.region_tree.query(geometries,predicate="intersects")
            for j in np.unique(pairs[0]):
                ids=tuple(sorted(pairs[1,pairs[0]==j]))
                predicate=shapely.covers if include_water_boundary else shapely.contains
                water[j]=any(predicate(self.regional_mask(i),geometries[j]) for i in ids)
        if hasattr(self,"observed_canal_water"):
            water|=shapely.covers(self.observed_canal_water,geometries)
        return water.reshape(p.shape[:-1])

    def clear(self, points):
        self._index_land()
        for p in split_dateline(points):
            # This is rejection only: passing samples never replaces the full
            # line/polygon test below. Avoid costly overlays for inland chords.
            indices=np.unique(np.linspace(0,len(p)-1,min(5,len(p)),dtype=int))
            if not self.is_water(np.asarray(p)[indices],include_water_boundary=True).all():
                return False
            line = shapely.LineString(p)
            if getattr(self,"regional_water",None):
                # Source-documented covered navigation passages have zero area.
                # Subtract exact original line segments before polygon clipping,
                # which otherwise perturbs collinearity at floating-point seams.
                for index in self.region_tree.query(line,predicate="intersects"):
                    if self.regional_water[index].get("navigation_passage_line"):
                        line=shapely.difference(line,self.regional_mask(index))
            outside=line
            if getattr(self,"regional_water",None):
                ids=tuple(sorted(self.region_tree.query(line,predicate="intersects")))
                if ids:
                    boxes=shapely.union_all(self.region_boxes[list(ids)])
                    inside=shapely.intersection(line,boxes)
                    # Subtract covered line pieces, avoiding an expensive
                    # polygon union of overlapping high-resolution coasts.
                    for index in ids:
                        wet=self.regional_mask(index)
                        if covers_components(wet,inside):
                            inside=shapely.LineString()
                            break
                        piece=shapely.intersection(inside,self.region_boxes[index])
                        if covers_components(wet,piece):
                            inside=shapely.difference(inside,self.region_boxes[index])
                        elif shapely.intersects(wet,piece):
                            inside=shapely.difference(inside,exact_overlay_window(wet,inside))
                    if not inside.is_empty and hasattr(self,"observed_canal_water"):
                        inside=shapely.difference(inside,self.observed_canal_water)
                    if not inside.is_empty:
                        return False
                    outside=shapely.difference(line,boxes)
            candidates = self.land_tree.query(outside)
            if shapely.intersects(self.polygons[candidates], outside).any():
                return False
        return True

    def offshore(self, point):
        point = normalise(np.asarray(point))
        if self.is_water(point):
            return point, 0.0
        shape = shapely.Point(point)
        i = int(self.land_tree.nearest(shape))
        boundary = nearest_points(shape, self.polygons[i].boundary)[1]
        anchor = np.asarray(boundary.coords[0])
        # Move outside a coastal polygon, recording the port-to-water offset.
        for metres in (50, 200, 1000, 5000):
            theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            offsets = np.c_[np.cos(theta) / max(np.cos(np.radians(anchor[1])), .03), np.sin(theta)] * metres / 111195
            candidates = normalise(anchor + offsets)
            water = candidates[self.is_water(candidates)]
            if len(water):
                d = haversine_vec(point[0], point[1], water[:, 0], water[:, 1])
                j = int(np.argmin(d))
                return water[j], float(d[j])
        raise ValueError(f"Unable to resolve offshore endpoint {point.tolist()}")

    def route(self, a, b, step=.05):
        a, b = normalise(np.asarray([a, b]))
        direct = densify(a, b)
        if self.clear(direct):
            return direct
        if not self.is_water(np.asarray([a, b])).all():
            return None
        # Expand a geographic search window; there is no access-distance cutoff.
        for margin in (.35, 1.0, 3.0, 8.0, 20.0):
            result = self._astar(a, b, step, margin)
            if result is not None:
                assert self.clear(result)
                return result
        return None

    def _astar(self, a, b, step, margin):
        centre = float(a[0])
        bx = centre + ((b[0] - centre + 180) % 360 - 180)
        xmin, xmax = min(centre, bx) - margin, max(centre, bx) + margin
        ymin, ymax = max(-85, min(a[1], b[1]) - margin), min(85, max(a[1], b[1]) + margin)
        nx, ny = int(np.ceil((xmax - xmin) / step)) + 1, int(np.ceil((ymax - ymin) / step)) + 1
        if nx * ny > 150000:
            return None
        def xy(node):
            y, x = divmod(node, nx)
            return np.asarray([((xmin + x * step + 180) % 360) - 180, ymin + y * step])
        def nearby(point):
            px = centre + ((point[0] - centre + 180) % 360 - 180)
            x, y = int(round((px - xmin) / step)), int(round((point[1] - ymin) / step))
            candidates = []
            for reach in (1, 2, 4, 8):
                for dy in range(-reach, reach + 1):
                    for dx in range(-reach, reach + 1):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < nx and 0 <= yy < ny:
                            n = yy * nx + xx
                            p = xy(n)
                            if self.is_water(p) and self.clear(densify(point, p)):
                                candidates.append((length([point, p]), n))
                if candidates:
                    return sorted(set(candidates))[:8]
            return []
        starts, ends = nearby(a), nearby(b)
        if not starts or not ends:
            return None
        targets = dict((n, d) for d, n in ends)
        distances, pred, queue = {}, {}, []
        for d, n in starts:
            distances[n], pred[n] = d, -1
            heapq.heappush(queue, (d + length([xy(n), b]), d, n))
        water_cache, edge_cache = {}, {}
        found, best = None, np.inf
        explored = 0
        while queue:
            bound, cost, u = heapq.heappop(queue)
            if bound >= best:
                break
            if cost > distances[u] + 1e-9:
                continue
            explored += 1
            if explored > 3000:
                return None
            if u in targets and cost + targets[u] < best:
                found, best = u, cost + targets[u]
            y, x = divmod(u, nx)
            pu = xy(u)
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                xx, yy = x + dx, y + dy
                if not (0 <= xx < nx and 0 <= yy < ny):
                    continue
                v = yy * nx + xx
                if v not in water_cache:
                    water_cache[v] = bool(self.is_water(xy(v)))
                if not water_cache[v]:
                    continue
                pv = xy(v)
                key = (min(u, v), max(u, v))
                if key not in edge_cache:
                    edge_cache[key] = self.clear([pu, pv])
                if not edge_cache[key]:
                    continue
                new = cost + length([pu, pv])
                if new < distances.get(v, np.inf) - 1e-9:
                    distances[v], pred[v] = new, u
                    heapq.heappush(queue, (new + length([pv, b]), new, v))
        if found is None:
            return None
        path = [found]
        while pred[path[-1]] >= 0:
            path.append(pred[path[-1]])
        path.reverse()
        grid = np.asarray([xy(n) for n in path])
        points = np.vstack([densify(a, grid[0])[:-1], grid,
                            densify(grid[-1], b)[1:]])
        # Keep all grid segments: no visual smoothing across land.
        return points


def geometries_to_npz(filename, records, **arrays):
    sizes = np.asarray([len(p) for p in records])
    np.savez_compressed(filename, offsets=np.r_[0, np.cumsum(sizes)],
        geometry=np.vstack(records) if records else np.empty((0, 2)), **arrays)
