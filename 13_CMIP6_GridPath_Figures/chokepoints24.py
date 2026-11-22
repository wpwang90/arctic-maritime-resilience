"""Published 24-chokepoint registry and auditable AIS proximity tests.

Public PortWatch point locations are not the unpublished author polygons.
Distances below describe proximity exposure, not verified canal transits.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "chokepoints24"
PORTWATCH_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/PortWatch_chokepoints_database/FeatureServer/0/query"
PAPER_URL = "https://doi.org/10.1038/s41467-025-65403-w"
CODE_URL = "https://zenodo.org/records/19663059"
NAMES = {
    "Bab_el_Mandeb": "Bab el-Mandeb Strait", "Bohai": "Bohai Strait",
    "Bosporus": "Bosporus Strait", "Cape_Good_Hope": "Cape of Good Hope",
    "Dover": "Dover Strait", "Gibraltar": "Gibraltar Strait",
    "Korea": "Korea Strait", "Lombok": "Lombok Strait", "Luzon": "Luzon Strait",
    "Magellan": "Magellan Strait", "Makassar": "Makassar Strait",
    "Malacca": "Malacca Strait", "Mona": "Mona Passage", "Ombai": "Ombai Strait",
    "Oresund": "Oresund Strait", "Panama": "Panama Canal",
    "Hormuz": "Strait of Hormuz", "Suez": "Suez Canal", "Sunda": "Sunda Strait",
    "Taiwan": "Taiwan Strait", "Torres": "Torres Strait", "Tsugaru": "Tsugaru Strait",
    "Windward": "Windward Passage", "Yucatan": "Yucatan Channel",
}
KEYS = list(NAMES)
RADII = (75, 100, 150)
EARTH_KM = 6371.0088


def build_registry() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "portwatch_chokepoints_response.json"
    if not cache.exists():
        response = requests.get(PORTWATCH_URL, params={
            "where": "1=1", "outFields": "*", "outSR": 4326, "f": "json",
        }, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if "features" not in payload or payload.get("exceededTransferLimit"):
            raise ValueError("Incomplete PortWatch response")
        cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload = json.loads(cache.read_text())
    features = payload["features"]
    risks = pd.read_csv(HERE / "data/public/verschuur_chokepoint_code_19663059/chokepoint_systemic_trade_risk.csv")
    detours = risks[["canal", "reroute_km"]].drop_duplicates()
    assert detours.canal.nunique() == len(detours) == 24
    detours = detours.set_index("canal").reroute_km
    old = {"Suez", "Panama", "Malacca", "Hormuz", "Bab_el_Mandeb", "Dover", "Gibraltar", "Oresund", "Bosporus", "Cape_Good_Hope"}
    rows = []
    for index, (key, name) in enumerate(NAMES.items()):
        matched = [f for f in features if name in f["attributes"].values()]
        if len(matched) != 1:
            raise ValueError(f"Expected one public location for {name}: {len(matched)}")
        f = matched[0]
        geom = f["geometry"]
        distance = float(detours[name])
        rows.append(dict(
            number=index + 1, key=key, name=name, lon=geom["x"], lat=geom["y"],
            legacy_key={"Oresund": "Danish", "Bosporus": "Turkish"}.get(key, key) if key in old else "",
            newly_added=key not in old, published_detour_km=distance,
            published_no_maritime_bypass=distance == 0,
            detour_group="Red_Sea" if key in ("Suez", "Bab_el_Mandeb") else key,
            primary_buffer_km=100, sensitivity_buffers_km="75;100;150",
            location_source=PORTWATCH_URL, detour_source=CODE_URL, paper_source=PAPER_URL,
            geometry_status="public_point_with_analyst_buffer_not_author_polygon",
        ))
    registry = pd.DataFrame(rows)
    registry.to_csv(OUT / "registry.csv", index=False)
    (OUT / "registry_provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": PAPER_URL, "author_code_and_data": CODE_URL,
        "public_location_service": PORTWATCH_URL,
        "public_location_count": len(features), "selected_count": 24,
        "author_polygon_availability": "Canals_mapping.shp referenced in notebook but absent from downloaded archive",
        "aliases": {"Danish": "Oresund", "Turkish": "Bosporus"},
        "alias_caveat": "Name alignment does not imply identical old and new geographic footprints.",
        "Bering": "Arctic gateway, not one of the 24 conventional disruption locations",
    }, indent=2), encoding="utf-8")
    return registry


def xyz(points: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(points).T
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


class Exposure:
    """Nearest sampled-polyline distance, with at most 5-km sampling error.

    Segments longer than 150 km are never connected. Densification supports
    geometric intersection only and is not counted as additional AIS data.
    """
    def __init__(self, registry: pd.DataFrame):
        self.centres = xyz(registry[["lon", "lat"]].to_numpy(float))
        self.tree = cKDTree(self.centres)
        second = self.tree.query(self.centres, k=2)[0][:, 1]
        # Ensures nearest-centre queries cannot miss a second 150-km buffer.
        assert np.min(2 * EARTH_KM * np.arcsin(second / 2)) > 300

    def distances(self, points: np.ndarray, densify: bool = True) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        valid = np.isfinite(points).all(axis=1) & (np.abs(points[:, 1]) <= 90) & (np.abs(points[:, 0]) <= 180)
        vectors = xyz(points[valid])
        if not len(vectors):
            return np.full(24, np.inf)
        if densify and len(vectors) > 1:
            chord = np.linalg.norm(vectors[1:] - vectors[:-1], axis=1)
            km = 2 * EARTH_KM * np.arcsin(np.minimum(chord / 2, 1))
            # Preserve gaps caused by invalid input rows as well as long jumps.
            consecutive = np.diff(np.flatnonzero(valid)) == 1
            selected = np.flatnonzero((km > 10) & (km <= 150) & consecutive)
            extras = []
            for i in selected:
                n = int(np.ceil(km[i] / 10))
                f = np.arange(1, n)[:, None] / n
                v = vectors[i] * (1 - f) + vectors[i + 1] * f
                extras.append(v / np.linalg.norm(v, axis=1)[:, None])
            if extras:
                vectors = np.vstack([vectors, *extras])
        chord, index = self.tree.query(vectors)
        distance = 2 * EARTH_KM * np.arcsin(np.minimum(chord / 2, 1))
        result = np.full(24, np.inf)
        nearby = distance <= 155
        np.minimum.at(result, index[nearby], distance[nearby])
        return result

    def masks(self, points: np.ndarray) -> dict[int, int]:
        distance = self.distances(points)
        return {radius: int(sum(1 << int(i) for i in np.flatnonzero(distance <= radius))) for radius in RADII}


def mask_names(value: int) -> str:
    return ";".join(key for i, key in enumerate(KEYS) if int(value) & (1 << i))


if __name__ == "__main__":
    print(build_registry().to_string(index=False))
