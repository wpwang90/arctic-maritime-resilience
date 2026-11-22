"""Sea-coast exclusion masks for all backbone geometries and monthly cores."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

TASK=Path(__file__).resolve().parent
ROOT=TASK.parent
NET=ROOT/'data/physical_arctic_embedding/oxmartrans_1390'
OUT=TASK/'results'
sys.path.insert(0,str(ROOT))
from repair_cmip6_monthly_routes import repaired_route_metadata


def main():
    OUT.mkdir(exist_ok=True)
    started=time.monotonic()
    cfg=json.loads((TASK/'scenarios.json').read_text())
    coastfile=ROOT/'51_TradeWeighted_CountryBetweenness/results/coastal_countries.parquet'
    code=ROOT/'51_TradeWeighted_CountryBetweenness/calculate_coastal_betweenness.py'
    paths=[NET/'physical_edges.parquet',NET/'edge_geometry.npz',coastfile,code,TASK/'scenarios.json']
    signature={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    marker=OUT/'coast_mask_audit.json'
    if marker.exists() and json.loads(marker.read_text()).get('inputs')==signature:
        print('Matching full-backbone coastal masks reused',flush=True)
        return
    spec=importlib.util.spec_from_file_location('coast_geometry',code)
    method=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)
    world=pd.read_parquet(coastfile).set_index('iso3')
    edges=pd.read_parquet(NET/'physical_edges.parquet')
    geo=np.load(NET/'edge_geometry.npz')
    points,offsets=geo['geometry'],geo['offsets']
    ngeom=len(offsets)-1
    gids=np.unique(edges.geometry_id.to_numpy())
    # Any point is at most its full polyline length from the first vertex.
    # This gives a conservative triangle-inequality screen, not a bbox guess.
    lengths=edges.groupby('geometry_id').km.max().reindex(gids).to_numpy()
    first=method.xyz(points[offsets[gids]])
    core,_=repaired_route_metadata()
    core_groups=list(core.groupby(['route','month']))
    core_samples=[method.sample_line(g.sort_values('order')[['lon','lat']].to_numpy()) for _,g in core_groups]
    core_offsets=np.r_[0,np.cumsum([len(x) for x in core_samples])]
    core_cloud=np.vstack(core_samples)
    backbone=np.zeros(ngeom,np.uint32)
    cores=np.zeros(len(core_groups),np.uint32)
    records=[]
    threshold=cfg['coastal_radius_km']+cfg['sampling_error_margin_km']
    for j,host in enumerate(cfg['coast_bit_order']):
        shore=shapely.from_wkb(world.loc[host,'coast_wkb'])
        cloud=np.vstack([method.sample_line(p) for p in method.coastal_parts(shore)])
        tree=cKDTree(cloud)
        chord,_=tree.query(first,k=1,workers=1)
        first_distance=2*method.EARTH*np.arcsin(np.minimum(chord/2,1))
        candidates=gids[first_distance<=threshold+lengths+2]
        hits=uncertain=samples=0
        for start in range(0,len(candidates),500):
            chunk=candidates[start:start+500]
            sampled=[method.sample_line(points[offsets[g]:offsets[g+1]]) for g in chunk]
            starts=np.r_[0,np.cumsum([len(p) for p in sampled])]
            chord,_=tree.query(np.vstack(sampled),k=1,workers=1)
            distance=2*method.EARTH*np.arcsin(np.minimum(chord/2,1))
            minimum=np.minimum.reduceat(distance,starts[:-1])
            bit=np.uint32(1<<int(24+j))
            backbone[chunk[minimum<=threshold]]|=bit
            hits+=int(np.sum(minimum<=threshold))
            uncertain+=int(np.sum((minimum>cfg['coastal_radius_km'])&(minimum<=threshold)))
            samples+=len(distance)
        chord,_=tree.query(core_cloud,k=1,workers=1)
        distance=2*method.EARTH*np.arcsin(np.minimum(chord/2,1))
        core_min=np.minimum.reduceat(distance,core_offsets[:-1])
        cores[core_min<=threshold]|=np.uint32(1<<int(24+j))
        records.append(dict(host=host,bit=24+j,screened_geometries=len(gids)-len(candidates),
            refined_geometries=len(candidates),excluded_geometries=hits,numerical_boundary_band_geometries=uncertain,
            sampled_route_points=samples,shore_samples=len(cloud),blocked_core_months=int(np.sum(core_min<=threshold))))
        print(json.dumps(records[-1]),flush=True)
    np.savez_compressed(OUT/'coastal_masks.npz',geometry_mask=backbone,core_mask=cores,
        core_route=np.array([key[0] for key,_ in core_groups]),
        core_month=np.array([key[1] for key,_ in core_groups]))
    pd.DataFrame(records).to_csv(OUT/'coastal_mask_counts.csv',index=False)
    marker.write_text(json.dumps(dict(status='FULL_BACKBONE_AND_ALL_STORED_CORES_MASKED',
        inputs=signature,geometry_count=ngeom,used_geometry_count=len(gids),core_month_count=len(core_groups),
        numerical_sampling_km=2,conservative_radius_km=threshold,
        caveat='Two km is numerical sampling allowance, not a guarantee of cartographic coastline accuracy',
        seconds=time.monotonic()-started),indent=2))


if __name__=='__main__':main()
