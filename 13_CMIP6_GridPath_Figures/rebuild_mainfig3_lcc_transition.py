"""Exhaustive finite-network A/B dominance test on the current water graph.

A and B partition economically selected OD connections. B is NOT A union B.
Trade exposure is measured on a fixed, explicitly covered trade-value subset.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from rebuild_mainfig2_latest_network import ROOT, NET, OLD, SHIPS, ROUTES, Geometry, simple, sha
from recompute_physical_arctic_embedding import PhysicalRouter, core_ice_masks
from repair_cmip6_monthly_routes import repaired_route_metadata
from prepare_physical_monthly_ice import bit_count
from physical_ocean_routing import length
from chokepoints24 import Exposure

OUT = ROOT / "27_MainFig3_NetworkTransition"
PREMIUM = .05


def components(src, dst, take, n):
    if not np.any(take):
        return 0, np.empty(0, int), 0.
    graph = coo_matrix((np.ones(int(take.sum())), (src[take], dst[take])), shape=(n,n)).tocsr()
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    winner = int(np.argmax(sizes))
    return int(sizes[winner]), np.flatnonzero(labels == winner), float(np.sum(sizes*(sizes-1))/max(n*(n-1),1))


def cost_choice(without, arctic, premium=PREMIUM):
    chosen = np.isfinite(arctic) & ((1+premium)*arctic < without-1e-6)
    cost = np.where(chosen, (1+premium)*arctic, without)
    return chosen, cost


def changed_cost(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.zeros(len(a))
    delta[finite] = np.abs(a[finite]-b[finite])
    return (np.isfinite(a) != np.isfinite(b)) | (delta > .01)


def dominance_crossing(a0,b0,a,b):
    """Relative trade dominance; new Arctic flow is reported, not required."""
    return bool(a0>b0 and b>a+1e-8)


def write_metrics(base, combos, arrays, destination):
    w = base.activity_weight.to_numpy(float)
    tw = base.trade_weight_usd.fillna(0).to_numpy(float)
    trade = base.trade_weight_usd.notna().to_numpy()
    without, arctic = arrays['without_km'], arrays['arctic_km']
    chosen, costs = cost_choice(without, arctic)
    lookup = {c:i for i,c in enumerate(combos)}
    rows, marginal = [], []
    registry = pd.read_csv(ROOT/'data/chokepoints24/registry.csv')
    for ship in SHIPS:
        typ = base.shiptype.eq(ship).to_numpy()
        for scope, eligible, weights in [('all_represented',typ,w), ('trade_covered',typ & trade,tw)]:
            ports = pd.Index(sorted(set(base.loc[eligible,'origin_port_id']) | set(base.loc[eligible,'dest_port_id'])))
            src = ports.get_indexer(base.origin_port_id)
            dst = ports.get_indexer(base.dest_port_id)
            total = weights[eligible].sum()
            a0 = eligible & np.isfinite(without[0]) & ~chosen[0]
            b0 = eligible & chosen[0]
            lcca0 = components(src,dst,a0,len(ports))[0]
            lccb0 = components(src,dst,b0,len(ports))[0]
            for si,c in enumerate(combos):
                closed = sum(1<<g for g in c)
                exposed = eligible & ((arrays['open_gate_mask'] & closed) != 0)
                a = eligible & np.isfinite(without[si]) & ~chosen[si]
                b = eligible & chosen[si]
                la,_,ra = components(src,dst,a,len(ports))
                lb,_,rb = components(src,dst,b,len(ports))
                # Evaluate each added gate against its immediate parent, in the
                # SAME fixed OD population and with the SAME cost parameter.
                margins=[]
                for gate in c:
                    parent=lookup.get(tuple(g for g in c if g!=gate))
                    if parent is None:
                        continue
                    changed=eligible & (changed_cost(costs[si],costs[parent]) | (chosen[si]!=chosen[parent]))
                    share=100*weights[changed].sum()/total if total else 0
                    margins.append(share)
                    marginal.append(dict(shiptype=ship,scope=scope,scenario_index=si,removed_gate=registry.iloc[gate]['name'],
                        parent_index=parent,changed_od=int(changed.sum()),changed_weight_pct=share))
                endpoint=eligible & ((base.endpoint_closure_mask.to_numpy(np.uint32)&closed)!=0)
                safe=eligible & ~endpoint
                ls_a=components(src,dst,a & safe,len(ports))[0]
                ls_b=components(src,dst,b & safe,len(ports))[0]
                fixed0_a=components(src,dst,a0 & exposed,len(ports))[0]
                fixed0_b=components(src,dst,b0 & exposed,len(ports))[0]
                ex_a=components(src,dst,a & exposed,len(ports))[0]
                ex_b=components(src,dst,b & exposed,len(ports))[0]
                rows.append(dict(shiptype=ship,scope=scope,month=int(base.start_month.iloc[0]),scenario_index=si,
                    gates='+'.join(registry.iloc[list(c)]['key']) if c else 'open',
                    closure_names=' + '.join(registry.iloc[list(c)]['name']) if c else 'Open',n_closed=len(c),
                    od_rows=int(eligible.sum()),ports=len(ports),weight_total=float(total),
                    p_exposed_pct=100*weights[exposed].sum()/total if total else 0,
                    A_lcc=la,B_lcc=lb,A0_lcc=lcca0,B0_lcc=lccb0,
                    A0_weight_pct=100*weights[a0].sum()/total if total else 0,
                    B0_weight_pct=100*weights[b0].sum()/total if total else 0,
                    A_reachability=ra,B_reachability=rb,
                    B_weight_pct=100*weights[b].sum()/total if total else 0,
                    A_weight_pct=100*weights[a].sum()/total if total else 0,
                    unresolved_pct=100*weights[eligible & ~np.isfinite(costs[si])].sum()/total if total else 0,
                    switched_to_B_pct=100*weights[b & ~chosen[0]].sum()/total if total else 0,
                    endpoint_closed_pct=100*weights[endpoint].sum()/total if total else 0,
                    endpoint_excluded_A_lcc=ls_a,endpoint_excluded_B_lcc=ls_b,
                    exposed_A0_lcc=fixed0_a,exposed_B0_lcc=fixed0_b,exposed_A_lcc=ex_a,exposed_B_lcc=ex_b,
                    nonredundant=bool(not c or (len(margins)==len(c) and min(margins)>1e-6)),
                    minimum_gate_marginal_pct=min(margins) if margins else 0,
                    lcc_crossover=bool(lcca0>lccb0 and lb>la),
                    crossover=dominance_crossing(weights[a0].sum(),weights[b0].sum(),weights[a].sum(),weights[b].sum())))
    pd.DataFrame(rows).to_csv(destination/'scenario_metrics.csv',index=False)
    pd.DataFrame(marginal).to_csv(destination/'marginal_gate_audit.csv',index=False)


def run(month=9,max_closures=3,custom=None,destination_root=None,
        network_dir=NET, baseline_path=None, trade_path=None, climate_path=None):
    started=time.monotonic()
    baseline_path = baseline_path or OLD/'baseline_all.parquet'
    trade_path = trade_path or OLD/'panel_c_trade_weighted_od.parquet'
    climate_path = climate_path or OLD/'climate/edge_monthly_ice.npz'
    destination=(destination_root or OUT)/f'month_{month:02d}'
    destination.mkdir(parents=True,exist_ok=True)
    combos=([()] + [c for k in range(1,max_closures+1) for c in itertools.combinations(range(24),k)]) if custom is None else custom
    index={c:i for i,c in enumerate(combos)}
    signatures={str(p.relative_to(ROOT)):sha(p) for p in [network_dir/'physical_edges.parquet',network_dir/'edge_geometry.npz',
        network_dir/'canonical_endpoint_access.csv',baseline_path,climate_path,
        trade_path,ROOT/'data/cmip6_monthly_repair/monthly_four_route_paths.csv']}
    signature=dict(inputs=signatures,month=month,combos=combos,premium=PREMIUM,version=1)
    (destination/'input_manifest.json').write_text(json.dumps(signature,indent=2))
    base=pd.read_parquet(baseline_path)
    base=base[base.start_month.eq(month)].reset_index(drop=True)
    trade=pd.read_parquet(trade_path)
    trade=trade[trade.selection_status.eq('included')][['od_id','trade_weight_usd','sea_share_imputed']]
    base=base.merge(trade,on='od_id',how='left',validate='one_to_one')
    base.to_parquet(destination/'demand.parquet',index=False)
    geo=Geometry(network_dir)
    ice=np.load(climate_path)['model_year_mask'][month-1]
    conventional=PhysicalRouter(geo.edges,geo.nodes,ice,100)
    arouter=PhysicalRouter(geo.edges[bit_count(ice[geo.edges.geometry_id.to_numpy()])>=30].reset_index(drop=True),geo.nodes,ice,100)
    endpoints=pd.read_csv(network_dir/'canonical_endpoint_access.csv').set_index('key')
    on=endpoints.loc[base.origin_port_id,'node'].to_numpy(int)
    dn=endpoints.loc[base.dest_port_id,'node'].to_numpy(int)
    cs,ct=conventional.to_compact[on],conventional.to_compact[dn]
    assert np.all(cs>=0) and np.all(ct>=0)
    ps,pt=arouter.to_compact[on],arouter.to_compact[dn]
    valid=(ps>=0)&(pt>=0)
    unique=np.unique(np.r_[ps[ps>=0],pt[pt>=0]])
    slots={x:j for j,x in enumerate(unique)}
    ss,tt=np.array([slots.get(x,0) for x in ps]),np.array([slots.get(x,0) for x in pt])
    points,metadata=repaired_route_metadata()
    gateways=pd.read_csv(network_dir/'gateway_mapping.csv')
    classifier=Exposure(pd.read_csv(ROOT/'data/chokepoints24/registry.csv'))
    cmasks=core_ice_masks()
    gates,cores={},{}
    for route in ROUTES:
        if (route,month) not in metadata:
            continue
        gp=gateways[gateways.route.eq(route)&gateways.month.eq(month)].set_index('side')
        pair=[]
        for side in ('a','b'):
            key=gp.loc[side,'key']; node=int(arouter.to_compact[int(endpoints.loc[key,'node'])])
            if node<0:
                break
            if key not in gates:
                gates[key]=dict(node=node,offset=len(gates)*len(unique))
            pair.append(gates[key])
        if len(pair)==2:
            p=points[points.route.eq(route)&points.month.eq(month)].sort_values('order')[['lon','lat']].to_numpy(float)
            cores[route]=dict(pair=pair,points=p,km=length(p),ice=cmasks[(route,month)],mask=classifier.masks(p)[100])
    sources=np.concatenate([np.full(len(unique),g['node']) for g in gates.values()]).astype(int) if gates else np.array([],int)
    targets=np.tile(unique,len(gates))

    @lru_cache(maxsize=1024)
    def geometry(pid):
        return geo.path(arouter.active[arouter.paths[pid]])

    @lru_cache(maxsize=65536)
    def valid_geometry(p,q,route,orientation):
        if set(arouter.paths[p]) & set(arouter.paths[q]):
            return False
        left,right=geometry(p)[::-1],geometry(q)
        core=cores[route]['points'] if orientation==1 else cores[route]['points'][::-1]
        if not len(left) or not len(right):
            return False
        assert max(length([left[-1],core[0]]),length([core[-1],right[0]]))<.001
        return simple(np.vstack([left,core[1:],right[1:]]))

    n,m=len(base),len(combos)
    arrays={k:np.full((m,n),np.inf,np.float64) for k in ('without_km','arctic_km')}
    for k in ('conventional_id','entry_id','exit_id'):
        arrays[k]=np.zeros((m,n),np.int32)
    for k in ('route_code','orientation'):
        arrays[k]=np.zeros((m,n),np.uint8)
    arrays['joint_ice']=np.zeros((m,n),np.uint64)
    parents_c,parents_a={},{}
    rejects=0
    required=set(parent for c in combos for j in range(len(c)) for parent in [c[:j]+c[j+1:]] if parent in index)
    for si,c in enumerate(combos):
        closed=np.uint32(sum(1<<g for g in c))
        parents=[index[c[:j]+c[j+1:]] for j in range(len(c)) if c[:j]+c[j+1:] in index]
        cp=conventional.solve(cs,ct,closed,np.stack([parents_c[j] for j in parents]) if parents else None)
        ap=arouter.solve(sources,targets,closed,np.stack([parents_a[j] for j in parents]) if parents else None) if gates else np.array([],np.int32)
        if c in required:
            parents_c[si]=cp; parents_a[si]=ap
        arrays['conventional_id'][si]=cp
        without=np.asarray(conventional.length)[cp]
        arrays['without_km'][si]=without
        if si==0:
            arrays['open_gate_mask']=np.asarray(conventional.path_mask,np.uint32)[cp]
        lengths=np.asarray(arouter.length)
        path_ice=np.asarray(arouter.path_ice,np.uint64)
        for ri,route in enumerate(ROUTES,1):
            if route not in cores or cores[route]['mask'] & int(closed):
                continue
            core=cores[route]
            for orientation,(left,right) in enumerate((core['pair'],core['pair'][::-1]),1):
                p,q=ap[left['offset']+ss],ap[right['offset']+tt]
                candidate=lengths[p]+core['km']+lengths[q]
                joint=path_ice[p]&path_ice[q]&core['ice']
                potential=valid & np.isfinite(candidate) & (candidate<without-1e-6) & (candidate<arrays['arctic_km'][si]-1e-6) & (bit_count(joint)>=30)
                for j in np.flatnonzero(potential):
                    if not valid_geometry(int(p[j]),int(q[j]),route,orientation):
                        potential[j]=False; rejects+=1
                arrays['arctic_km'][si,potential]=candidate[potential]
                arrays['entry_id'][si,potential]=p[potential]
                arrays['exit_id'][si,potential]=q[potential]
                arrays['orientation'][si,potential]=orientation
                arrays['route_code'][si,potential]=ri
                arrays['joint_ice'][si,potential]=joint[potential]
        if si%50==0 or si==m-1:
            print(json.dumps(dict(month=month,scenario=si+1,total=m,seconds=round(time.monotonic()-started),
                conventional_trees=conventional.dijkstra_trees,arctic_trees=arouter.dijkstra_trees,
                conventional_paths=len(conventional.paths),arctic_paths=len(arouter.paths))),flush=True)
    np.savez_compressed(destination/'scenario_routes.npz',**arrays)
    conventional.save_catalogue(destination/'conventional')
    arouter.save_catalogue(destination/'arctic')
    (destination/'scenarios.json').write_text(json.dumps(combos))
    print('Computing port-level components and gate marginal effects',flush=True)
    write_metrics(base,combos,arrays,destination)
    audit=dict(month=month,scenario_count=m,od_count=n,loops_rejected=rejects,
        conventional_trees=conventional.dijkstra_trees,arctic_trees=arouter.dijkstra_trees,
        elapsed_seconds=time.monotonic()-started,available_cores=list(cores),status='computed')
    (destination/'calculation_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--month',type=int,default=9)
    parser.add_argument('--max-closures',type=int,default=3)
    parser.add_argument('--combinations',type=Path)
    args=parser.parse_args()
    run(args.month,args.max_closures,[tuple(c) for c in json.loads(args.combinations.read_text())] if args.combinations else None)
