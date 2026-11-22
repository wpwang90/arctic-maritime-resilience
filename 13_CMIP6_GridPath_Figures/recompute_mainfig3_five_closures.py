"""Exhaustive 0--5 gate search with bounded state storage and exact reuse.

The primary outcome is trade-share dominance at a 5% uniform Arctic premium.
Full geometry is solved; only summary metrics and parent states are retained.
"""
import argparse
import hashlib
import itertools
import json
import time
from functools import lru_cache
from multiprocessing import get_context

import numpy as np
import pandas as pd

from rebuild_mainfig3_lcc_transition import OUT as PRIOR, PREMIUM, cost_choice, changed_cost
from rebuild_mainfig2_latest_network import ROOT, NET, OLD, SHIPS, ROUTES, Geometry, simple, sha
from recompute_physical_arctic_embedding import PhysicalRouter, core_ice_masks
from repair_cmip6_monthly_routes import repaired_route_metadata
from prepare_physical_monthly_ice import bit_count
from physical_ocean_routing import length
from chokepoints24 import Exposure

OUT=ROOT/'28_MainFig3_FiveClosures'
COMBOS=[()]+[c for k in range(1,6) for c in itertools.combinations(range(24),k)]
INDEX={c:i for i,c in enumerate(COMBOS)}
N_PARENT=sum(len(c)<5 for c in COMBOS)
MASKS=np.array([sum(1<<g for g in c) for c in COMBOS],np.uint32)
INPUTS=[NET/'physical_edges.parquet',NET/'edge_geometry.npz',NET/'canonical_endpoint_access.csv',
    NET/'gateway_mapping.csv',OLD/'baseline_all.parquet',OLD/'climate/edge_monthly_ice.npz',
    OLD/'panel_c_trade_weighted_od.parquet',ROOT/'data/chokepoints24/registry.csv',
    ROOT/'data/cmip6_monthly_repair/monthly_four_route_paths.csv',
    ROOT/'data/cmip6_monthly_repair/model_year_month_feasibility.csv']


class Intern:
    def __init__(self):
        self.values={}

    def __call__(self,value):
        key=hashlib.sha256(value.tobytes()).digest()
        previous=self.values.get(key)
        if previous is None:
            self.values[key]=value
            return value
        assert np.array_equal(previous,value)
        return previous


def run_month(month):
    started=time.monotonic()
    folder=OUT/f'month_{month:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    fingerprints={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
    signature=dict(inputs=fingerprints,premium=PREMIUM,max_closures=5,version=1,month=month)
    if (folder/'calculation_audit.json').exists():
        saved=json.loads((folder/'input_manifest.json').read_text())
        if saved==signature:
            print(f'Month {month:02d}: completed matching calculation reused',flush=True)
            return
    (folder/'input_manifest.json').write_text(json.dumps(signature,indent=2))
    base=pd.read_parquet(PRIOR/f'month_{month:02d}'/'demand.parquet')
    base.to_parquet(folder/'demand.parquet',index=False)
    geo=Geometry()
    ice=np.load(OLD/'climate/edge_monthly_ice.npz')['model_year_mask'][month-1]
    conventional=PhysicalRouter(geo.edges,geo.nodes,ice,100)
    arouter=PhysicalRouter(geo.edges[bit_count(ice[geo.edges.geometry_id.to_numpy()])>=30].reset_index(drop=True),geo.nodes,ice,100)
    endpoints=pd.read_csv(NET/'canonical_endpoint_access.csv').set_index('key')
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
    gateways=pd.read_csv(NET/'gateway_mapping.csv')
    registry=pd.read_csv(ROOT/'data/chokepoints24/registry.csv')
    classifier=Exposure(registry)
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

    @lru_cache(maxsize=131072)
    def valid_geometry(p,q,route,orientation):
        if set(arouter.paths[p]) & set(arouter.paths[q]):
            return False
        left,right=geometry(p)[::-1],geometry(q)
        core=cores[route]['points'] if orientation==1 else cores[route]['points'][::-1]
        if not len(left) or not len(right):
            return False
        assert max(length([left[-1],core[0]]),length([core[-1],right[0]]))<.001
        return simple(np.vstack([left,core[1:],right[1:]]))

    n=len(base)
    # Access candidates depend only on the access solution; expensive geometry
    # checks are deferred until a candidate can beat the conventional cost.
    @lru_cache(maxsize=64)
    def arctic_candidates(key,blocked):
        ap=np.frombuffer(key,np.int32)
        lengths=np.asarray(arouter.length)
        masks=np.asarray(arouter.path_ice,np.uint64)
        candidates=[]
        for ri,route in enumerate(ROUTES):
            if route not in cores or blocked & (1<<ri):
                continue
            c=cores[route]
            for orientation,(left,right) in enumerate((c['pair'],c['pair'][::-1]),1):
                p,q=ap[left['offset']+ss],ap[right['offset']+tt]
                candidate=lengths[p]+c['km']+lengths[q]
                joint=masks[p]&masks[q]&c['ice']
                feasible=valid & np.isfinite(candidate) & (bit_count(joint)>=30)
                candidates.append((candidate,feasible,p,q,route,orientation))
        return candidates

    @lru_cache(maxsize=512)
    def economic_solution(conventional_key,arctic_key,blocked):
        without=np.asarray(conventional.length)[np.frombuffer(conventional_key,np.int32)]
        best=np.full(n,np.inf)
        for candidate,feasible,p,q,route,orientation in arctic_candidates(arctic_key,blocked):
                potential=feasible & (candidate<best-1e-6) & ((1+PREMIUM)*candidate<without-1e-6)
                for j in np.flatnonzero(potential):
                    if not valid_geometry(int(p[j]),int(q[j]),route,orientation):
                        potential[j]=False
                best[potential]=candidate[potential]
        return cost_choice(without,best)

    cohorts=[]; weights=[]; masks=[]
    for ship in SHIPS:
        for scope in ('all_represented','trade_covered'):
            eligible=base.shiptype.eq(ship).to_numpy()
            w=base.activity_weight.to_numpy(float)
            if scope=='trade_covered':
                eligible &= base.trade_weight_usd.notna().to_numpy()
                w=base.trade_weight_usd.fillna(0).to_numpy(float)
            w=np.where(eligible,w,0)
            total=w.sum()
            assert total>0 and np.all(w>=0)
            weights.append(100*w/total); masks.append(eligible)
            cohorts.append(dict(shiptype=ship,scope=scope,month=month,od_rows=int(eligible.sum()),
                ports=len(set(base.loc[eligible,'origin_port_id'])|set(base.loc[eligible,'dest_port_id'])),weight_total=float(total)))
    weights=np.array(weights); cohort_masks=np.array(masks)
    def aggregate(take):
        return np.einsum('ij,j->i',weights,take,optimize=False)
    fields=('p_exposed_pct','A_weight_pct','B_weight_pct','unresolved_pct','switched_to_B_pct','endpoint_closed_pct','minimum_gate_marginal_pct')
    stats={f:np.zeros((len(COMBOS),8)) for f in fields}
    margins=np.full((len(COMBOS),5,8),np.nan,np.float32)
    marginal_od=np.zeros((len(COMBOS),5,8),np.uint16)
    parent_c={};parent_a={};parent_cost={};parent_selected={}
    intern_c,intern_a,intern_cost,intern_selected=Intern(),Intern(),Intern(),Intern()
    old=np.load(PRIOR/f'month_{month:02d}'/'scenario_routes.npz')
    old_without=old['without_km']; old_arctic=old['arctic_km']
    old_chosen,old_cost=cost_choice(old_without,old_arctic)
    matched=0; raw_distance_differences=0
    endpoint_mask=base.endpoint_closure_mask.to_numpy(np.uint32)
    lengths=np.asarray(conventional.length); previous_size=len(lengths)
    for si,c in enumerate(COMBOS):
        closed=MASKS[si]
        parents=[INDEX[c[:j]+c[j+1:]] for j in range(len(c))]
        cp=conventional.solve(cs,ct,closed,np.stack([parent_c[j] for j in parents]) if parents else None)
        ap=arouter.solve(sources,targets,closed,np.stack([parent_a[j] for j in parents]) if parents else None) if gates else np.array([],np.int32)
        if len(conventional.length)!=previous_size:
            lengths=np.asarray(conventional.length);previous_size=len(lengths)
        without=lengths[cp]
        blocked=sum(1<<ri for ri,r in enumerate(ROUTES) if r in cores and cores[r]['mask']&int(closed))
        if gates:
            chosen,cost=economic_solution(cp.tobytes(),ap.tobytes(),blocked)
        else:
            chosen=np.zeros(n,bool);cost=without
        if si==0:
            open_mask=np.asarray(conventional.path_mask,np.uint32)[cp]
            open_chosen=chosen.copy()
        if si<N_PARENT:
            parent_c[si]=intern_c(cp);parent_a[si]=intern_a(ap)
            parent_cost[si]=intern_cost(cost);parent_selected[si]=intern_selected(chosen)
        for j,pi in enumerate(parents):
            pcost=parent_cost[pi]
            changed=changed_cost(cost,pcost) | (chosen!=parent_selected[pi])
            margins[si,j]=aggregate(changed)
            marginal_od[si,j]=np.count_nonzero(cohort_masks & changed,axis=1)
            assert np.all(without+1e-6>=lengths[parent_c[pi]])
        if parents:
            stats['minimum_gate_marginal_pct'][si]=margins[si,:len(parents)].min(axis=0)
        stats['p_exposed_pct'][si]=aggregate((open_mask&closed)!=0)
        stats['B_weight_pct'][si]=aggregate(chosen)
        stats['A_weight_pct'][si]=aggregate(~chosen & np.isfinite(without))
        stats['unresolved_pct'][si]=aggregate(~np.isfinite(cost))
        stats['switched_to_B_pct'][si]=aggregate(chosen & ~open_chosen)
        stats['endpoint_closed_pct'][si]=aggregate((endpoint_mask&closed)!=0)
        if si<len(old_without):
            assert np.allclose(without,old_without[si],atol=1e-6),(month,si,'conventional mismatch')
            assert np.array_equal(chosen,old_chosen[si]),(month,si,'choice mismatch')
            assert np.allclose(cost,old_cost[si],atol=1e-6),(month,si,'cost mismatch')
            matched+=n
        if si%2500==0 or si==len(COMBOS)-1:
            print(json.dumps(dict(month=month,scenario=si+1,total=len(COMBOS),elapsed_seconds=round(time.monotonic()-started),
                conventional_trees=conventional.dijkstra_trees,arctic_trees=arouter.dijkstra_trees,
                arctic_cache=economic_solution.cache_info()._asdict(),parent_cost_states=len(intern_cost.values))),flush=True)
    frames=[]
    for i,cohort in enumerate(cohorts):
        frame=pd.DataFrame({f:stats[f][:,i] for f in fields})
        for k,v in cohort.items():
            frame[k]=v
        frame['scenario_index']=np.arange(len(COMBOS))
        frame['n_closed']=[len(c) for c in COMBOS]
        frame['nonredundant']=(frame.minimum_gate_marginal_pct>1e-6)|frame.n_closed.eq(0)
        frame['A0_weight_pct']=stats['A_weight_pct'][0,i]
        frame['B0_weight_pct']=stats['B_weight_pct'][0,i]
        frame['crossover']=(frame.A0_weight_pct>frame.B0_weight_pct)&(frame.B_weight_pct>frame.A_weight_pct+1e-10)
        frame['arctic_premium_pct']=100*PREMIUM
        assert np.allclose(frame.A_weight_pct+frame.B_weight_pct+frame.unresolved_pct,100,atol=1e-8)
        frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    result.to_parquet(folder/'scenario_metrics.parquet',index=False)
    np.savez_compressed(folder/'marginal_gate_audit.npz',marginal_pct=margins,changed_od=marginal_od)
    (folder/'cohorts.json').write_text(json.dumps(cohorts,indent=2))
    audit=dict(status='PASS',month=month,max_closures=5,premium_pct=100*PREMIUM,scenarios=len(COMBOS),
        od_rows=n,old_OD_scenario_choices_and_costs_matched=matched,all_immediate_parent_distances_monotone=True,
        fixed_denominator_conservation=True,conventional_dijkstra_trees=conventional.dijkstra_trees,
        arctic_dijkstra_trees=arouter.dijkstra_trees,elapsed_seconds=time.monotonic()-started,
        primary_criterion='B trade share > A trade share; conventional dominance at open baseline',
        geometry='same complete monthly core and physical access checks as prior Figure 3',
        no_full_OD_scenario_matrix_saved=True)
    (folder/'calculation_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--months',type=int,nargs='+',default=list(range(1,13)))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT/'scenarios.json').write_text(json.dumps(COMBOS))
    if args.workers==1:
        for month in args.months:
            run_month(month)
    else:
        with get_context('spawn').Pool(args.workers) as pool:
            pool.map(run_month,args.months,chunksize=1)
