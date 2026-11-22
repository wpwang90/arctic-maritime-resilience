"""Full-water-path, four-class hypothetical stress analysis, 5% Arctic premium."""
import argparse
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

TASK=Path(__file__).resolve().parent
ROOT=TASK.parent
sys.path.insert(0,str(ROOT))
from rebuild_mainfig2_latest_network import Geometry,NET,OLD,SHIPS,ROUTES,simple
from recompute_physical_arctic_embedding import PhysicalRouter,core_ice_masks
from repair_cmip6_monthly_routes import repaired_route_metadata
from prepare_physical_monthly_ice import bit_count
from physical_ocean_routing import length
from chokepoints24 import Exposure

OUT=TASK/'results'
COMBOS=[()]+[c for k in range(1,6) for c in itertools.combinations(range(24),k)]
COMBO_INDEX={sum(1<<g for g in c):i for i,c in enumerate(COMBOS)}


def scenario_rows(cfg,registry):
    gate_bits={r.key:1<<(int(r.number)-1) for r in registry.itertuples()}
    coast_bits={country:1<<(24+j) for j,country in enumerate(cfg['coast_bit_order'])}
    rows=[]
    for group in cfg['classes']:
        mask=0;ports=[];labels=[]
        rows.append(dict(class_id=group['id'],stage=0,closed_mask=0,ports=[],new_unit='Baseline',units=[]))
        for k,step in enumerate(group['steps'],1):
            for gate in step.get('gates',[]):mask|=gate_bits[gate]
            for country in step.get('coasts',[]):mask|=coast_bits[country]
            ports.extend(step.get('ports',[]));labels.append(step['label'])
            rows.append(dict(class_id=group['id'],stage=k,closed_mask=mask,ports=list(ports),
                             new_unit=step['label'],units=list(labels)))
    return rows


def choose(a,b,service,premium):
    use_b=service & np.isfinite(b) & ((1+premium)*b<a-1e-6)
    use_a=service & np.isfinite(a) & ~use_b
    unresolved=~(use_a|use_b)
    return use_a,use_b,unresolved,np.where(service,np.minimum(a,(1+premium)*b),np.inf)


def changed(x,y):
    both=np.isfinite(x)&np.isfinite(y)
    out=np.isfinite(x)!=np.isfinite(y)
    out[both]=np.abs(x[both]-y[both])>1e-6
    return out


def run_month(month):
    started=time.monotonic()
    folder=OUT/f'month_{month:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((TASK/'scenarios.json').read_text())
    files=[TASK/'scenarios.json',TASK/'calculate_thresholds.py',OUT/'coastal_masks.npz',
           NET/'physical_edges.parquet',NET/'edge_geometry.npz',NET/'canonical_endpoint_access.csv',
           OLD/'baseline_all.parquet',OLD/'panel_c_trade_weighted_od.parquet',
           OLD/'climate/edge_monthly_ice.npz',ROOT/'data/cmip6_monthly_repair/monthly_four_route_paths.csv',
           ROOT/'data/cmip6_monthly_repair/model_year_month_feasibility.csv']
    signature={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    marker=folder/'audit.json'
    if marker.exists() and json.loads(marker.read_text()).get('inputs')==signature:
        print('Month %02d matching complete calculation reused'%month,flush=True)
        return
    base=pd.read_parquet(OLD/'baseline_all.parquet')
    base=base[base.start_month.eq(month)]
    trade=pd.read_parquet(OLD/'panel_c_trade_weighted_od.parquet')
    trade=trade.loc[trade.selection_status.eq('included'),['od_id','trade_weight_usd']]
    base=base.merge(trade,on='od_id',how='inner',validate='one_to_one').reset_index(drop=True)
    base.to_parquet(folder/'demand.parquet',index=False)
    geo=Geometry()
    coast=np.load(OUT/'coastal_masks.npz')
    edges=geo.edges.copy()
    edges['mask_100']=edges.mask_100.to_numpy(np.uint32)|coast['geometry_mask'][edges.geometry_id.to_numpy()]
    ice=np.load(OLD/'climate/edge_monthly_ice.npz')['model_year_mask'][month-1]
    conventional=PhysicalRouter(edges,geo.nodes,ice,100)
    arouter=PhysicalRouter(edges[bit_count(ice[edges.geometry_id.to_numpy()])>=30].reset_index(drop=True),geo.nodes,ice,100)
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
    core_ice=core_ice_masks()
    core_extra={(str(r),int(m)):int(mask) for r,m,mask in zip(coast['core_route'],coast['core_month'],coast['core_mask'])}
    gates,cores={},{}
    for route in ROUTES:
        if (route,month) not in metadata or int(bit_count(np.array([core_ice[(route,month)]]))[0])<30:
            continue
        group=gateways[gateways.route.eq(route)&gateways.month.eq(month)].set_index('side')
        pair=[]
        for side in ('a','b'):
            key=group.loc[side,'key']
            node=int(arouter.to_compact[int(endpoints.loc[key,'node'])])
            if node<0:break
            if key not in gates:gates[key]=dict(node=node,offset=len(gates)*len(unique))
            pair.append(gates[key])
        if len(pair)==2:
            core=points[points.route.eq(route)&points.month.eq(month)].sort_values('order')[['lon','lat']].to_numpy(float)
            cores[route]=dict(pair=pair,points=core,km=length(core),ice=core_ice[(route,month)],
                mask=classifier.masks(core)[100]|core_extra[(route,month)])
    sources=np.concatenate([np.full(len(unique),g['node']) for g in gates.values()]).astype(int) if cores else np.array([],int)
    targets=np.tile(unique,len(gates)) if cores else np.array([],int)

    @lru_cache(maxsize=2048)
    def geometry(pid):return geo.path(arouter.active[arouter.paths[pid]])

    @lru_cache(maxsize=131072)
    def geometry_ok(p,q,route,orientation):
        if set(arouter.paths[p])&set(arouter.paths[q]):return False
        left,right=geometry(p)[::-1],geometry(q)
        core=cores[route]['points'] if orientation==1 else cores[route]['points'][::-1]
        if not len(left) or not len(right):return False
        assert max(length([left[-1],core[0]]),length([core[-1],right[0]]))<.001
        return simple(np.vstack([left,core[1:],right[1:]]))

    scenarios=scenario_rows(cfg,registry)
    masks=sorted({x['closed_mask'] for x in scenarios},key=lambda x:(bin(x).count('1'),x))
    saved={}
    n=len(base)
    print(json.dumps(dict(month=month,od=n,route_states=len(masks),available_cores=list(cores))),flush=True)
    for si,mask in enumerate(masks):
        possible=[m for m in saved if m & mask==m]
        parents=sorted(possible,key=lambda m:bin(m).count('1'),reverse=True)[:3]
        cp=conventional.solve(cs,ct,np.uint32(mask),np.stack([saved[m]['cp'] for m in parents]) if parents else None)
        ap=arouter.solve(sources,targets,np.uint32(mask),np.stack([saved[m]['ap'] for m in parents]) if parents else None) if cores else np.array([],np.int32)
        without=np.asarray(conventional.length)[cp]
        best=np.full(n,np.inf)
        entry,exit_,routecode,orientation=[np.zeros(n,np.int32) for _ in range(4)]
        joint_best=np.zeros(n,np.uint64)
        if cores:
            lengths=np.asarray(arouter.length);icepaths=np.asarray(arouter.path_ice,np.uint64)
            for ri,route in enumerate(ROUTES,1):
                if route not in cores or cores[route]['mask']&mask:continue
                c=cores[route]
                for ori,(left,right) in enumerate((c['pair'],c['pair'][::-1]),1):
                    p,q=ap[left['offset']+ss],ap[right['offset']+tt]
                    candidate=lengths[p]+c['km']+lengths[q]
                    joint=icepaths[p]&icepaths[q]&c['ice']
                    potential=valid & np.isfinite(candidate) & (candidate<best-1e-6) & ((1+cfg['premium'])*candidate<without-1e-6) & (bit_count(joint)>=30)
                    for j in np.flatnonzero(potential):
                        if not geometry_ok(int(p[j]),int(q[j]),route,ori):potential[j]=False
                    best[potential]=candidate[potential]
                    entry[potential]=p[potential];exit_[potential]=q[potential]
                    routecode[potential]=ri;orientation[potential]=ori;joint_best[potential]=joint[potential]
        saved[mask]=dict(cp=cp,ap=ap,without=without,arctic=best,entry=entry,exit=exit_,
                         routecode=routecode,orientation=orientation,joint=joint_best)
        print(json.dumps(dict(month=month,state=si+1,total=len(masks),seconds=round(time.monotonic()-started),
            conventional_trees=conventional.dijkstra_trees,arctic_trees=arouter.dijkstra_trees)),flush=True)
    open_mask=np.asarray(conventional.path_mask,np.uint32)[saved[0]['cp']]
    nstates=len(scenarios)
    arrays={name:np.empty((nstates,n),dtype=dtype) for name,dtype in [
        ('without_km',float),('arctic_km',float),('A',bool),('B',bool),('U',bool),('exposed',bool),
        ('service_allowed',bool),('cost_proxy',float),('conventional_id',np.int32),('entry_id',np.int32),
        ('exit_id',np.int32),('route_code',np.int32),('orientation',np.int32),('joint_ice',np.uint64)]}
    rows=[]
    for i,case in enumerate(scenarios):
        s=saved[case['closed_mask']]
        service=~(base.origin_port_id.isin(case['ports'])|base.dest_port_id.isin(case['ports'])).to_numpy()
        a,b,u,cost=choose(s['without'],s['arctic'],service,cfg['premium'])
        exposed=((open_mask & np.uint32(case['closed_mask']))!=0)|~service
        for key,value in [('without_km',s['without']),('arctic_km',s['arctic']),('A',a),('B',b),('U',u),
            ('exposed',exposed),('service_allowed',service),('cost_proxy',cost),('conventional_id',s['cp']),
            ('entry_id',s['entry']),('exit_id',s['exit']),('route_code',s['routecode']),
            ('orientation',s['orientation']),('joint_ice',s['joint'])]:arrays[key][i]=value
        for ship in SHIPS:
            take=base.shiptype.eq(ship).to_numpy()
            w=np.where(take,base.trade_weight_usd.to_numpy(),0.)
            total=w.sum();pct=lambda z:float(100*np.sum(w*np.asarray(z,float))/total)
            first=i-case['stage']
            a0,b0=pct(arrays['A'][first]),pct(arrays['B'][first])
            prev=i-1 if case['stage'] else i
            effect=changed(cost,arrays['cost_proxy'][prev]) | (b!=arrays['B'][prev]) | (service!=arrays['service_allowed'][prev])
            bp,ap,up=pct(b),pct(a),pct(u)
            rows.append(dict(class_id=case['class_id'],shiptype=ship,month=month,stage=case['stage'],state_row=i,
                new_unit=case['new_unit'],cumulative_units=' + '.join(case['units']),original_weight_usd=float(total),
                od_count=int(take.sum()),p_exposed_pct=pct(exposed),A_pct=ap,B_pct=bp,U_pct=up,
                denied_service_pct=pct(~service),new_Arctic_flow_pct=pct(b & ~arrays['B'][first]),
                marginal_Arctic_share_pp=bp-pct(arrays['B'][prev]),marginal_changed_trade_pct=pct(effect),
                effective_added_unit=bool(case['stage'] and np.any(effect&take)),A0_pct=a0,B0_pct=b0,
                crossover=bool(a0>b0+1e-8 and bp>ap+1e-8),B_exceeds_original_half=bp>50,
                status='HYPOTHETICAL_GEOGRAPHIC_OR_SERVICE_PRESSURE_TEST'))
    metrics=pd.DataFrame(rows)
    metrics['effective_stage']=metrics.groupby(['class_id','shiptype']).effective_added_unit.cumsum().astype(int)
    assert np.max(np.abs(metrics.A_pct+metrics.B_pct+metrics.U_pct-100))<1e-8
    assert all(metrics.groupby(['class_id','shiptype']).p_exposed_pct.diff().dropna()>=-1e-8)
    metrics.to_csv(folder/'metrics.csv',index=False)
    pure={mask:COMBO_INDEX[mask] for mask in masks if mask<(1<<24) and mask in COMBO_INDEX}
    old=pd.read_parquet(ROOT/f'28_MainFig3_FiveClosures/month_{month:02d}/scenario_metrics.parquet')
    old=old[old.scope.eq('trade_covered')&old.scenario_index.isin(pure.values())].set_index(['scenario_index','shiptype'])
    errors=[]
    for i,case in enumerate(scenarios):
        if case['closed_mask'] not in pure or case['ports']:continue
        for ship in SHIPS:
            row=metrics[(metrics.state_row==i)&metrics.shiptype.eq(ship)].iloc[0]
            ref=old.loc[(pure[case['closed_mask']],ship)]
            errors.extend(abs(float(row[a])-float(ref[b])) for a,b in [
                ('A_pct','A_weight_pct'),('B_pct','B_weight_pct'),('U_pct','unresolved_pct'),('p_exposed_pct','p_exposed_pct')])
    max_error=float(max(errors,default=0))
    assert max_error<1e-6,('Cached full-search disagreement',month,max_error)
    np.savez_compressed(folder/'od_states.npz',**arrays)
    conventional.save_catalogue(folder/'conventional')
    arouter.save_catalogue(folder/'arctic')
    (folder/'scenarios.json').write_text(json.dumps(scenarios,indent=2))
    marker.write_text(json.dumps(dict(status='PASS',month=month,od_records=n,unique_route_states=len(masks),
        available_cores=list(cores),cached_full_search_max_share_error_pp=max_error,
        numerical_share_conservation_pp=float(np.max(np.abs(metrics.A_pct+metrics.B_pct+metrics.U_pct-100))),
        conventional_trees=conventional.dijkstra_trees,arctic_trees=arouter.dijkstra_trees,
        elapsed_seconds=time.monotonic()-started,inputs=signature),indent=2))
    print('MONTH COMPLETE '+str(month),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--months',type=int,nargs='*',default=[1,7,9,8,10,11,12,2,3,4,5,6])
    for month in p.parse_args().months:run_month(month)
