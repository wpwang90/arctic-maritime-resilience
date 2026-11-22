"""Fixed-denominator thresholds and country-OD cluster bootstrap."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

TASK=Path(__file__).resolve().parent
ROOT=TASK.parent
OUT=TASK/'results'
FIG=TASK/'figures'
SHIPS=['container','bulk','tanker','lng']
LABELS={'container':'Container','bulk':'Dry bulk','tanker':'Tanker','lng':'LNG'}
CLASSES=['T1','T2','T3','T4']
CLASS_LABELS={'T1':'Security/conflict','T2':'Weather/hydrology','T3':'Coastal exclusion','T4':'Infrastructure/logistics'}
N_BOOT=2000


def cluster_bootstrap(base,arrays,indices,ship,seed):
    take=base.shiptype.eq(ship).to_numpy()
    demand=base[take]
    labels,_=pd.factorize(pd.MultiIndex.from_frame(demand[['origin_iso3','dest_iso3']]))
    n=int(labels.max()+1)
    w=demand.trade_weight_usd.to_numpy()
    groups=np.bincount(labels,weights=w,minlength=n)
    info=dict(country_pair_clusters=n,bootstrap_draws=N_BOOT,
        scope='Resampling represented directed country OD, not climate, event likelihood or coverage-bias uncertainty')
    if n<5:
        return pd.DataFrame([dict(stage=k,**info,bootstrap_status='INSUFFICIENT_CLUSTERS') for k in range(6)]),dict(**info,bootstrap_status='INSUFFICIENT_CLUSTERS')
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(n,np.full(n,1/n),size=N_BOOT)
    denominator=np.einsum('ij,j->i',counts,groups,optimize=False)
    draw={}
    for name in ['A','B','U','exposed']:
        sums=np.vstack([np.bincount(labels,weights=w*arrays[name][i,take],minlength=n) for i in indices])
        draw[name]=100*np.einsum('ij,kj->ik',counts,sums,optimize=False)/denominator[:,None]
    crossed=(draw['A'][:,0]>draw['B'][:,0]+1e-8)[:,None] & (draw['B']>draw['A']+1e-8)
    crossed[:,0]=False
    rows=[]
    for k in range(6):
        row=dict(stage=k,**info,bootstrap_status='RESAMPLED')
        for key in ['A','B','U','exposed']:
            lo,hi=np.quantile(draw[key][:,k],[.025,.975])
            row[key+'_lo_pct']=float(lo);row[key+'_hi_pct']=float(hi)
        gap=draw['B'][:,k]-draw['A'][:,k]
        row['gap_lo_pp'],row['gap_hi_pp']=[float(x) for x in np.quantile(gap,[.025,.975])]
        row['resampled_trigger_frequency']=float(crossed[:,k].mean())
        row['resampled_B_gt_A_frequency']=float((gap>1e-8).mean())
        rows.append(row)
    any_cross=crossed.any(axis=1)
    earliest=np.argmax(crossed,axis=1)
    conditional_p=draw['exposed'][np.flatnonzero(any_cross),earliest[any_cross]]
    summary=dict(**info,bootstrap_status='RESAMPLED',resampled_any_trigger_frequency=float(any_cross.mean()))
    if any_cross.any():
        summary['conditional_first_stage_lo'],summary['conditional_first_stage_hi']=[float(x) for x in np.quantile(earliest[any_cross],[.025,.975])]
        summary['conditional_p_star_lo'],summary['conditional_p_star_hi']=[float(x) for x in np.quantile(conditional_p,[.025,.975])]
    return pd.DataFrame(rows),summary


def main():
    FIG.mkdir(exist_ok=True)
    metrics=[];intervals=[];boot_summary=[];route_rows=[];audits=[];forecast_rows=[]
    forecast_path=ROOT/'41_MaritimeShare_2024_ML/results/sensitivity_od_weights_NOT_APPLIED.parquet'
    forecast=pd.read_parquet(forecast_path).set_index('od_id')
    for month in range(1,13):
        folder=OUT/f'month_{month:02d}'
        audit=json.loads((folder/'audit.json').read_text());assert audit['status']=='PASS'
        audits.append(audit)
        frame=pd.read_csv(folder/'metrics.csv')
        base=pd.read_parquet(folder/'demand.parquet')
        arrays=np.load(folder/'od_states.npz')
        fw=forecast.loc[base.od_id,'sensitivity_weight_usd'].to_numpy(float)
        assert np.all(np.isfinite(fw)) and np.all(fw>=0)
        for r in frame.itertuples():
            take=base.shiptype.eq(r.shiptype).to_numpy()
            w=np.where(take,fw,0.)
            total=np.sum(w)
            pct=lambda key:float(100*np.sum(w*arrays[key][r.state_row])/total)
            forecast_rows.append(dict(class_id=r.class_id,shiptype=r.shiptype,month=month,stage=r.stage,
                A_pct=pct('A'),B_pct=pct('B'),U_pct=pct('U'),p_exposed_pct=pct('exposed'),
                weight_specification='2024_modal_forecast_where_available_otherwise_existing_2021_fallback'))
        for ci,category in enumerate(CLASSES):
            for vi,ship in enumerate(SHIPS):
                g=frame[frame.class_id.eq(category)&frame.shiptype.eq(ship)].sort_values('stage')
                indices=g.state_row.to_numpy(int)
                ci_rows,summary=cluster_bootstrap(base,arrays,indices,ship,20261101+month*100+ci*10+vi)
                ci_rows['class_id']=category;ci_rows['shiptype']=ship;ci_rows['month']=month
                intervals.append(ci_rows)
                boot_summary.append(dict(class_id=category,shiptype=ship,month=month,**summary))
                take=base.shiptype.eq(ship).to_numpy();w=base.trade_weight_usd.to_numpy()
                for r in g.itertuples():
                    for code,route in enumerate(['NWP','NEP','NSR','TSR'],1):
                        mask=take & arrays['B'][r.state_row] & (arrays['route_code'][r.state_row]==code)
                        value=float(np.sum(w[mask]))
                        route_rows.append(dict(class_id=category,shiptype=ship,month=month,stage=r.stage,route=route,
                            selected_trade_weight_usd=value,pct_original_trade=100*value/float(np.sum(w[take]))))
        metrics.append(frame)
    data=pd.concat(metrics,ignore_index=True)
    ci=pd.concat(intervals,ignore_index=True)
    data=data.merge(ci,on=['class_id','shiptype','month','stage'],how='left',validate='one_to_one')
    assert data[['A_pct','B_pct','U_pct','p_exposed_pct']].apply(np.isfinite).all().all()
    assert np.max(np.abs(data.A_pct+data.B_pct+data.U_pct-100))<1e-8
    data.to_csv(OUT/'four_class_monthly_shares.csv',index=False)
    pd.DataFrame(boot_summary).to_csv(OUT/'country_OD_bootstrap_thresholds.csv',index=False)
    pd.DataFrame(route_rows).to_csv(OUT/'four_passage_composition.csv',index=False)
    thresholds=[]
    for (category,ship),g in data.groupby(['class_id','shiptype']):
        crosses=g[g.crossover & g.stage.gt(0)]
        best=g.assign(gap=g.B_pct-g.A_pct).sort_values(['gap','p_exposed_pct'],ascending=[False,True]).iloc[0]
        row=dict(class_id=category,shiptype=ship,status='CROSSOVER_DETECTED' if len(crosses) else 'NOT_DETECTED_WITHIN_TESTED_STEPS',
            crossing_months='|'.join(str(x) for x in sorted(crosses.month.unique())),
            minimum_nominal_stage=int(crosses.stage.min()) if len(crosses) else np.nan,
            maximum_B_pct=float(g.B_pct.max()),maximum_gap_pp=float((g.B_pct-g.A_pct).max()),
            gap_max_month=int(best.month),gap_max_stage=int(best.stage),gap_max_A_pct=float(best.A_pct),
            gap_max_B_pct=float(best.B_pct),gap_max_U_pct=float(best.U_pct))
        if len(crosses):
            r=crosses.sort_values(['p_exposed_pct','stage','month']).iloc[0]
            for key in ['month','stage','effective_stage','p_exposed_pct','A_pct','B_pct','U_pct',
                'new_Arctic_flow_pct','marginal_Arctic_share_pp','gap_lo_pp','gap_hi_pp',
                'resampled_trigger_frequency','resampled_B_gt_A_frequency','country_pair_clusters']:
                row['selected_'+key]=r[key] if key in r else np.nan
            row['selected_units']=r.cumulative_units
            row['bootstrap_gap_95pct_above_zero']=bool(r.get('gap_lo_pp',np.nan)>0)
        thresholds.append(row)
    summary=pd.DataFrame(thresholds)
    summary.to_csv(OUT/'threshold_summary_by_shiptype.csv',index=False)
    forecast_rows=pd.DataFrame(forecast_rows)
    original=forecast_rows[forecast_rows.stage.eq(0)][['class_id','shiptype','month','A_pct','B_pct']].rename(columns={'A_pct':'A0_pct','B_pct':'B0_pct'})
    forecast_rows=forecast_rows.merge(original,on=['class_id','shiptype','month'],validate='many_to_one')
    forecast_rows['crossover']=(forecast_rows.A0_pct>forecast_rows.B0_pct+1e-8)&(forecast_rows.B_pct>forecast_rows.A_pct+1e-8)
    forecast_rows.to_csv(OUT/'forecast2024_weight_sensitivity.csv',index=False)
    fs=[]
    for (category,ship),g in forecast_rows.groupby(['class_id','shiptype']):
        crossed=g[g.crossover & g.stage.gt(0)]
        row=dict(class_id=category,shiptype=ship,crossover_detected=bool(len(crossed)),
                 crossing_months='|'.join(str(x) for x in sorted(crossed.month.unique())))
        if len(crossed):
            r=crossed.sort_values(['p_exposed_pct','stage','month']).iloc[0]
            for key in ['month','stage','A_pct','B_pct','U_pct','p_exposed_pct']:row[key]=r[key]
        fs.append(row)
    pd.DataFrame(fs).to_csv(OUT/'forecast2024_threshold_sensitivity.csv',index=False)
    pooled=[]
    for (category,ship,stage),g in data.groupby(['class_id','shiptype','stage']):
        total=g.original_weight_usd.sum()
        row=dict(class_id=category,shiptype=ship,stage=stage,weight_usd=float(total),
            interpretation='Pooled same-stress monthly counterfactual, not an observed annual event')
        for key in ['A_pct','B_pct','U_pct','p_exposed_pct','new_Arctic_flow_pct']:
            row[key]=float(np.sum(g.original_weight_usd*g[key])/total)
        pooled.append(row)
    pooled=pd.DataFrame(pooled)
    pooled['crossover']=(pooled.B_pct>pooled.A_pct+1e-8)
    pooled.to_csv(OUT/'pooled_twelve_month_shares.csv',index=False)
    plot(data,summary)
    report=dict(status='COMPUTED_HYPOTHETICAL_PRESSURE_TEST',months=12,shiptypes=4,classes=4,
        monthly_stage_rows=len(data),unique_trade_covered_OD_records=int(sum(a['od_records'] for a in audits)),
        maximum_cached_global_search_share_disagreement_pp=float(max(a['cached_full_search_max_share_error_pp'] for a in audits)),
        maximum_conservation_residual_pp=float(np.max(np.abs(data.A_pct+data.B_pct+data.U_pct-100))),
        bootstrap=dict(draws=N_BOOT,cluster='directed origin-country/destination-country within ship/month',
            caveats='Sampling resampling only; not climate/event/coverage uncertainty; conditional threshold intervals exclude nondetections'),
        pressure_test_not_historical_reconstruction=True,arctic_premium_pct=5,
        climate='Existing SSP585 2091-2100 60 model-years, PC6-core assumption and >=50% common voyage support',
        main_weight_scope='Covered trade cohort, not entire global trade or full fleet',
        monetary_cost_model_calibrated=False,
        forecast_modal_weight_sensitivity=str(forecast_path.relative_to(ROOT)),
        findings=summary.astype(object).where(pd.notna(summary),None).to_dict('records'))
    (OUT/'analysis_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(summary.to_string(index=False),flush=True)
    print('POOLED CROSSOVERS',pooled[pooled.crossover].to_string(index=False),flush=True)


def plot(data,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    colors={'A':'#2476A5','B':'#D66A32','U':'#9A9EA3'}
    fig,axes=plt.subplots(4,4,figsize=(12.8,9.7),sharex=True,sharey=True)
    selection=[]
    for ri,ship in enumerate(SHIPS):
        for col,category in enumerate(CLASSES):
            ax=axes[ri,col]
            row=summary[(summary.class_id==category)&(summary.shiptype==ship)].iloc[0]
            month=int(row.selected_month) if row.status=='CROSSOVER_DETECTED' else int(row.gap_max_month)
            g=data[(data.class_id==category)&(data.shiptype==ship)&(data.month==month)].sort_values('stage')
            for key in ['A','B','U']:
                ax.plot(g.stage,g[key+'_pct'],color=colors[key],lw=1.7,marker='o',ms=3,ls='--' if key=='U' else '-')
                if key+'_lo_pct' in g:
                    ax.fill_between(g.stage,g[key+'_lo_pct'],g[key+'_hi_pct'],color=colors[key],alpha=.09,lw=0)
            crossed=g[g.crossover]
            if len(crossed):ax.axvline(crossed.stage.min(),color='#444444',lw=.8,ls=':')
            ax.set_ylim(0,103);ax.set_xlim(-.12,5.12);ax.set_xticks(range(6));ax.set_yticks([0,25,50,75,100])
            ax.grid(axis='y',alpha=.16,lw=.6)
            ax.text(.02,.96,chr(97+ri*4+col),transform=ax.transAxes,weight='bold',va='top')
            ax.text(.98,.96,'Month %02d'%month,transform=ax.transAxes,ha='right',va='top',fontsize=8)
            if ri==0:ax.set_title(CLASS_LABELS[category],fontsize=10,pad=9)
            if col==0:ax.set_ylabel(LABELS[ship]+'\nOriginal covered trade (%)')
            if ri==3:ax.set_xlabel('Cumulative scope stage')
            selection.append(dict(panel=chr(97+ri*4+col),class_id=category,shiptype=ship,month=month,
                reason='minimum exposed share at crossover' if len(crossed) else 'largest tested B-A gap; no detected crossover'))
    fig.legend(handles=[Line2D([0],[0],color=colors[k],lw=2,label=v) for k,v in [
        ('A','Traditional A'),('B','Arctic B'),('U','Unserved U')]],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,.005))
    fig.subplots_adjust(left=.095,right=.985,bottom=.085,top=.96,wspace=.19,hspace=.20)
    for ext in ['png','pdf']:fig.savefig(FIG/('FourClass_Shiptype_ShareCurves.'+ext),dpi=240)
    plt.close(fig)
    pd.DataFrame(selection).to_csv(OUT/'curve_panel_selection.csv',index=False)
    cmap=ListedColormap(['#D8E9EF','#A9D1D4','#67AAA7','#D3A654','#B36239'])
    cmap.set_bad('#E4E4E4')
    fig,axes=plt.subplots(4,1,figsize=(11,7.3),sharex=True)
    for ax,category in zip(axes,CLASSES):
        values=np.full((4,12),np.nan)
        for ri,ship in enumerate(SHIPS):
            for month in range(1,13):
                g=data[(data.class_id==category)&(data.shiptype==ship)&(data.month==month)&data.crossover]
                if len(g):values[ri,month-1]=g.stage.min()
        shown=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,norm=BoundaryNorm(np.arange(.5,6.5),5),aspect='auto')
        for y in range(4):
            for x in range(12):ax.text(x,y,str(int(values[y,x])) if np.isfinite(values[y,x]) else '-',ha='center',va='center',fontsize=10)
        ax.set_yticks(range(4),[LABELS[s] for s in SHIPS]);ax.set_ylabel(category,rotation=0,labelpad=22,weight='bold')
        ax.set_xticks(np.arange(-.5,12,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
        ax.grid(which='minor',color='white',lw=1.1);ax.tick_params(which='both',length=0)
    axes[-1].set_xticks(range(12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    axes[-1].set_xlabel('Calendar month in the fixed future-climate experiment')
    fig.subplots_adjust(left=.18,right=.98,bottom=.18,top=.98,hspace=.18)
    cax=fig.add_axes([.37,.055,.38,.017])
    fig.colorbar(shown,cax=cax,orientation='horizontal',ticks=range(1,6),label='First nominal scope stage with B > A')
    fig.text(.18,.045,'Grey / - : not detected',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(FIG/('FourClass_Crossover_Calendar.'+ext),dpi=230)
    plt.close(fig)


if __name__=='__main__':main()
