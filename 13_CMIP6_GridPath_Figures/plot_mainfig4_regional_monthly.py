"""World-map matrix and aligned, full-year economic-loss diagnostics."""
import calendar
import json
import shutil
from datetime import datetime

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely

from rebuild_mainfig4_regional_monthly import ROOT, OUT, OLD, SHIPS, CHANNELS, sha

LABELS = dict(container='Container',bulk='Dry bulk',tanker='Tanker',lng='LNG')
PALETTES = dict(container='Blues',bulk='BuGn',tanker='YlOrBr',lng='RdPu')
COLORS = dict(container='#246fa7',bulk='#16806c',tanker='#c57919',lng='#9a3978')
CASECOLORS = {'without_arctic':'#0077b6','with_arctic':'#df751d'}
CHANNEL_COLORS = ['#466d9e','#dcb75c','#ba6680']
MONTHS = [calendar.month_abbr[m] for m in range(1,13)]
FIG = OUT/'figures'
PROJ = '+proj=robin +lon_0=0 +datum=WGS84 +units=m'


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','hatch.linewidth':.25})
    FIG.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(FIG/f'{name}.png',dpi=260,facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',facecolor='white')
    plt.close(fig)


def geography():
    nations = gpd.read_file(ROOT/'data/public/naturalearth_10m_countries/ne_10m_admin_0_countries.shp')
    nations = nations[nations.ADM0_A3.ne('ATA')].copy()
    nations['country_iso3'] = nations.ISO_A3
    for column in ['ISO_A3_EH','ADM0_A3','SOV_A3']:
        if column in nations:
            missing = nations.country_iso3.eq('-99') | nations.country_iso3.isna()
            nations.loc[missing,'country_iso3'] = nations.loc[missing,column]
    nations.geometry = nations.geometry.intersection(shapely.box(-180,-60,180,89.9)).simplify(.065)
    nations = nations[~nations.geometry.is_empty].to_crs(PROJ)
    return nations


def country_matrix(fig, nations, country, ports, design='common_five', yshift=0):
    bounds = [.01,.1,1,10,100,1000]
    transformer = Transformer.from_crs('EPSG:4326',PROJ,always_xy=True)
    for j, ship in enumerate(SHIPS):
        x = .06+j*.235
        fig.text(x+.108,.958+yshift,LABELS[ship],ha='center',weight='bold',fontsize=12)
        cmap = ListedColormap(plt.get_cmap(PALETTES[ship])(np.linspace(.18,.91,256)))
        norm = LogNorm(vmin=bounds[0],vmax=bounds[-1],clip=True)
        for k,case in enumerate(['without_arctic','with_arctic']):
            ax = fig.add_axes([x,.705-k*.255+yshift,.215,.235])
            part = country[(country.design.eq(design)) & country.shiptype.eq(ship) & country.case.eq(case)]
            values = part.set_index('dest_iso3').total_loss_usd/1e6
            frame = nations.copy()
            frame['loss'] = frame.country_iso3.map(values)
            frame.plot(ax=ax,color='#fcfcfc',edgecolor='#d6d9dc',linewidth=.20,hatch='///')
            zero = frame.loss.eq(0)
            positive = frame.loss.gt(0)
            if zero.any():
                frame[zero].plot(ax=ax,color='#bcbfc2',edgecolor='white',linewidth=.25)
            if positive.any():
                frame[positive].plot(ax=ax,column='loss',cmap=cmap,norm=norm,edgecolor='white',linewidth=.28)
            pp = ports[(ports.design.eq(design)) & ports.shiptype.eq(ship) & ports.case.eq(case)]
            xy = transformer.transform(pp.dest_lon.to_numpy(),pp.dest_lat.to_numpy())
            ax.scatter(*xy,s=2,color='#242a31',alpha=.48,linewidths=0,zorder=5,rasterized=True)
            ax.set_xlim(-17100000,17100000); ax.set_ylim(-6500000,8750000)
            ax.set_axis_off()
            total = part.total_loss_usd.sum()/1e9
            other = country[(country.design.eq(design)) & country.shiptype.eq(ship) & country.case.eq('without_arctic')].total_loss_usd.sum()/1e9
            text = f'US${total:.2f} bn'
            if k:
                text += f'  |  {100*(other-total)/other:.1f}% avoided' if other else '  |  no baseline loss'
            ax.text(.5,-.034,text,ha='center',va='top',transform=ax.transAxes,fontsize=9,color='#25313b')
        cax = fig.add_axes([x+.025,.408+yshift,.17,.010])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,
                          orientation='horizontal',ticks=bounds,spacing='uniform')
        cb.set_ticklabels(['<0.01','0.1','1','10','100','1,000'])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=7,length=2,pad=2)
        cb.outline.set_linewidth(.4)
    fig.text(.018,.82+yshift,'Without Arctic',rotation=90,va='center',weight='bold',color=CASECOLORS['without_arctic'],fontsize=11)
    fig.text(.018,.565+yshift,'With Arctic',rotation=90,va='center',weight='bold',color=CASECOLORS['with_arctic'],fontsize=11)
    fig.text(.525,.376+yshift,'Mean 30-day event loss attributed to receiving-port countries (US$ million)',ha='center',fontsize=9)
    fig.legend(handles=[Patch(facecolor='#bcbfc2',label='Covered; zero loss'),
        Patch(facecolor='#fcfcfc',edgecolor='#c5c9cc',hatch='///',label='Not covered / not displayed'),
        Line2D([],[],ls='',marker='o',markersize=3,color='#252c32',label='Covered receiving port')],
        loc='center',bbox_to_anchor=(.525,.350+yshift),ncol=3,frameon=False,fontsize=8)


def main_figure():
    country = pd.read_csv(OUT/'country_mean_event_losses.csv')
    ports = pd.read_csv(OUT/'port_mean_event_losses.csv')
    monthly = pd.read_csv(OUT/'monthly_loss_channels.csv')
    monthly = monthly[monthly.design.eq('common_five')]
    fig = plt.figure(figsize=(18,9.5),facecolor='white')
    fig.text(.025,.978,'a',fontweight='bold',fontsize=17)
    country_matrix(fig,geography(),country,ports)
    positions = [[.065,.095,.273,.215],[.404,.095,.267,.215],[.741,.095,.239,.215]]
    axes = [fig.add_axes(p) for p in positions]
    for ax,letter in zip(axes,'bcd'):
        ax.text(-.095,1.065,letter,transform=ax.transAxes,fontsize=16,fontweight='bold')
    # b: paired monthly stacks; cases share a single physical stress scenario.
    ax = axes[0]
    group = monthly.groupby(['start_month','case'])[CHANNELS].sum()
    for case,offset,hatch in [('without_arctic',-.19,'///'),('with_arctic',.19,None)]:
        bottom = np.zeros(12)
        for col,color in zip(CHANNELS,CHANNEL_COLORS):
            value = group.xs(case,level='case')[col].reindex(range(1,13)).to_numpy()/1e9
            ax.bar(np.arange(12)+offset,value,.34,bottom=bottom,color=color,hatch=hatch,
                   edgecolor='#ffffff' if hatch is None else '#46525c',linewidth=.28)
            bottom += value
    ax.set_xticks(range(12),MONTHS,rotation=60,ha='right'); ax.set_xlim(-.65,11.65)
    ax.set_ylabel('Conditional event loss (US$ bn)')
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(CHANNEL_COLORS,
        ['Transport + insurance','Delay','Stock-out proxy'])],
        loc='lower left',bbox_to_anchor=(-.015,1.005),ncol=2,frameon=False,fontsize=7.5,columnspacing=.8,handlelength=1)
    fig.legend(handles=[Patch(facecolor='white',edgecolor='#46525c',hatch='///',label='Without Arctic'),
        Patch(facecolor='#a5aeb6',label='With Arctic')],loc='center',bbox_to_anchor=(.20,.017),ncol=2,frameon=False,fontsize=8)
    # c: full 12-month relative loss avoided. Grey means a computed zero.
    ax = axes[1]
    p = monthly.pivot(index=['shiptype','start_month'],columns='case',values='total_loss_usd')
    p['avoided_pct'] = 100*(p.without_arctic-p.with_arctic)/p.without_arctic.replace(0,np.nan)
    mat = p.avoided_pct.unstack().reindex(index=SHIPS,columns=range(1,13)).to_numpy()
    cmap = plt.get_cmap('YlGnBu').copy(); cmap.set_bad('#bcbfc2')
    img = ax.imshow(np.ma.masked_where(np.isclose(mat,0)|~np.isfinite(mat),mat),aspect='auto',cmap=cmap,vmin=0,vmax=100)
    ax.set_xticks(range(12),MONTHS,rotation=60,ha='right'); ax.set_yticks(range(4),[LABELS[s] for s in SHIPS])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5,12,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.8); ax.tick_params(which='minor',length=0)
    for i in range(4):
        for j in range(12):
            if not np.isfinite(mat[i,j]):
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='white',edgecolor='#c6ccd0',hatch='///',linewidth=.4))
            text = f'{mat[i,j]:.1f}' if np.isfinite(mat[i,j]) else 'NA'
            ax.text(j,i,text,ha='center',va='center',fontsize=6.8,color='white' if mat[i,j]>55 else '#182b39')
    cax = fig.add_axes([.463,.024,.15,.009])
    cb = fig.colorbar(img,cax=cax,orientation='horizontal',ticks=[0,25,50,75,100])
    cb.ax.tick_params(labelsize=7,length=2); cb.outline.set_linewidth(.4)
    cb.ax.xaxis.set_label_position('top'); cb.set_label('Loss avoided (%)',fontsize=8,labelpad=2)
    # d: absolute monthly losses, preserving vessel heterogeneity.
    ax = axes[2]
    for ship in SHIPS:
        pp = monthly[monthly.shiptype.eq(ship)]
        for case,ls in [('without_arctic','--'),('with_arctic','-')]:
            y = pp[pp.case.eq(case)].set_index('start_month').total_loss_usd.reindex(range(1,13))/1e9
            ax.plot(range(12),y,color=COLORS[ship],ls=ls,lw=1.5 if case=='with_arctic' else 1.0,
                    alpha=1 if case=='with_arctic' else .65,marker='o' if case=='with_arctic' else None,markersize=2)
    ax.set_xticks(range(12),MONTHS,rotation=60,ha='right'); ax.set_xlim(-.2,11.2)
    ax.set_ylabel('Conditional event loss (US$ bn)')
    ax.set_ylim(bottom=0); ax.grid(axis='y',alpha=.18)
    ax.legend(handles=[Line2D([],[],color=COLORS[s],lw=2,label=LABELS[s]) for s in SHIPS],
        loc='lower left',bbox_to_anchor=(-.005,1.005),ncol=2,frameon=False,fontsize=7.7,columnspacing=.8)
    fig.legend(handles=[Line2D([],[],color='#414b54',ls='--',label='Without Arctic'),
        Line2D([],[],color='#414b54',ls='-',marker='o',markersize=3,label='With Arctic')],
        loc='center',bbox_to_anchor=(.863,.017),ncol=2,frameon=False,fontsize=8)
    p.reset_index().to_csv(OUT/'panel_c_values.csv',index=False)
    save(fig,'MainFig4_RegionalMonthlyLoss')


def supplementary():
    regions = ['Africa','Asia','Europe','North America','Oceania','South America']
    region = pd.read_csv(OUT/'region_mean_event_losses.csv')
    summary = pd.read_csv(OUT/'summary.csv')
    figs = []
    for design,suffix in [('common_five','CommonStress'),('fig3_fixed','Fig3StressCases')]:
        fig,axes = plt.subplots(2,2,figsize=(12.5,8.5))
        fig.subplots_adjust(left=.12,right=.965,bottom=.12,top=.91,wspace=.30,hspace=.35)
        for i,(ax,ship) in enumerate(zip(axes.flat,SHIPS)):
            g = region[(region.design.eq(design)) & region.shiptype.eq(ship)]
            for case,offset in [('without_arctic',-.17),('with_arctic',.17)]:
                v = g[g.case.eq(case)].set_index('dest_region').total_loss_usd.reindex(regions)
                ax.barh(np.arange(6)+offset,v/1e9,height=.30,color=CASECOLORS[case])
            ax.set_yticks(range(6),regions); ax.invert_yaxis(); ax.grid(axis='x',alpha=.17)
            ax.set_xlabel('Mean event loss (US$ bn)')
            row = summary[summary.design.eq(design)&summary.shiptype.eq(ship)].iloc[0]
            ax.text(-.02,1.08,f'{chr(97+i)}  {LABELS[ship]}',transform=ax.transAxes,fontsize=12,weight='bold')
            ax.text(1,1.08,f'{row.avoided_pct:.1f}% avoided',ha='right',transform=ax.transAxes,fontsize=10)
        fig.legend(handles=[Patch(facecolor=CASECOLORS[c],label=l) for c,l in
            [('without_arctic','Without Arctic'),('with_arctic','With Arctic')]],
            loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
        save(fig,f'SI_Fig4_Regional_{suffix}')
    sensitivity = pd.read_csv(OUT/'SI_parameter_sensitivity.csv')
    central = pd.read_csv(OUT/'panel_c_values.csv').set_index(['shiptype','start_month'])
    fig,axes = plt.subplots(2,2,figsize=(12.5,7.8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.065,right=.98,top=.91,bottom=.13,hspace=.29,wspace=.10)
    for i,(ax,ship) in enumerate(zip(axes.flat,SHIPS)):
        s = sensitivity[sensitivity.shiptype.eq(ship)].groupby('month').avoided_pct.agg(['min','max']).reindex(range(1,13))
        central_values = central.loc[ship].avoided_pct.reindex(range(1,13))
        ax.fill_between(range(12),s['min'],s['max'],color=COLORS[ship],alpha=.22)
        ax.plot(range(12),central_values,color=COLORS[ship],marker='o',lw=1.5,markersize=3)
        ax.set_xticks(range(12),MONTHS,rotation=45,ha='right'); ax.set_ylim(-1,101)
        ax.grid(axis='y',alpha=.18)
        ax.text(0,1.08,f'{chr(97+i)}  {LABELS[ship]}',transform=ax.transAxes,weight='bold',fontsize=12)
        if i%2==0:
            ax.set_ylabel('Loss avoided (%)')
    fig.legend(handles=[Patch(facecolor='#aebbc7',alpha=.5,label='Deterministic parameter range (not a confidence interval)'),
        Line2D([],[],color='#293e50',marker='o',label='30 days; 16 kn; 5% Arctic premium')],
        loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.5,.005),fontsize=9)
    save(fig,'SI_Fig4_ParameterSensitivity')
    raw = pd.read_parquet(OLD/'baseline_all.parquet')
    od = pd.read_parquet(OUT/'paired_od_losses.parquet')
    od = od[od.design.eq('common_five') & od.case.eq('without_arctic')]
    coverage = raw.groupby('start_month').size().rename('all_represented_od').to_frame()
    coverage['trade_covered_od'] = od.groupby('start_month').size()
    coverage['allocated_trade_usd'] = od.groupby('start_month').trade_weight_usd.sum()
    coverage.to_csv(OUT/'SI_monthly_sample_coverage.csv')
    fig,axes = plt.subplots(1,2,figsize=(12.5,4.5))
    fig.subplots_adjust(left=.075,right=.97,bottom=.20,top=.86,wspace=.26)
    ax = axes[0]
    ax.bar(np.arange(12)-.18,coverage.all_represented_od.reindex(range(1,13)),.34,color='#acb9c3',label='All represented OD')
    ax.bar(np.arange(12)+.18,coverage.trade_covered_od.reindex(range(1,13)),.34,color='#416b89',label='Trade-covered OD')
    ax.set_ylabel('Vessel-type / OD / month records')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=8)
    axes[1].bar(range(12),coverage.allocated_trade_usd.reindex(range(1,13))/1e9,color='#6b889c')
    axes[1].set_ylabel('Allocated trade value (US$ bn)')
    for i,ax in enumerate(axes):
        ax.set_xticks(range(12),MONTHS,rotation=45,ha='right'); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
        ax.text(-.04,1.09,'ab'[i],transform=ax.transAxes,fontweight='bold',fontsize=14)
    save(fig,'SI_Fig4_MonthlyCoverage')


def publish():
    validation=json.loads((OUT/'independent_validation.json').read_text())
    assert validation['status']=='PASS'
    target = ROOT/'figures'
    backup = OUT/'previous_figure_backup'
    backup.mkdir(exist_ok=True)
    for extension in ['png','pdf']:
        dest = target/f'EthanAIS_MainFig4_EconomicLosses.{extension}'
        if dest.exists() and not (backup/dest.name).exists():
            shutil.copy2(dest,backup/dest.name)
        shutil.copy2(FIG/f'MainFig4_RegionalMonthlyLoss.{extension}',dest)
    files = [p for p in FIG.iterdir() if p.suffix in ['.pdf','.png']]
    (OUT/'publication_manifest.json').write_text(json.dumps(dict(
        published_at=datetime.now().isoformat(),canonical=str(target/'EthanAIS_MainFig4_EconomicLosses.png'),
        files={str(p.relative_to(ROOT)):sha(p) for p in files},backup=str(backup),
        manuscript_docx='Not modified in this figure-specific request'),indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    style()
    main_figure()
    supplementary()
    if args.publish:
        publish()
