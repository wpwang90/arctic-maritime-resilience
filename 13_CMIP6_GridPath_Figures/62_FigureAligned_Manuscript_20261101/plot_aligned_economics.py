"""Replace economic panels using the exact new Figure-4 route experiments."""
import calendar
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT, FIG = HERE / 'results', HERE / 'figures'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '32_NCC2023_IndirectExposure'))
import plot_figure5 as original
from plot_mainfig4_regional_monthly import geography

SHIPS = ['container', 'bulk', 'tanker', 'lng']
LABELS = dict(container='Container', bulk='Dry bulk', tanker='Tanker', lng='LNG')
SHIP_COLORS = ['#2e7296', '#428c77', '#d68b41', '#a76586']
CHANNELS = ['transport_insurance_usd', 'delay_usd', 'inventory_usd']
SECTORS = original.DISPLAY_GROUPS
NAME = original.name
ZERO = '#c6cbce'
MONTHS = [calendar.month_abbr[i] for i in range(1, 13)]


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(FIG / (name + '.' + ext), dpi=250, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def tag(ax, label):
    ax.text(-.11, 1.035, label, transform=ax.transAxes, fontsize=13, weight='bold', va='bottom')


def country_axis(ax, countries):
    ax.set_xticks(range(len(countries)), [NAME(r) for r in countries], rotation=90, ha='center', fontsize=8)
    ax.tick_params(axis='x', length=0, pad=4)
    ax.set_xlim(-.5, len(countries)-.5)
    for text, r in zip(ax.get_xticklabels(), countries):
        text.set_weight('normal')


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})
    monthly = pd.read_csv(OUT / 'monthly_loss_channels.csv')
    data = pd.read_parquet(OUT / 'mrio_industry_month_vessel_exposure.parquet')
    primary = data[data.primary_case].groupby(['buffer_days', 'case', 'region'])[
        ['first_hit_usd', 'higher_order_usd', 'output_exposure_usd']].sum().div(4).reset_index()
    primary.to_csv(OUT / 'primary_mean_country_exposure.csv', index=False)
    wo = primary[primary.buffer_days.eq(0) & primary.case.eq('without_arctic')].set_index('region')
    wi = primary[primary.buffer_days.eq(0) & primary.case.eq('with_arctic')].set_index('region')
    order = wo[~wo.index.str.startswith('W') & (wo.index != 'TW')].sort_values('output_exposure_usd', ascending=False).head(20).index.tolist()
    country = wo.loc[order].copy()
    country['with_arctic_usd'] = wi.loc[order, 'output_exposure_usd']
    country['avoided_pct'] = 100 * (country.output_exposure_usd - country.with_arctic_usd) / country.output_exposure_usd
    country['name'] = [NAME(r) for r in order]
    country.to_csv(OUT / 'MainFig5_country_order_and_values.csv')
    sources = pd.read_parquet(OUT / 'primary_country_source_receiver_exposure.parquet')
    cross = sources[sources.source_region.ne(sources.receiver_region)]
    pairs = cross.groupby(['case', 'source_region', 'receiver_region']).output_exposure_usd.sum().div(4).reset_index()
    base = pairs[pairs.case.eq('without_arctic')]
    importance = base[base.receiver_region.isin(order)].groupby('source_region').output_exposure_usd.sum().add(
        base[base.source_region.isin(order)].groupby('receiver_region').output_exposure_usd.sum(), fill_value=0)
    partners = importance[~importance.index.str.startswith('W') & (importance.index != 'TW')].nlargest(10).index.tolist() + ['OTHER']
    tables = {}
    for case in ['without_arctic', 'with_arctic']:
        part = pairs[pairs.case.eq(case)].copy()
        part['supplying_partner'] = part.source_region.where(part.source_region.isin(partners), 'OTHER')
        part['receiving_partner'] = part.receiver_region.where(part.receiver_region.isin(partners), 'OTHER')
        incoming = part.groupby(['receiver_region', 'supplying_partner']).output_exposure_usd.sum().unstack().reindex(index=order, columns=partners).fillna(0)
        outgoing = part.groupby(['source_region', 'receiving_partner']).output_exposure_usd.sum().unstack().reindex(index=order, columns=partners).fillna(0)
        tables[case] = outgoing, incoming
        outgoing.to_csv(OUT / ('crossborder_' + case + '_outgoing_usd.csv'))
        incoming.to_csv(OUT / ('crossborder_' + case + '_incoming_usd.csv'))
    denominator = tables['without_arctic'][0].sum(1) + tables['without_arctic'][1].sum(1)
    groupmap = {g: label for label, groups in SECTORS for g in groups}
    selected = data[data.primary_case & data.buffer_days.eq(0)].copy()
    selected['sector'] = selected.group.map(groupmap)
    matrices = [selected[selected.case.eq(case)].groupby(['region', 'sector']).output_exposure_usd.sum().div(4e6).unstack().reindex(
        index=order, columns=[label for label, _ in SECTORS]) for case in ['without_arctic', 'with_arctic']]
    delta = (matrices[0] - matrices[1]).to_numpy().T
    assert delta.min() > -1e-5
    delta = np.maximum(delta, 0)
    rate = np.divide(100 * delta, matrices[0].to_numpy().T, out=np.full_like(delta, np.nan), where=matrices[0].to_numpy().T > 1e-10)
    for values, name in [(delta, 'sector_avoided_million_usd'), (rate, 'sector_avoided_pct')]:
        pd.DataFrame(values, index=[label for label, _ in SECTORS], columns=order).to_csv(OUT / (name + '.csv'))

    fig = plt.figure(figsize=(18, 16.7))
    axes = [fig.add_axes(rect) for rect in [[.065,.785,.25,.17],[.395,.785,.26,.17],[.745,.785,.24,.17],
        [.08,.445,.39,.235],[.585,.445,.39,.235],[.08,.115,.39,.22],[.585,.115,.39,.22]]]
    for ax, label in zip(axes, 'abcdefg'):
        tag(ax, label)
    mean = monthly.groupby(['month', 'case'])[CHANNELS].sum().div(4e9)
    for case, shift, hatch in [('without_arctic', -.19, '///'), ('with_arctic', .19, None)]:
        bottom = np.zeros(12)
        for channel, color in zip(CHANNELS, ['#4c7c9e', '#d5b25d', '#b36d85']):
            values = mean.xs(case, level='case')[channel].reindex(range(1,13)).to_numpy()
            axes[0].bar(np.arange(12)+shift, values, .34, bottom=bottom, color=color, hatch=hatch,
                        edgecolor='#52606b' if hatch else 'white', linewidth=.3)
            bottom += values
    axes[0].set_ylabel('Mean transport-loss proxy\n(US$ bn per class-specific test)')
    axes[0].legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(['#4c7c9e','#d5b25d','#b36d85'],
        ['Transport + insurance', 'Delay', 'Stock-out proxy'])], loc='lower left', bbox_to_anchor=(0,1.02), ncol=2, frameon=False, fontsize=7.5)
    pivot = monthly.pivot(index=['shiptype', 'month'], columns='case', values='total_loss_usd')
    ratios = 100 * (pivot.without_arctic-pivot.with_arctic) / pivot.without_arctic.replace(0,np.nan)
    values = ratios.unstack('month').reindex(index=SHIPS, columns=range(1,13)).to_numpy()
    cm = plt.get_cmap('YlGnBu').copy(); cm.set_bad('#ffffff'); cm.set_under(ZERO)
    im = axes[1].imshow(np.ma.masked_invalid(np.where(np.isclose(values,0), -1e-8, values)), cmap=cm, vmin=1e-10, vmax=100, aspect='auto')
    axes[1].set_yticks(range(4), [LABELS[s] for s in SHIPS]); axes[1].set_xticks(range(12), MONTHS, rotation=60, ha='right')
    for y in range(4):
        for x in range(12):
            axes[1].text(x,y,'--' if not np.isfinite(values[y,x]) else '%.1f'%values[y,x],ha='center',va='center',fontsize=7,
                         color='white' if values[y,x]>55 else '#243641')
    cax = fig.add_axes([.425,.747,.20,.008]); cb=fig.colorbar(im,cax=cax,orientation='horizontal'); cb.set_label('Transport loss avoided (%)',fontsize=8)
    for ship,color in zip(SHIPS,SHIP_COLORS):
        for case,style in [('without_arctic','--'),('with_arctic','-')]:
            y = monthly[monthly.shiptype.eq(ship)&monthly.case.eq(case)].set_index('month').total_loss_usd.reindex(range(1,13))/1e9
            axes[2].plot(range(12),y,color=color,ls=style,marker=None if style=='--' else 'o',ms=2,lw=1.3)
    axes[2].set_ylabel('Transport-loss proxy\n(US$ bn per test)')
    axes[2].legend(handles=[Line2D([],[],color=c,label=LABELS[s]) for s,c in zip(SHIPS,SHIP_COLORS)],ncol=2,
        frameon=False,loc='lower left',bbox_to_anchor=(0,1.02),fontsize=8)
    for ax in [axes[0],axes[2]]:
        ax.set_xticks(range(12), MONTHS, rotation=60, ha='right'); ax.set_ylim(bottom=0)
        ax.grid(axis='y',color='#e3e8eb',lw=.4); ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='white',edgecolor='#52606b',hatch='///',label='Without Arctic'),
                        Patch(facecolor='#a4afb7',label='With Arctic')],bbox_to_anchor=(.19,.735),loc='center',ncol=2,frameon=False,fontsize=8)
    fig.legend(handles=[Line2D([],[],color='#52606b',ls='--',label='Without Arctic'),Line2D([],[],color='#52606b',label='With Arctic')],
        bbox_to_anchor=(.87,.735),loc='center',ncol=2,frameon=False,fontsize=8)

    for ax,case in zip(axes[3:5],['without_arctic','with_arctic']):
        outgoing,incoming = [100*t.div(denominator,axis=0) for t in tables[case]]
        below,above=np.zeros(20),np.zeros(20)
        for partner,color in zip(partners,original.PARTNER_COLORS):
            down,up=outgoing[partner].to_numpy(),incoming[partner].to_numpy()
            ax.bar(range(20),-down,bottom=-below,color=color,width=.65,edgecolor='white',lw=.2)
            ax.bar(range(20),up,bottom=above,color=color,width=.65,edgecolor='white',lw=.2)
            below+=down;above+=up
        ax.scatter(range(20),above-below,s=20,c='#1b242b',edgecolors='white',lw=.2,zorder=5)
        ax.axhline(0,c='#52606b',lw=.6); ax.set_ylim(-105,105)
        ax.set_ylabel('Cross-border exposure (% of baseline)\nReceived (+) / propagated (-)')
        country_axis(ax,order); ax.grid(axis='y',color='#e4e9eb',lw=.4); ax.set_axisbelow(True)
        ax.text(.99,1.035,'Without Arctic' if case=='without_arctic' else 'With Arctic',ha='right',transform=ax.transAxes,fontsize=9)
    legend=[Patch(facecolor=color,label=NAME(partner) if partner!='OTHER' else 'Other partners') for partner,color in zip(partners,original.PARTNER_COLORS)]
    legend.append(Line2D([],[],c='#17232c',marker='o',ls='',label='Net received exposure'))
    fig.legend(handles=legend,loc='center',bbox_to_anchor=(.53,.371),ncol=4,frameon=False,fontsize=8,title='Partner country in d,e',title_fontsize=9)
    for ax,values,kind in zip(axes[5:],[delta,rate],['absolute','relative']):
        cmap=(original.ABSOLUTE_CMAP if kind=='absolute' else original.RELATIVE_CMAP).copy()
        cmap.set_bad('#ffffff');cmap.set_under(ZERO)
        if kind=='absolute':
            positives=values[values>1e-8]; norm=LogNorm(max(positives.min(),1e-6),positives.max())
            im=ax.imshow(np.ma.masked_invalid(np.where(values<=1e-8,norm.vmin/10,values)),norm=norm,cmap=cmap,aspect='auto')
        else:
            im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=1e-10,vmax=100,aspect='auto')
        ax.set_yticks(range(12),[label for label,_ in SECTORS],fontsize=8); country_axis(ax,order)
        box=ax.get_position(); cax=fig.add_axes([box.x0,.043,box.width,.008]); cb=fig.colorbar(im,cax=cax,orientation='horizontal')
        cb.set_label('Avoided output exposure (million US$; log)' if kind=='absolute' else 'Avoided / no-Arctic output exposure (%)',fontsize=8)
    fig.legend(handles=[Patch(facecolor=ZERO,label='Computed zero'),Patch(facecolor='white',edgecolor='#a3aeb5',label='Undefined / unavailable')],
        loc='center',bbox_to_anchor=(.53,.010),ncol=2,frameon=False,fontsize=8)
    save(fig,'MainFig5')

    fig=plt.figure(figsize=(18,12.8)); nations=geography()
    country_loss=pd.read_csv(OUT/'country_endpoint_losses.csv'); country_loss=country_loss[country_loss.primary_case]
    lognorm=LogNorm(.001,1000); relnorm=Normalize(0,100)
    for j,ship in enumerate(SHIPS):
        p=country_loss[country_loss.shiptype.eq(ship)].pivot(index='iso3',columns='case',values='total_loss_usd')
        p['relative']=100*(p.without_arctic-p.with_arctic)/p.without_arctic.replace(0,np.nan)
        left=.055+j*.235
        fig.text(left+.106,.968,LABELS[ship]+' | '+MONTHS[int(monthly[monthly.shiptype.eq(ship)&monthly.primary_case].month.iloc[0])-1],ha='center',fontsize=10,weight='bold')
        for row in range(2):
            ax=fig.add_axes([left,.725-row*.255,.215,.19]); tag(ax,chr(97+row*4+j))
            frame=nations.copy(); frame['value']=frame.country_iso3.map(p.without_arctic/1e6 if row==0 else p.relative)
            frame.plot(ax=ax,color='#fafbfc',edgecolor='#d6dde1',linewidth=.2,hatch='///')
            zero=frame.value.eq(0)
            if zero.any(): frame[zero].plot(ax=ax,color=ZERO,edgecolor='white',lw=.2)
            positive=frame.value.gt(0)
            if positive.any(): frame[positive].plot(ax=ax,column='value',cmap='YlOrRd' if row==0 else 'YlGnBu',norm=lognorm if row==0 else relnorm,edgecolor='white',lw=.2)
            ax.set(xlim=(-17100000,17100000),ylim=(-6500000,8750000));ax.set_axis_off()
    for y,cmap,norm,label in [(.70,'YlOrRd',lognorm,'Transport-loss proxy without Arctic (US$ million; log)'),
                             (.445,'YlGnBu',relnorm,'Transport loss avoided / without-Arctic loss (%)')]:
        cax=fig.add_axes([.32,y,.38,.009]); cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal');cb.set_label(label,fontsize=8)
    fig.legend(handles=[Patch(facecolor=ZERO,label='Computed zero'),Patch(facecolor='#fafbfc',hatch='///',edgecolor='#ccd2d6',label='Not mapped or undefined')],
        loc='center',bbox_to_anchor=(.52,.396),ncol=2,frameon=False,fontsize=8)
    axes=[fig.add_axes([.08,.13,.39,.215]),fig.add_axes([.585,.13,.39,.215])]
    for ax,label in zip(axes,'ij'):tag(ax,label)
    for case,color,marker,offset in [('without_arctic',original.WITHOUT,'o',-.10),('with_arctic',original.WITH,'s',.10)]:
        p=primary[primary.case.eq(case)].pivot(index='region',columns='buffer_days',values='output_exposure_usd').reindex(order)/1e9
        axes[0].errorbar(np.arange(20)+offset,p[0],yerr=np.array([p[0]-p.min(1),p.max(1)-p[0]]),fmt=marker,color=color,ms=4,lw=.8,capsize=2,label='Without Arctic' if case=='without_arctic' else 'With Arctic')
    axes[0].set_yscale('symlog',linthresh=.001);axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel('Mean output exposure\n(US$ bn per class-specific test)');country_axis(axes[0],order)
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=8)
    first=100*country.first_hit_usd/country.output_exposure_usd
    axes[1].bar(range(20),first,color=original.FIRST,label='First-hit exposure')
    axes[1].bar(range(20),100-first,bottom=first,color=original.HIGHER,label='Higher-order propagation')
    axes[1].set_ylim(0,100);axes[1].set_ylabel('No-Arctic output exposure (%)');country_axis(axes[1],order)
    axes[1].legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=8)
    save(fig,'SI_EconomicGeography_and_NationalExposure')
    sensitivity_plots(data,monthly)
    relief=sources[sources.case.eq('avoided')]
    primary_transport=monthly[monthly.primary_case].groupby('case').total_loss_usd.sum()/4
    stats=dict(primary_mean_transport_without_usd=float(primary_transport.without_arctic),
        primary_mean_transport_with_usd=float(primary_transport.with_arctic),
        primary_transport_relief_pct=float(100*(1-primary_transport.with_arctic/primary_transport.without_arctic)),
        mean_output_avoided_usd=float(relief.output_exposure_usd.sum()/4),
        higher_order_avoided_pct=float(100*(1-relief.first_hit_usd.sum()/relief.output_exposure_usd.sum())),
        top20_countries=order, country_order_shared=True, original_old_economic_artwork_used=False,
        primary_aggregation='Equal-weight mean of four independent Figure4 vessel-month cases, not a joint hazard')
    (OUT/'figure5_key_results.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2),flush=True)


def sensitivity_plots(data,monthly):
    d=data[data.buffer_days.eq(0)].groupby(['case','shiptype','month']).output_exposure_usd.sum()
    fig,axes=plt.subplots(3,1,figsize=(12,8.5));fig.subplots_adjust(left=.10,right=.91,top=.96,bottom=.08,hspace=.46)
    wo=d.xs('without_arctic').unstack('month').reindex(SHIPS);wi=d.xs('with_arctic').unstack('month').reindex(SHIPS)
    for i,(ax,values) in enumerate(zip(axes,[wo/1e9,wi/1e9,100*(wo-wi)/wo.replace(0,np.nan)])):
        cm=plt.get_cmap('YlOrRd' if i<2 else 'YlGnBu').copy();cm.set_bad('#ffffff');cm.set_under(ZERO)
        kwargs=dict(norm=LogNorm(max(float(np.nanmin(values.to_numpy()[values.to_numpy()>0])),1e-6),max(wo.max().max(),wi.max().max())/1e9)) if i<2 else dict(vmin=1e-10,vmax=100)
        image=ax.imshow(np.ma.masked_invalid(values.to_numpy()),cmap=cm,aspect='auto',**kwargs)
        ax.set_yticks(range(4),[LABELS[s] for s in SHIPS]);ax.set_xticks(range(12),MONTHS);tag(ax,chr(97+i))
        for yy in range(4):
            for xx in range(12):
                v=values.iloc[yy,xx];ax.text(xx,yy,'--' if not np.isfinite(v) else ('%.2f'%v if i<2 else '%.1f'%v),ha='center',va='center',fontsize=8)
        cb=fig.colorbar(image,ax=ax,fraction=.023,pad=.02);cb.set_label('US$ bn / test' if i<2 else 'Avoided (%)',fontsize=8)
    save(fig,'SI_Monthly_OutputExposure')
    support=pd.read_csv(OUT/'commodity_bridge_coverage_summary.csv');support=support[support.primary_case].set_index('shiptype').reindex(SHIPS)
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(range(4),support.supported_pct,color=SHIP_COLORS)
    for i,value in enumerate(support.supported_pct):ax.text(i,value+2,'%.1f%%'%value,ha='center',fontsize=10)
    ax.set(xticks=range(4),xticklabels=[LABELS[s] for s in SHIPS],ylim=(0,105),ylabel='Trade value supported by commodity bridge (%)')
    fig.tight_layout();save(fig,'SI_CommodityMapping_Coverage')


if __name__=='__main__':
    main()
