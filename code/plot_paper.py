"""Generate Figures 2–10 from archived inputs and solutions without optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,FancyArrowPatch
from analyse_results import rows

COL=['#236192','#B05B28','#5A7653','#845B8F','#595959']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',default='figures');a=p.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    d=json.loads((root/'tables/analysis.json').read_text());bench=rows(root,'benchmark')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,n):
        fig.savefig(out/f'Figure_{n}.png',dpi=400,bbox_inches='tight')
        fig.savefig(out/f'Figure_{n}.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,s):
        ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fc='white',ec='#34495e',lw=.9))
        ax.text(x,y,s,ha='center',va='center',fontsize=8.3,linespacing=1.4)
    def arrow(a,b,**kw):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color='#34495e',lw=.85,**kw))
    box(.5,.92,.64,.1,'M1 inputs and resource constraints')
    box(.18,.70,.30,.19,'Lagrangian inner problem\nSP1 allocation\nSP2 field opening\nSP3 drone assignment')
    box(.18,.43,.30,.19,'Update multipliers\nRetain best LR bound\nPeriodic feasible recovery\nFinal local search')
    box(.52,.60,.26,.17,'Facility-only relaxation\nSolver best bound $L_F$\nAgg-Pack starting fields')
    box(.84,.70,.27,.17,'Exact count formulation\nInteger drone counts\nType-specific flows')
    box(.84,.43,.27,.19,'Equal-share\nreconstruction\nAudited $U_{count}$\nSolver bound $L_{count}$')
    box(.5,.12,.90,.14,'LR-H certificate: $U_{LR}$ and max($L_{LR}$, $L_F$)\nCount certificate: $U_{count}$ and $L_{count}$\nFeasible plans and optimality gaps')
    arrow((.30,.87),(.18,.795));arrow((.52,.87),(.52,.685));arrow((.75,.87),(.84,.785))
    arrow((.18,.605),(.18,.525));arrow((.84,.615),(.84,.525))
    arrow((.03,.43),(.03,.70),connectionstyle='arc3,rad=-.6');ax.text(-.005,.565,'iterate',rotation=90,ha='center',va='center',fontsize=7)
    arrow((.18,.335),(.25,.19));arrow((.52,.515),(.52,.19));arrow((.84,.335),(.75,.19))
    save(fig,2)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.65),layout='constrained')
    for ax,layout in zip(axes,['C','R','RC']):
        r=next(r for r in bench if r['task']=='count_equal' and r['instance_id']==f'I30_{layout}_seed202601_grid')
        raw=json.loads((root/'results/benchmark/instances'/(r['instance_id']+'.json')).read_text());xy=np.array(raw['xy']);sites=np.array(raw['sites'])
        z=np.load((root/r['_path']).with_suffix('.npz'));q=z['q'].sum(axis=2);opened=z['y']>.5
        ax.scatter(sites[:,0],sites[:,1],s=5,c='.8',marker='+')
        for i,j in zip(*np.where(q>1e-7)):
            ax.plot([xy[i,0],sites[j,0]],[xy[i,1],sites[j,1]],color=COL[0],lw=.5,alpha=.5,zorder=1)
        ax.scatter(xy[:,0],xy[:,1],s=13,c=COL[0],zorder=3)
        ax.scatter(sites[opened,0],sites[opened,1],s=55,c=COL[1],marker='s',edgecolor='white',lw=.5,zorder=4)
        ax.set(xlim=(-.4,10.4),ylim=(-.4,10.4),aspect='equal',xlabel='x (km)',title=layout,xticks=[0,5,10],yticks=[0,5,10])
    axes[0].set_ylabel('y (km)');save(fig,3)
    fig,ax=plt.subplots(figsize=(7.0,3.8),layout='constrained')
    groups=[(i,l) for i in [30,60,100] for l in ['C','R','RC']]
    mat=np.array([[next(r['own_gap_pct'] for r in d['primary_rows'] if r['I']==i and r['layout']==l and r['seed']==s) for s in range(202601,202606)]for i,l in groups])
    im=ax.imshow(mat,cmap='YlGnBu',aspect='auto',vmin=0,vmax=10)
    for i in range(9):
        for j in range(5):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>6 else 'black')
    ax.set(xticks=range(5),xticklabels=range(202601,202606),yticks=range(9),yticklabels=[f'{i} / {l}'for i,l in groups],xlabel='Random seed',ylabel='Customers / layout')
    fig.colorbar(im,ax=ax,label='LR-H certified gap (%)',pad=.02);save(fig,4)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
    tasks=['count_equal','AggPack','LR_full','no_guidance','label_equal'];names=['Count MILP','Agg-Pack','LR-H','Static-Pack','Label MILP']
    opts={r['instance_id']:r['upper_bound']for r in bench if r['task']=='count_equal'}
    for k,(task,name) in enumerate(zip(tasks,names)):
        rr=[r for r in bench if r['I']==30 and r['task']==task];ex=[100*(r['upper_bound']/opts[r['instance_id'].replace('_labels','')]-1)for r in rr]
        jitter=np.linspace(-.14,.14,len(ex));axes[0].scatter(ex,k+jitter,s=15,facecolors='white',edgecolors=COL[k],lw=.8)
        axes[0].plot(np.mean(ex),k,'D',color=COL[k],ms=5)
    axes[0].set(yticks=range(5),yticklabels=names,xlabel='Excess over optimum (%)');axes[0].invert_yaxis();axes[0].grid(axis='x',color='.9',lw=.5)
    times=[next(r['mean_wall_seconds']for r in d['comparison']if r['method']==t)for t in tasks]
    axes[1].barh(range(5),times,color=COL,height=.55)
    axes[1].set(yticks=range(5),yticklabels=[],xlabel='Mean total wall time (s)');axes[1].invert_yaxis()
    for k,v in enumerate(times):axes[1].text(v+.1,k,f'{v:.2f}',va='center',fontsize=8)
    axes[1].set_xlim(0,max(times)*1.23);save(fig,5)
    ab=rows(root,'ablation_small');fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    tasks=['LR_full','no_guidance','no_local','final_only','no_aggregate'];names=['Full','No LR guidance','No final local search','Final recovery only','No facility bound']
    for k,task in enumerate(tasks):
        rr=[r for r in ab if r['task']==task];ex=[]
        for r in rr:
            ref=next(s['upper_bound']for s in ab if s['instance_id']==r['instance_id']and s['task']=='LR_full');ex.append(100*(r['upper_bound']/ref-1))
        axes[0].scatter(ex,k+np.linspace(-.13,.13,9),s=15,edgecolors=COL[k],facecolors='white',lw=.8)
        axes[0].plot(np.mean(ex),k,'D',color=COL[k],ms=5)
    axes[0].axvline(0,c='.6',lw=.7);axes[0].set(yticks=range(5),yticklabels=names,xlabel='Paired objective change from full (%)');axes[0].invert_yaxis()
    cert=[r['mean_certificate_pct']for r in d['ablation']]
    axes[1].barh(range(5),cert,color=COL,height=.55);axes[1].set(yticks=range(5),yticklabels=[],xlabel='Mean certified gap (%)');axes[1].invert_yaxis()
    for k,v in enumerate(cert):axes[1].text(v+.6,k,f'{v:.2f}',va='center',fontsize=8)
    axes[1].set_xlim(0,73);save(fig,6)
    st=rows(root,'steps_small');fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained')
    for ax,l in zip(axes,['C','R','RC']):
        lp=next(r['lp_value']for r in st if r['task']=='LP'and r['instance_id'].split('_')[1]==l)
        for task,col,ls,label in zip(['scale_0.25','scale_1.0','scale_2.0','LP_start'],COL,[':','-','--','-.'],['Scale 0.25','Scale 1','Scale 2','LP initialization']):
            r=next(r for r in st if r['task']==task and r['instance_id'].split('_')[1]==l);h=r['history']
            ax.plot([v['iteration']for v in h],[100*v['best_lower_bound']/lp for v in h],c=col,ls=ls,lw=1.2,label=label)
        ax.set(title=l,xlabel='Iteration',ylim=(-2,105),xticks=[0,100,200]);ax.grid(axis='y',color='.9',lw=.5)
    axes[0].set_ylabel('Best LR bound / LP optimum (%)');axes[1].legend(loc='lower right',frameon=False,fontsize=6.5);save(fig,7)
    bat=json.loads((root/'data/legacy/battery_sensitivity_cumulative_payload_roundtrip_v2.json').read_text())['sweeps']
    checks=rows(root,'battery_exact_check');fig,axes=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
    for ax,(group,rr) in zip(axes.ravel(),bat.items()):
        rr=sorted(rr,key=lambda r:r['battery_budget']);seed=rr[0]['seed'];xs=[r['battery_budget']for r in rr]
        lower=[];upper=[]
        for x in xs:
            cc=[r for r in d['battery_reference'] if r.get('seed')==seed and r.get('budget')==x]
            lower.append(max(r['lower_bound']for r in cc if r.get('lower_bound')is not None))
            upper.append(min(r['upper_bound']for r in cc if r.get('upper_bound')is not None))
        ax.plot(xs,[r['solution']['objective']for r in rr],'s--',c=COL[1],mfc='white',ms=4,lw=1,label='Direct LR')
        ax.plot(xs,upper,'o-',c=COL[0],mfc='white',ms=4,lw=1,label='Audited feasible envelope')
        ax.fill_between(xs,lower,upper,color=COL[0],alpha=.17,label='Certificate interval')
        ax.set(title=group.replace('_','-'),xlabel='Battery budget',ylabel='Objective',xticks=[5,10,20,30,50]);ax.grid(axis='y',color='.9',lw=.5)
    axes[0,0].legend(frameon=False,fontsize=7);save(fig,8)
    payload=json.loads((root/'data/legacy/payload_capacity_sensitivity_cumulative_payload_roundtrip_v2_2.json').read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    for l,col,marker in zip(['C','R'],COL,['o','s']):
        rr=[r for r in payload if r['geometry']==l];base={r['seed']:r['reusable_upper_bound']for r in rr if r['payload_capacity_Ck']==5};xs=sorted({r['payload_capacity_Ck']for r in rr})
        means=[];sd=[];counts=[];csd=[]
        for x in xs:
            ss=[r for r in rr if r['payload_capacity_Ck']==x];vals=[100*(r['reusable_upper_bound']/base[r['seed']]-1)for r in ss]
            means.append(np.mean(vals));sd.append(np.std(vals,ddof=1));cc=[r['used_drones'] if 'used_drones'in r else r['solution']['used_drones'] for r in ss]
            counts.append(np.mean(cc));csd.append(np.std(cc,ddof=1))
        axes[0].errorbar(xs,means,yerr=sd,c=col,marker=marker,mfc='white',ms=4,capsize=2,lw=1,label='LR envelope '+l)
        axes[1].errorbar(xs,counts,yerr=csd,c=col,marker=marker,mfc='white',ms=4,capsize=2,lw=1,label=l)
    axes[0].axhline(0,c='.25',ls='--',lw=1,label='Exact optimum change');axes[0].set_ylabel('Cost change from capacity 5 (%)');axes[1].set_ylabel('Recovered drones')
    for ax in axes:ax.set(xlabel='Cumulative payload capacity',xticks=[3,4,5,6,8]);ax.grid(axis='y',color='.9',lw=.5);ax.legend(frameon=False,fontsize=7)
    save(fig,9)
    site=rows(root,'sites_small');fig,axes=plt.subplots(1,3,figsize=(7.2,2.9),layout='constrained')
    designs=['grid','random','sparse','kmeans'];names=['Grid\n90','Random\n90','Grid\n30','Centres\n10']
    for ax,l in zip(axes,['C','R','RC']):
        for k,seed in enumerate([202601,202602,202603]):
            rr=[r for r in site if r['instance_id'].startswith(f'I30_{l}_seed{seed}_')];base=next(r['upper_bound']for r in rr if r['instance_id'].endswith('_grid'))
            val=[100*(next(r['upper_bound']for r in rr if r['instance_id'].endswith('_'+design))/base-1)for design in designs]
            ax.plot(range(4),val,marker=['o','s','^'][k],c=COL[k],ms=4,lw=.8,mfc='white',label=str(seed))
        ax.axhline(0,c='.6',lw=.7);ax.set(title=l,xticks=range(4),xticklabels=names,xlabel='Candidate design and count');ax.tick_params(axis='x',labelsize=7);ax.grid(axis='y',color='.9',lw=.5)
    axes[0].set_ylabel('Optimal cost change from grid (%)');axes[1].legend(frameon=False,fontsize=6.5);save(fig,10)
    print('Figures 2–10 saved to',out)


if __name__=='__main__':main()
