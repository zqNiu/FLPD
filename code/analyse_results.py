"""Recompute manuscript tables from saved experimental records."""
from pathlib import Path
import argparse
import csv
import json
import re
import numpy as np
from scipy.stats import t
from flpd_revision import Instance,audit


def rows(root, folder):
    result=[]
    for p in sorted((root/'results'/folder).glob('*.json')):
        r=json.loads(p.read_text(encoding='utf-8'))
        if isinstance(r,dict) and 'task' in r:
            r['_path']=str(p.relative_to(root));result.append(r)
    return result


def stats(v):
    x=np.asarray([a for a in v if a is not None],float)
    if not len(x):return {'n':0,'mean':None,'sd':None,'max':None,'ci95':None}
    sd=float(x.std(ddof=1)) if len(x)>1 else None
    ci=None if sd is None else (x.mean()+np.array([-1,1])*t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x))).tolist()
    return {'n':len(x),'mean':float(x.mean()),'sd':sd,'max':float(x.max()),'ci95':ci}


def export(root,name,data):
    if not data:return
    with (root/'tables'/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');args=ap.parse_args()
    root=Path(args.root);(root/'tables').mkdir(exist_ok=True)
    evidence=json.loads((root/'results/evidence/paper_evidence.json').read_text())
    bench=rows(root,'benchmark');small=[r for r in bench if r['I']==30]
    opt={r['instance_id']:r['upper_bound'] for r in small if r['task']=='count_equal'}
    assert len(opt)==15
    comparison=[]
    for task in ['LR_full','no_guidance','AggPack','count_equal','label_equal','label_extended']:
        rr=[r for r in small if r['task']==task]
        ex=[100*(r['upper_bound']/opt[r['instance_id'].replace('_labels','')]-1) for r in rr if r.get('upper_bound') is not None]
        runtime=[r['wall_seconds']+(next(s['wall_seconds'] for s in small if s['task']=='facility' and s['instance_id']==r['instance_id']) if task=='LR_full' else 0) for r in rr]
        comparison.append({'method':task,'n':len(rr),'mean_excess_pct':float(np.mean(ex)),'max_excess_pct':max(ex),
            'optimal_objectives':sum(abs(x)<1e-5 for x in ex),'solver_optimal':sum(r.get('status')=='optimal' for r in rr),
            'mean_wall_seconds':float(np.mean(runtime)),'max_wall_seconds':max(runtime)})
    count=[r for r in small if r['task']=='count_equal']+[r for r in rows(root,'bounds_short') if r['task']=='count']
    count_summary=[]
    for size in [30,60,100]:
        rr=[r for r in count if r['I']==size]
        for layout in ['C','R','RC','ALL']:
            ss=[r for r in rr if layout=='ALL' or r['instance_id'].split('_')[1]==layout]
            count_summary.append({'I':size,'layout':layout,'n':len(ss),'solver_optimal':sum(r['status']=='optimal' for r in ss),
                'mean_gap_pct':float(np.mean([r['gap_pct'] for r in ss])), 'max_gap_pct':max(r['gap_pct'] for r in ss),
                'mean_wall_seconds':float(np.mean([r['wall_seconds'] for r in ss]))})
    ablation=[];ab=rows(root,'ablation_small')
    for task in ['LR_full','no_guidance','no_local','final_only','no_aggregate']:
        rr=[r for r in ab if r['task']==task];ex=[];gaps=[];paired=[]
        for r in rr:
            ref=next(s for s in ab if s['instance_id']==r['instance_id'] and s['task']=='LR_full')
            fac=next(s for s in ab if s['instance_id']==r['instance_id'] and s['task']=='facility')
            ub=r['upper_bound'];lb=r.get('lower_bound') or 0
            if task!='no_aggregate':lb=max(lb,fac['lower_bound'])
            ex.append(100*(ub/opt[r['instance_id']]-1));gaps.append(100*max(0,ub-lb)/ub)
            paired.append(100*(ub/ref['upper_bound']-1))
        ablation.append({'method':task,'n':len(rr),'mean_excess_pct':float(np.mean(ex)),'max_excess_pct':max(ex),
            'mean_certificate_pct':float(np.mean(gaps)),'mean_paired_change_pct':float(np.mean(paired)),
            'worse_than_full':sum(v>1e-5 for v in paired),'better_than_full':sum(v < -1e-5 for v in paired),
            'mean_wall_seconds':float(np.mean([r['wall_seconds'] for r in rr]))})
    steps=[];st=rows(root,'steps_small')
    for task in ['LR_full','scale_0.25','scale_0.5','scale_1.0','scale_2.0','check_10','check_25','check_50',
                 'factors_1.1_0.7','factors_1.3_0.5','factors_1.5_0.3','LP_start']:
        rr=[r for r in st if r['task']==task]
        lp=[next(s['lp_value'] for s in st if s['instance_id']==r['instance_id'] and s['task']=='LP') for r in rr]
        steps.append({'setting':task,'n':len(rr),'mean_excess_pct':float(np.mean([100*(r['upper_bound']/opt[r['instance_id']]-1) for r in rr])),
            'mean_LR_LP_pct':float(np.mean([100*r['lower_bound']/v for r,v in zip(rr,lp)])),
            'mean_wall_seconds':float(np.mean([r['wall_seconds'] for r in rr]))})
    site=rows(root,'sites_small');sites=[]
    for layout in ['C','R','RC']:
        for design in ['grid','random','sparse','kmeans']:
            rr=[r for r in site if r['instance_id'].split('_')[1]==layout and r['instance_id'].endswith('_'+design)]
            relative=[]
            for r in rr:
                key=r['instance_id'].rsplit('_',1)[0]+'_grid'
                baseline=next(s['upper_bound'] for s in site if s['instance_id']==key)
                relative.append(100*(r['upper_bound']/baseline-1))
            vals=stats([r['upper_bound'] for r in rr])
            sites.append({'layout':layout,'design':design,'n':len(rr),'sites':rr[0]['J'],'mean_objective':vals['mean'],
                'sd_objective':vals['sd'],'mean_change_from_grid_pct':float(np.mean(relative)),
                'mean_fields':float(np.mean([r['audit']['field_count'] for r in rr]))})
    splitting=[]
    for r in [s for s in small if s['task']=='LR_full']:
        p=root/r['_path'];z=np.load(p.with_suffix('.npz'));ix=z['drone_index'];v=z['drone_fraction'];n=r['I']
        counts=np.bincount(ix[v>1e-7,0],minlength=n)
        splitting.append({'instance':r['instance_id'],'field_split_share':r['audit']['split_field_share'],
            'drone_split_share':float(np.mean(counts>1)),'max_drone_pairs_per_customer':int(max(counts))})
    battery=[];checks=rows(root,'battery_exact_check');reconstructed=rows(root,'reconstructed_legacy_sweeps')
    old_battery=json.loads((root/'data/legacy/battery_sensitivity_cumulative_payload_roundtrip_v2.json').read_text())['sweeps']
    for seed in [1301,1302,1601,1602]:
        cr=[r for r in checks if r.get('seed')==seed]
        old=next(rr for rr in old_battery.values() if rr[0]['seed']==seed)
        recon=[r for r in reconstructed if f'_seed{seed}_'in r['instance_id'] and r['budget_type']=='battery']
        for budget in sorted({r['budget']for r in cr}):
            target=next(r for r in cr if r['budget']==budget)
            inst=Instance.load(root/'results/battery_exact_check/instances'/(target['instance_id']+'.json'))
            lb=max([r['lower_bound']for r in cr if r['budget']>=budget and r.get('lower_bound')is not None]
                +[r['aggregate_relaxation']['lower_bound']for r in old])
            feasible=[]
            for r in cr+recon:
                fp=(root/r['_path']).with_suffix('.npz')
                if not fp.exists():continue
                with np.load(fp) as z:rep=audit(inst,z['q'],z['y'],z['n'])
                if rep['feasible']:feasible.append((rep['objective'],r,rep))
            ub,source,rep=min(feasible,key=lambda x:x[0])
            assert lb<=ub+1e-5
            direct=next(r['solution']['objective']for r in old if r['battery_budget']==budget)
            lr_reuse=min(r['solution']['objective']for r in old if r['solution']['max_battery_load']<=budget+1e-6)
            battery.append({'I':target['I'],'layout':target['layout'],'seed':seed,'budget':budget,'lower_bound':lb,
                'upper_bound':ub,'gap_pct':100*max(0,ub-lb)/ub,'source_budget':source['budget'],
                'source_file':source['_path'],'source_method':source['task'],'fields':rep['field_count'],'drones':rep['drones_used'],
                'direct_LR_upper_bound':direct,'reusable_LR_upper_bound':lr_reuse})
    fleet=rows(root,'local_fleet');perturb=rows(root,'local_perturb')
    output={'comparison':comparison,'count_summary':count_summary,'ablation':ablation,'steps':steps,'sites':sites,
        'splitting':splitting,'battery_reference':battery,'fleet':[ {k:r.get(k) for k in ['instance_id','status','upper_bound','lower_bound','gap_pct']}|{'audit':r.get('audit')} for r in fleet],
        'perturbation':[ {k:r.get(k) for k in ['instance_id','task','status','upper_bound','lower_bound']} for r in perturb],
        'primary_rows':evidence['primary_rows'],'primary_groups':evidence['primary_groups'],
        'environment':json.loads((root/'results/benchmark/run_metadata.json').read_text())['environment']}
    (root/'tables/analysis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for name in ['comparison','count_summary','ablation','steps','sites','splitting','battery_reference']:export(root,name,output[name])
    print(json.dumps({k:output[k] for k in ['comparison','ablation','steps','sites']},indent=2))


if __name__=='__main__':main()
