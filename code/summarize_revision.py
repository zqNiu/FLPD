"""Summaries from raw files only. Historical and new timing remain separate."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.stats import t,ttest_ind,ttest_rel


def stat(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    if not len(x):return {'n':0,'mean':None,'sd':None,'ci95':None}
    mean=float(x.mean());sd=float(x.std(ddof=1)) if len(x)>1 else None
    ci=[mean-t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)),mean+t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x))] if len(x)>1 else None
    return {'n':len(x),'mean':mean,'sd':sd,'ci95':ci,'max':float(x.max())}


def archive_rows(p):
    found=[]
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if n.endswith('.json') and 'benchmark_results' in n:
                d=json.loads(z.read(n));found.extend(d if isinstance(d,list) else d.get('rows',[]))
    return found


def holm(tests):
    order=sorted(range(len(tests)),key=lambda k:tests[k]['p_raw']);prev=0.
    for rank,k in enumerate(order):
        prev=max(prev,min(1.,tests[k]['p_raw']*(len(tests)-rank)));tests[k]['p_holm']=prev
    return tests


def historical(folder):
    root=Path(folder)
    base=archive_rows(root/'results_multiseed_final_v2_1.zip')
    exact={r['instance_id']:r.get('gurobi') for r in archive_rows(root/'results_multiseed_I30_exact_v2_1.zip')}
    for seed in (202602,202605):
        for r in archive_rows(root/f'results_rerun_I30_R_seed{seed}_v2_1.zip'):
            if r.get('gurobi'):exact[r['instance_id']]=r['gurobi']
    rows=[]
    for r in base:
        lr=r['lr'];agg=r['aggregate_relaxation'];ub=lr['best_upper_bound'];lb=max(lr['best_lower_bound'],agg['lower_bound'])
        e=exact.get(r['instance_id']) or {}
        eu=e.get('incumbent_objective');el=e.get('best_bound')
        combu=min([x for x in (ub,eu) if x is not None]);combl=max([x for x in (lb,el) if x is not None])
        rows.append({'instance_id':r['instance_id'],'I':r['I'],'layout':r['geometry'],'seed':r['seed'],
            'upper_bound':ub,'lr_lower_bound':lr['best_lower_bound'],'facility_lower_bound':agg['lower_bound'],
            'own_gap_pct':100*max(0.,ub-lb)/ub,'lr_seconds':lr['runtime_seconds'],
            'aggregate_seconds':agg['runtime_seconds'],'total_seconds':lr['runtime_seconds']+agg['runtime_seconds'],
            'field_count':r['solution']['field_count'],'drones_used':r['solution']['used_drones'],
            'gurobi_status':e.get('status'),'gurobi_UB':eu,'gurobi_LB':el,'gurobi_seconds':e.get('runtime_seconds'),
            'gurobi_own_gap_pct':100*max(0.,eu-el)/eu if eu and el is not None else None,
            'combined_gap_pct':100*max(0.,combu-combl)/combu,
            'lr_excess_over_gurobi_pct':100*(ub/eu-1) if eu else None})
    groups={}
    for size in (30,60,100):
        for layout in ('C','R','RC','ALL'):
            rr=[r for r in rows if r['I']==size and (layout=='ALL' or r['layout']==layout)]
            groups[f'I{size}_{layout}']={k:stat([r[k] for r in rr]) for k in
                ('upper_bound','own_gap_pct','lr_seconds','aggregate_seconds','total_seconds','lr_lower_bound','facility_lower_bound')}
    tests={}
    for metric in ('upper_bound','own_gap_pct'):
        ts=[]
        for size in (30,60,100):
            for x,y in (('C','R'),('C','RC'),('R','RC')):
                vx=[r[metric] for r in rows if r['I']==size and r['layout']==x]
                vy=[r[metric] for r in rows if r['I']==size and r['layout']==y]
                v=ttest_ind(vx,vy,equal_var=False)
                ts.append({'I':size,'pair':x+' vs '+y,'p_raw':float(v.pvalue)})
        tests[metric]=holm(ts)
    return {'source':'historical_user_runs','rows':rows,'groups':groups,'layout_tests_exploratory':tests,
        'notes':['Historical runtimes are not comparable to current-environment runtimes.',
            'Holm correction covers all nine layout contrasts per metric.',
            'Layouts share RNG seed labels; the Welch analysis is exploratory, not a causal layout effect.',
            'Source artifacts lack full x arrays; historical customer-to-drone splitting cannot be recovered.']}


def new_results(folder):
    rows=[]
    for p in Path(folder).rglob('*.json'):
        if p.parent.name=='instances' or p.name in ('summary.json','run_metadata.json'):continue
        r=json.loads(p.read_text())
        if isinstance(r,dict) and 'task' in r:
            rows.append({k:r.get(k) for k in ('instance_id','I','J','T','task','status','termination','upper_bound','lower_bound','lp_value','gap_pct','wall_seconds','solver_seconds')}
                | {'feasible':r.get('audit',{}).get('feasible'),'field_count':r.get('audit',{}).get('field_count'),
                   'drones_used':r.get('audit',{}).get('drones_used')})
    return {'source':'new_runs','rows':rows,'upper_bound':stat([r['upper_bound'] for r in rows]),
        'note':'Do not pool unlike tasks or budgets into a paper performance average.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy');p.add_argument('--results');p.add_argument('--output',required=True);a=p.parse_args()
    d=historical(a.legacy) if a.legacy else new_results(a.results)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')
    if d['rows']:
        with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(d['rows'][0]));w.writeheader();w.writerows(d['rows'])
    print(out/'summary.json')
if __name__=='__main__':main()
