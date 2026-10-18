"""Round-one experiments. No results are invented or filled from paper tables.

Run --suite validate first. JSON results and compressed solution arrays are saved
after each task, with complete input instances, code hash and environment data.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

# Set before NumPy/SciPy import. Users can explicitly override these values.
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_v,'1')
import numpy as np
from flpd_revision import Instance,synthetic,label_instance,solve,audit,save_result,finite,environment


def as_original(a):
    from flpd_core_v2 import ModelData
    if len(a.C)!=1 or np.any(a.h) or a.energy_price:
        raise ValueError('The historical LR is restricted to homogeneous, unpriced fleets')
    return ModelData(a.w,a.c*a.distance.T,a.energy[:,:,0].T,int(a.K[0]),a.P,
        a.fixed,a.capacity,a.B[0],a.C[0],distance_ji=a.distance.T)


def run_lr(a,args,variant='full',lp=None,extra=None):
    from flpd_core_v2 import solve_lagrangian_relaxation,LagrangianMultipliers,primal_heuristic,validate_solution
    start=time.perf_counter();d=as_original(a)
    # Same feasible initialization for controlled ablations, included in runtime.
    initial,initial_ub=primal_heuristic(d,None)
    kw=dict(max_iterations=args.iterations,time_limit=max(.001,args.lr_seconds-(time.perf_counter()-start)),verbose=False,
        initial_solution=initial)
    if variant=='no_guidance':kw['no_guidance']=True
    if variant=='no_local':kw['post_local_search']=False
    if variant=='final_only':kw['final_recovery_only']=True
    if variant=='lp_start':
        if lp is None or not lp.get('dual_multipliers'):raise ValueError('Optimal LP required for warm start')
        mu=LagrangianMultipliers.zeros(d);dm=lp['dual_multipliers']
        for name,key in [('a_jk','link'),('beta_jk','energy'),('delta_jk','payload')]:
            setattr(mu,name,np.broadcast_to(np.array(dm[key]).reshape(-1,1),(d.num_candidates,d.num_drones)).copy())
        mu.gamma_j=np.array(dm['capacity']);kw['initial_multipliers']=mu
    kw.update(extra or {})
    r=solve_lagrangian_relaxation(d,**kw)
    out={'method':'LR-H','variant':variant,'upper_bound':finite(r.best_upper_bound),
        'lower_bound':finite(r.best_lower_bound),'iterations':r.iterations,'termination':r.termination,
        'wall_seconds':time.perf_counter()-start,'history':r.history,'initial_upper_bound':finite(initial_ub)}
    if r.best_variables is not None:
        v=r.best_variables;rep=validate_solution(d,v,tolerance=1e-6)
        out['audit']={'feasible':rep.is_feasible,'max_violation':rep.max_violation,
            'field_count':int(sum(v.y_j>.5)),'drones_used':int(sum(v.z_jk.ravel()>.5)),
            'split_field_share':float(np.mean((v.x_ijk.sum(axis=2)>1e-7).sum(axis=1)>1))}
        out['_q']=v.x_ijk.sum(axis=2)[:,:,None]*a.w[:,None,None]
        out['_y']=v.y_j;out['_n']=v.z_jk.sum(axis=1)[:,None]
        # Save actual sparse drone allocations, preserving observed split counts.
        ii,jj,kk=np.where(v.x_ijk>1e-9)
        out['_drone_index']=np.column_stack([ii,jj,kk]);out['_drone_fraction']=v.x_ijk[ii,jj,kk]
    return out


def agg_pack(a,aggregate,args):
    from flpd_core_v2 import construct_feasible_from_open_fields,refine_open_fields
    start=time.perf_counter();d=as_original(a);opened=aggregate.get('open_fields')
    out={'method':'Agg-Pack','upper_bound':None,'lower_bound':aggregate.get('lower_bound')}
    if opened:
        built=construct_feasible_from_open_fields(d,opened)
        if built:
            _,v,ub=refine_open_fields(d,opened)
            if v is not None:
                out['upper_bound']=float(ub)
                out['_q']=v.x_ijk.sum(axis=2)[:,:,None]*a.w[:,None,None]
                out['_y']=v.y_j;out['_n']=v.z_jk.sum(axis=1)[:,None]
                out['audit']=audit(a,out['_q'],out['_y'],out['_n'])
    out['wall_seconds']=time.perf_counter()-start+aggregate['wall_seconds']
    out['status']='feasible' if out['upper_bound'] is not None else 'recovery_failed'
    return out


def codehash():
    h=hashlib.sha256()
    names=('flpd_core_v2.py','flpd_instances_v2.py','flpd_revision.py','run_revision.py')
    for name in names:
        p=Path(__file__).parent/name;h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def execute(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy();config.pop('resume',None);signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    meta={'configuration':config,'config_sha256':signature,'code_sha256':codehash(),'environment':environment()}
    path=out/'run_metadata.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['code_sha256']!=meta['code_sha256'] or old['config_sha256']!=signature:
            raise ValueError('Output configuration/code changed. Choose a new output folder.')
    path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    def task(a,name,fn):
        p=out/(a.name+'__'+name+'.json')
        if args.resume and p.exists():
            r=json.loads(p.read_text())
            if r.get('input_sha256')!=a.fingerprint():raise ValueError('Resume input mismatch')
            npz=p.with_suffix('.npz')
            if npz.exists():
                with np.load(npz) as z:r.update({'_'+k:z[k] for k in z.files})
            return r
        (out/'instances').mkdir(exist_ok=True);a.save(out/'instances'/(a.name+'.json'))
        try:r=fn()
        except (MemoryError,ImportError) as e:r={'status':'error','error':type(e).__name__+': '+str(e)}
        r['instance_id']=a.name;r['I'],r['J'],r['T']=a.shape;r['task']=name
        r['input_sha256']=a.fingerprint();r['code_sha256']=meta['code_sha256']
        save_result(r,p);print(a.name,name,r.get('status',r.get('termination')),r.get('upper_bound'),flush=True)
        return r
    if args.suite=='validate':
        checks=[]
        for seed in range(1,7):
            a=synthetic(4,'R',seed);a.K=np.array([4.]);a.C=np.array([4.]);a.B=np.array([20.])
            if seed%2==0:a.h=np.array([20.])
            if seed>4:
                a.C=np.array([3.,6.]);a.B=np.array([12.,35.]);a.alpha=np.array([1.43,1.2]);a.h=np.array([10.,25.]);a.K=np.array([2.,2.])
            b=label_instance(a)
            r=task(a,'count',lambda:solve(a,time_limit=args.mip_seconds,backend=args.backend))
            s=task(b,'label',lambda:solve(b,time_limit=args.mip_seconds,backend=args.backend))
            lp=task(a,'LP',lambda:solve(a,relax=True,time_limit=args.mip_seconds))
            lp2=task(b,'labelLP',lambda:solve(b,relax=True,time_limit=args.mip_seconds))
            if r['status']=='optimal' and s['status']=='optimal':
                assert abs(r['upper_bound']-s['upper_bound'])<1e-5
            else: assert r['status']==s['status']=='infeasible',(r['status'],s['status'])
            if lp['status']=='optimal': assert abs(lp['lp_value']-lp2['lp_value'])<1e-5
            checks.append({'seed':seed,'MIP_status':r['status'],'label_status':s['status'],'LP_status':lp['status']})
        # Directly verify lifted LP multipliers attain the full-model LP value.
        a=synthetic(6,'C',8);lp=solve(a,relax=True,time_limit=args.mip_seconds)
        r=run_lr(a,args,'lp_start',lp,{'max_iterations':1,'post_local_search':False})
        assert abs(r['lower_bound']-lp['lp_value'])<1e-5,(r['lower_bound'],lp['lp_value'])
        checks.append({'lifted_LP_duals_verified':True,'LP':lp['lp_value'],'LR':r['lower_bound']})
        (out/'validation_checks.json').write_text(json.dumps(checks,indent=2));return
    sizes=args.sizes if args.suite!='larger' else args.large_sizes
    for size in sizes:
      for layout in args.layouts:
       for seed in args.seeds:
        base=synthetic(size,layout,seed)
        if args.suite in ('bounds','benchmark','ablation','steps'):
            lp=task(base,'LP',lambda:solve(base,relax=True,time_limit=args.mip_seconds))
            if args.suite=='bounds':
                task(base,'count',lambda:solve(base,backend=args.backend,time_limit=args.mip_seconds));continue
            full=task(base,'LR_full',lambda:run_lr(base,args))
            agg=task(base,'facility',lambda:solve(base,mode='facility',backend=args.backend,time_limit=args.mip_seconds))
            if args.suite=='benchmark':
                budget=full['wall_seconds']+agg['wall_seconds']
                task(base,'count_equal',lambda:solve(base,backend=args.backend,time_limit=budget))
                if args.extended_seconds>0:task(base,'count_extended',lambda:solve(base,backend=args.backend,time_limit=args.extended_seconds))
                task(base,'no_guidance',lambda:run_lr(base,args,'no_guidance'))
                task(base,'AggPack',lambda:agg_pack(base,agg,args))
                if args.full_label:
                    nv=size*len(base.fixed)*int(sum(base.K))
                    if nv<=args.max_label_variables:
                        labelled=label_instance(base)
                        task(labelled,'label_equal',lambda:solve(labelled,backend=args.backend,time_limit=budget))
                        if args.extended_seconds>0:task(labelled,'label_extended',lambda:solve(labelled,backend=args.backend,time_limit=args.extended_seconds))
                    else:task(base,'label_equal',lambda:{'status':'skipped_memory_guard','allocation_variables':nv})
            elif args.suite=='ablation':
                for v in ('no_guidance','no_local','final_only'):
                    task(base,v,lambda v=v:run_lr(base,args,v))
                # A4 reuses the same full primal run; only the certificate is changed.
                task(base,'no_aggregate',lambda:{'upper_bound':full['upper_bound'],'lower_bound':full['lower_bound'],
                    'wall_seconds':full['wall_seconds'],'paired_source':'LR_full','status':'derived_certificate'})
            else:
                options=[('scale_'+str(s),'full',{'initial_step_scale':s}) for s in (.25,.5,1.,2.)]
                options += [('check_'+str(k),'full',{'checkpoint_every':k}) for k in (10,25,50)]
                options += [('factors_'+str(e)+'_'+str(c),'full',{'expansion_factor':e,'contraction_factor':c}) for e,c in ((1.1,.7),(1.3,.5),(1.5,.3))]
                options += [('LP_start','lp_start',{})]
                for name,v,kw in options:task(base,name,lambda v=v,kw=kw:run_lr(base,args,v,lp,kw))
        elif args.suite in ('battery','payload'):
            # Store direct LR, direct exact-model bound, and later a reusable envelope.
            values=args.batteries if args.suite=='battery' else args.payloads
            saved=[]
            for value in sorted(values):
                a=copy.deepcopy(base)
                if args.suite=='battery':a.B[:]=value
                else:a.C[:]=value
                a.name+='_'+args.suite+str(value)
                r=task(a,'direct_LR',lambda:run_lr(a,args))
                exact=task(a,'count',lambda:solve(a,backend=args.backend,time_limit=args.mip_seconds))
                saved.append((value,a,r,exact))
            for value,a,r,exact in saved:
                candidates=[]
                for source,b,rr,ee in saved:
                    for kind,run in [('direct_LR',rr),('count',ee)]:
                        if '_q' not in run:continue
                        check=audit(a,run['_q'],run['_y'],run['_n'])
                        if check['feasible']:candidates.append((check['objective'],source,kind,run))
                if candidates:
                    cost,source,kind,run=min(candidates,key=lambda x:x[0])
                    task(a,'reusable',lambda:{'status':'audited_reuse','upper_bound':cost,'source_budget':source,
                        'source_method':kind,'lower_bound':exact.get('lower_bound'),
                        '_q':run['_q'],'_y':run['_y'],'_n':run['_n'],'audit':audit(a,run['_q'],run['_y'],run['_n'])})
                # Separate LR-only reuse; do not silently mix exact incumbents with LR-H.
                lr_candidates=[x for x in candidates if x[2]=='direct_LR']
                if lr_candidates:
                    cost,source,kind,run=min(lr_candidates,key=lambda x:x[0])
                    task(a,'reusable_LR_only',lambda:{'status':'audited_reuse','upper_bound':cost,'source_budget':source,
                        'lower_bound':exact.get('lower_bound'),'_q':run['_q'],'_y':run['_y'],'_n':run['_n']})
        elif args.suite=='fleet':
            for B in args.batteries:
              for h in args.drone_costs:
               for Pextra in (0,2):
                a=copy.deepcopy(base);a.B[:]=B;a.h[:]=h
                a.K[:]=math.ceil(1.25*a.w.sum()/a.C[0]);nmin=math.ceil(a.w.sum()/a.capacity[0]);a.P=min(len(a.fixed),nmin+Pextra)
                a.name+=f'_B{B}_h{h}_P{a.P}_K{int(a.K[0])}'
                task(a,'count',lambda:solve(a,backend=args.backend,time_limit=args.mip_seconds))
            # Equal payload capacity at fleet level is not equal energy capability.
            a=copy.deepcopy(base);a.C=np.array([4.,7.]);a.B=np.array([20.,50.]);a.alpha=np.array([1.43,1.2]);a.h=np.array([10.,25.])
            K=math.ceil(1.25*a.w.sum()/5.);a.K=np.array([max(1,K//2),max(1,K-K//2)]);a.name+='_two_types'
            task(a,'count',lambda:solve(a,backend=args.backend,time_limit=args.mip_seconds))
        elif args.suite in ('larger','sites'):
            designs=('grid',) if args.suite=='larger' else ('grid','random','sparse','kmeans')
            for design in designs:
                a=synthetic(size,layout,seed,design)
                task(a,'count',lambda:solve(a,backend=args.backend,time_limit=args.mip_seconds))
                if args.suite=='larger' and args.include_large_lr:
                    task(a,'LR_full',lambda:run_lr(a,args))
        elif args.suite=='perturb':
            reference=task(base,'nominal_count',lambda:solve(base,backend=args.backend,time_limit=args.mip_seconds))
            if '_y' not in reference:continue
            fixed=np.flatnonzero(reference['_y']>.5).tolist();rng=np.random.default_rng(seed+90271)
            for k in range(args.scenarios):
                a=copy.deepcopy(base);a.w*=rng.uniform(.8,1.2,len(a.w));a.name+=f'_scenario{k:03d}'
                # Keep nominal field capacities fixed; changing U_j would confound perturbations.
                task(a,'fixed_fields',lambda:solve(a,backend=args.backend,time_limit=args.mip_seconds,fixed_open=fixed))
                task(a,'reoptimized',lambda:solve(a,backend=args.backend,time_limit=args.mip_seconds))
    print('Completed. Summarize with summarize_revision.py --results',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['validate','bounds','benchmark','ablation','steps','battery','payload','fleet','larger','sites','perturb'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--backend',choices=['highs','gurobi'],default='highs')
    p.add_argument('--sizes',nargs='+',type=int,default=[30,60,100]);p.add_argument('--large-sizes',nargs='+',type=int,default=[200,300])
    p.add_argument('--layouts',nargs='+',choices=['C','R','RC'],default=['C','R','RC'])
    p.add_argument('--seeds',nargs='+',type=int,default=[202601,202602,202603,202604,202605])
    p.add_argument('--iterations',type=int,default=1000);p.add_argument('--lr-seconds',type=float,default=300.)
    p.add_argument('--mip-seconds',type=float,default=300.);p.add_argument('--extended-seconds',type=float,default=0.)
    p.add_argument('--batteries',nargs='+',type=float,default=[5,10,20,50]);p.add_argument('--payloads',nargs='+',type=float,default=[3,4,5,6,8])
    p.add_argument('--drone-costs',nargs='+',type=float,default=[0,20])
    p.add_argument('--scenarios',type=int,default=100);p.add_argument('--resume',action='store_true')
    p.add_argument('--full-label',action='store_true');p.add_argument('--max-label-variables',type=int,default=10000000)
    p.add_argument('--include-large-lr',action='store_true')
    args=p.parse_args()
    if args.iterations<1 or min(args.lr_seconds,args.mip_seconds)<=0: p.error('Budgets must be positive')
    execute(args)
if __name__=='__main__':main()
