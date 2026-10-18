"""Exact symmetry quotient of divisible-demand FLPD; SciPy and Gurobi backends.

Amounts q[i,j,t] (not fractions) are carried by identical drones of type t.
Integer n[j,t] counts drones. This is EXACT, not a drone-budget relaxation:
divide q[:,j,t] equally among n[j,t] drones to reconstruct M1.
The label formulation is obtained by giving each drone its own type of size 1.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass
class Instance:
    w: np.ndarray
    xy: np.ndarray
    sites: np.ndarray
    distance: np.ndarray  # I,J
    capacity: np.ndarray  # J
    fixed: np.ndarray     # J
    C: np.ndarray         # T
    B: np.ndarray         # T
    alpha: np.ndarray     # T
    h: np.ndarray         # T
    K: np.ndarray         # T, integer
    P: int
    c: float = 5.0
    energy_price: float = 0.0
    name: str = "instance"
    provenance: str = "synthetic"

    def __post_init__(self):
        for key in ('w','xy','sites','distance','capacity','fixed','C','B','alpha','h','K'):
            setattr(self,key,np.asarray(getattr(self,key),dtype=float))
        I,J,T=len(self.w),len(self.fixed),len(self.C)
        if self.distance.shape!=(I,J) or self.xy.shape!=(I,2) or self.sites.shape!=(J,2):
            raise ValueError('Inconsistent geometry dimensions')
        if any(len(getattr(self,k))!=T for k in ('B','alpha','h','K')):
            raise ValueError('Inconsistent type dimensions')
        if not all(np.all(np.isfinite(getattr(self,k))) for k in ('w','distance','capacity','fixed','C','B','alpha','h','K')):
            raise ValueError('Nonfinite input')
        if any(np.any(getattr(self,k)<=0) for k in ('w','capacity','C','B','K')):
            raise ValueError('Demand, budgets, capacities and type availability must be positive')
        if any(np.any(getattr(self,k)<0) for k in ('distance','fixed','alpha','h')) or self.c<0 or self.energy_price<0:
            raise ValueError('Costs, distances and energy rates must be nonnegative')
        if np.any(self.K!=np.rint(self.K)) or not 1<=self.P<=J:
            raise ValueError('Invalid fleet counts or field limit')

    @property
    def shape(self): return len(self.w),len(self.fixed),len(self.C)
    @property
    def energy(self): return 2*self.distance[:,:,None]*self.alpha[None,None,:]
    def payload(self):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(self).items()}
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.payload(),sort_keys=True).encode()).hexdigest()
    def save(self,path): Path(path).write_text(json.dumps(self.payload(),indent=2),encoding='utf-8')
    @classmethod
    def load(cls,path): return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


def synthetic(size,layout,seed,site_design='grid'):
    # Reuse exactly the historical RNG and candidate grid; U_j = 15 median(w).
    from flpd_instances_v2 import generate_synthetic_geometry,_balanced_candidate_grid
    xy,w,sites=generate_synthetic_geometry(size,layout,seed=seed)
    rng=np.random.default_rng(seed+871001)
    if site_design=='random': sites=rng.uniform(0,10,(3*size,2))
    elif site_design=='sparse': sites=_balanced_candidate_grid(size,10.)
    elif site_design=='kmeans':
        # K-means cannot have 3I distinct nonempty customer clusters. Use ceil(I/3).
        from scipy.cluster.vq import kmeans2
        sites,_=kmeans2(xy,max(2,int(math.ceil(size/3))),minit='++',seed=int(seed))
    elif site_design!='grid': raise ValueError(site_design)
    return Instance(w,xy,sites,np.linalg.norm(xy[:,None,:]-sites[None,:,:],axis=2),
        np.full(len(sites),15*np.median(w)),np.full(len(sites),350.),
        np.array([5.]),np.array([50.]),np.array([1.43]),np.array([0.]),np.array([3*size]),
        min(size,len(sites)),name=f'I{size}_{layout}_seed{seed}_{site_design}')


def label_instance(a):
    """Same problem with each drone label treated as a singleton type."""
    d=a.payload()
    for k in ('C','B','alpha','h'): d[k]=np.repeat(getattr(a,k),a.K.astype(int)).tolist()
    d['K']=[1]*int(sum(a.K))
    d['name']=a.name+'_labels'
    return Instance(**d)


def matrices(a,mode='count',fixed_open=None):
    I,J,T=a.shape
    if mode=='facility': T=0
    Q=I*J*(T if T else 1); N=J*T; V=Q+J+N
    q=np.arange(Q).reshape((I,J,T) if T else (I,J))
    y=np.arange(Q,Q+J); n=np.arange(Q+J,V).reshape(J,T)
    eqrow=np.repeat(np.arange(I),J*(T if T else 1))
    Ae=sparse.coo_matrix((np.ones(Q),(eqrow,np.arange(Q))),shape=(I,V)).tocsr()
    rr=[]; cc=[]; vv=[]; rhs=[]; tags={}
    def add(name,row,col,val,b):
        offset=len(rhs); rows=np.asarray(row).ravel(); cols=np.asarray(col).ravel()
        vals=np.broadcast_to(val,rows.shape).ravel(); b=np.atleast_1d(b)
        rr.append(rows+offset);cc.append(cols);vv.append(vals);rhs.extend(b.tolist())
        tags[name]=slice(offset,offset+len(b))
    add('field_limit',np.zeros(J,int),y,1,[a.P])
    qr=np.broadcast_to(np.arange(J)[None,:,None],(I,J,T)).ravel() if T else np.tile(np.arange(J),I)
    add('capacity',np.r_[qr,np.arange(J)],np.r_[np.arange(Q),y],np.r_[np.ones(Q),-a.capacity],np.zeros(J))
    if T:
        pair=np.tile(np.arange(N),I)
        add('payload',np.r_[pair,np.arange(N)],np.r_[np.arange(Q),n.ravel()],
            np.r_[np.ones(Q),-np.tile(a.C,J)],np.zeros(N))
        add('energy',np.r_[pair,np.arange(N)],np.r_[np.arange(Q),n.ravel()],
            np.r_[a.energy.ravel(),-np.tile(a.B,J)],np.zeros(N))
        add('link',np.repeat(np.arange(N),2),np.stack([n.ravel(),np.repeat(y,T)],axis=1).ravel(),
            np.stack([np.ones(N),-np.tile(a.K,J)],axis=1).ravel(),np.zeros(N))
        add('fleet',np.tile(np.arange(T),J),n.ravel(),1,a.K)
    A=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(len(rhs),V)).tocsr()
    costs=(a.c*a.distance[:,:,None]+a.energy_price*a.energy).ravel() if T else (a.c*a.distance).ravel()
    objective=np.r_[costs,a.fixed,np.tile(a.h,J) if T else []]
    upper=np.r_[np.repeat(a.w,J*(T if T else 1)),np.ones(J),np.tile(a.K,J) if T else []]
    lower=np.zeros(V)
    if fixed_open is not None:
        lower[y]=0; upper[y]=0; lower[y[list(fixed_open)]]=1;upper[y[list(fixed_open)]]=1
    return objective,A,np.array(rhs),Ae,a.w,lower,upper,tags,(Q,y,n)


def audit(a,q,y,n,tol=1e-5):
    I,J,T=a.shape
    def pos(x): return float(max(0.,np.max(x,initial=0.)))
    load=q.sum(axis=0); energy=(a.energy*q).sum(axis=0)
    err={
        'demand_balance':float(np.max(np.abs(q.sum(axis=(1,2))-a.w),initial=0)),
        'nonnegative_flow':pos(-q),'field_capacity':pos(load.sum(axis=1)-a.capacity*y),
        'field_limit':pos(np.array([sum(y)-a.P])),
        'payload':pos(load-n*a.C),'energy':pos(energy-n*a.B),
        'fleet':pos(n.sum(axis=0)-a.K),'link':pos(n-y[:,None]*a.K),
        'count_nonnegative':pos(-n),'y_lower':pos(-y),'y_upper':pos(y-1),
        'y_integrality':float(np.max(abs(y-np.rint(y)),initial=0)),
        'n_integrality':float(np.max(abs(n-np.rint(n)),initial=0)),
    }
    # Independent reconstruction: each of n_jt drones receives q_ijt/n_jt.
    denom=np.where(n>.5,np.rint(n),1.)
    per_payload=load/denom; per_energy=energy/denom
    err['reconstructed_payload']=pos(per_payload-a.C)
    err['reconstructed_energy']=pos(per_energy-a.B)
    err['zero_count_flow']=pos(np.where(n<.5,load,0.))
    support=q.sum(axis=2)>tol
    obj=float(a.fixed@y+np.sum((a.c*a.distance[:,:,None]+a.energy_price*a.energy)*q)+np.sum(n*a.h))
    return {'feasible':max(err.values())<=tol,'max_violation':err,'objective':obj,
        'opened_fields':np.flatnonzero(y>.5).tolist(),'field_count':int(sum(y>.5)),
        'drones_used':int(np.rint(n).sum()),'split_field_customers':int(sum(support.sum(axis=1)>1)),
        'split_field_share':float(np.mean(support.sum(axis=1)>1)),
        'type_drone_counts':np.rint(n.sum(axis=0)).astype(int).tolist()}


def solve(a,mode='count',backend='highs',time_limit=30.,relax=False,fixed_open=None,threads=1,arrays=True):
    """time_limit includes matrix building. Solver overruns are measured, not hidden."""
    if mode=='facility' and (np.any(a.h) or a.energy_price):
        raise ValueError('Facility-only reference is restricted to the original zero drone/energy cost model')
    start=time.perf_counter();c,A,b,Ae,be,lo,hi,tags,idx=matrices(a,mode,fixed_open)
    built=time.perf_counter()-start; remaining=max(0.,time_limit-built)
    result={'formulation':mode,'backend':backend,'relax':relax,'status':'build_budget_exhausted',
        'build_seconds':built,'solver_seconds':0.,'wall_seconds':built,'upper_bound':None,'lower_bound':None,
        'input_sha256':a.fingerprint(),'variables':len(c),'constraint_rows':A.shape[0]+Ae.shape[0]}
    if remaining<=0: return result
    Q,y,n=idx
    if relax:
        r=linprog(c,A_ub=A,b_ub=b,A_eq=Ae,b_eq=be,bounds=np.column_stack([lo,hi]),method='highs',
            options={'time_limit':remaining})
        result['status']={0:'optimal',1:'limit',2:'infeasible',3:'unbounded'}.get(r.status,'error')
        if r.success:
            result['lp_value']=float(r.fun);result['lower_bound']=float(r.fun)
            if mode=='count':
                result['dual_multipliers']={k:np.maximum(0.,-r.ineqlin.marginals[tags[k]]).tolist()
                    for k in ('capacity','payload','energy','link')}
        v=r.x
    elif backend=='highs':
        integ=np.zeros(len(c),int);integ[Q:]=1
        ts=time.perf_counter()
        r=milp(c,integrality=integ,bounds=Bounds(lo,hi),constraints=[LinearConstraint(A,-np.inf,b),LinearConstraint(Ae,be,be)],
            options={'time_limit':remaining,'mip_rel_gap':1e-7})
        result['solver_seconds']=time.perf_counter()-ts
        result['status']={0:'optimal',1:'limit',2:'infeasible',3:'unbounded'}.get(r.status,'error')
        result['lower_bound']=finite(getattr(r,'mip_dual_bound',None));v=r.x
        result['solver_mip_gap']=finite(getattr(r,'mip_gap',None))
    elif backend=='gurobi':
        import gurobipy as gp
        model=gp.Model('FLPD_'+mode);model.Params.OutputFlag=0;model.Params.Threads=threads;model.Params.Seed=0
        model.Params.Symmetry=2;model.Params.MIPGap=1e-7
        vt=np.full(len(c),'C');vt[y]='B'
        if n.size: vt[n.ravel()]='I'
        x=model.addMVar(len(c),lb=lo,ub=hi,vtype=vt)
        model.setObjective(c@x);model.addConstr(A@x<=b);model.addConstr(Ae@x==be);model.update()
        remaining=time_limit-(time.perf_counter()-start)
        if remaining<=0:
            model.dispose();result['wall_seconds']=time.perf_counter()-start;return result
        model.Params.TimeLimit=remaining;model.optimize()
        result['solver_seconds']=float(model.Runtime)
        result['status']={2:'optimal',3:'infeasible',9:'limit',11:'interrupted'}.get(model.Status,str(model.Status))
        result['lower_bound']=finite(model.ObjBound) if model.Status not in (3,4,5) else None
        v=x.X if model.SolCount else None
        result['solver_mip_gap']=float(model.MIPGap) if model.SolCount else None
        model.dispose()
    else: raise ValueError(backend)
    if v is not None:
        I,J,T=a.shape
        if mode=='facility':
            result['facility_incumbent']=float(c@v);result['open_fields']=np.flatnonzero(v[y]>.5).tolist()
            if arrays: result['_flow']=v[:Q].reshape(I,J)
        elif not relax:
            q=v[:Q].reshape(I,J,T);yy=v[y];nn=v[n].copy()
            # Remove gratuitous unpriced idle assignments for fixed flows.
            # This minimizes counts only conditional on q, not over all networks.
            need=np.ceil(np.maximum(q.sum(axis=0)/a.C,(a.energy*q).sum(axis=0)/a.B)-1e-8)
            for t in range(T):
                if a.h[t]==0:nn[:,t]=np.minimum(np.rint(nn[:,t]),np.maximum(0.,need[:,t]))
            check=audit(a,q,yy,nn);result['audit']=check
            if check['feasible']:
                result['upper_bound']=check['objective']
                if arrays: result['_q']=q;result['_y']=yy;result['_n']=nn
            else: result['status']='audit_failed'
    result['wall_seconds']=time.perf_counter()-start
    if relax: result['solver_seconds']=result['wall_seconds']-built
    ub,lb=result['upper_bound'],result['lower_bound']
    if ub is not None and lb is not None:
        if lb>ub+1e-5*max(1.,abs(ub)): raise RuntimeError('Weak duality failed')
        result['gap_pct']=100*max(0.,ub-lb)/max(1e-12,abs(ub))
    return result


def finite(x): return float(x) if x is not None and np.isfinite(x) else None


def environment():
    import os,scipy,sys
    return {'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
        'cpu_count':os.cpu_count(),'numpy':np.__version__,'scipy':scipy.__version__,
        'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}


def save_result(result,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arraykeys=[k for k in result if k.startswith('_')]
    if arraykeys:
        np.savez_compressed(path.with_suffix('.npz'),**{k[1:]:result[k] for k in arraykeys})
    def safe(value):
        if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [safe(v) for v in value]
        if isinstance(value,np.ndarray): return safe(value.tolist())
        if isinstance(value,(float,np.floating)): return finite(value)
        if isinstance(value,np.integer): return int(value)
        return value
    clean=safe({k:v for k,v in result.items() if not k.startswith('_')})
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
    return clean
