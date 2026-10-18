"""Solve the EXACT count model for each previously uploaded sensitivity row.

Direct LR objectives and reusable LR envelopes remain separate. The historical
JSON files are evidence supplied by the author, not outputs regenerated here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from flpd_revision import synthetic,solve,save_result
from run_revision import codehash


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backend',choices=['highs','gurobi'],default='highs');p.add_argument('--seconds',type=float,default=10)
    p.add_argument('--which',choices=['battery','payload','all'],default='all');p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(a.legacy);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    inputs=[]
    if a.which in ('battery','all'):
        p=root/'battery_sensitivity_cumulative_payload_roundtrip_v2.json';d=json.loads(p.read_text())
        for group,rr in d['sweeps'].items():
            for row in rr:inputs.append(('battery',p,row))
    if a.which in ('payload','all'):
        p=root/'payload_capacity_sensitivity_cumulative_payload_roundtrip_v2_2.json';d=json.loads(p.read_text())
        for row in d['rows']:inputs.append(('payload',p,row))
    for kind,source,row in inputs:
        size=row['I'];layout=row.get('geometry');seed=row['seed'];b=synthetic(size,layout,seed)
        value=row['battery_budget'] if kind=='battery' else row['payload_capacity_Ck']
        if kind=='battery':b.B[:]=value
        else:b.C[:]=value
        b.name+=f'_{kind}{value}'
        target=out/(b.name+'.json')
        if a.resume and target.exists():continue
        (out/'instances').mkdir(exist_ok=True);b.save(out/'instances'/(b.name+'.json'))
        r=solve(b,backend=a.backend,time_limit=a.seconds)
        r.update({'instance_id':b.name,'I':size,'layout':layout,'seed':seed,'budget_type':kind,'budget':value,
            'task':'exact_count_check','code_sha256':codehash(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_file':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'historical_direct_LR_UB':row.get('direct_lr_upper_bound',row.get('solution',{}).get('objective')),
            'historical_reusable_LR_UB':row.get('reusable_upper_bound',row.get('heuristic_upper_bound'))})
        save_result(r,target);print(b.name,r['status'],r.get('upper_bound'),flush=True)
if __name__=='__main__':main()
