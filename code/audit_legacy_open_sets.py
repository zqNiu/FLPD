"""Rebuild and independently audit the open sets reported in legacy sweeps.

This does not assume that the historical primary-customer plot encodes all flows.
It solves the field transportation LP anew and constructs an equal-share drone
allocation. Failure of this particular reconstruction is not infeasibility proof.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from flpd_revision import synthetic,audit,save_result
from run_revision import as_original,codehash
from flpd_core_v2 import _facility_transport_lp


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.legacy);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    bat=json.loads((root/'battery_sensitivity_cumulative_payload_roundtrip_v2.json').read_text())
    payload=json.loads((root/'payload_capacity_sensitivity_cumulative_payload_roundtrip_v2_2.json').read_text())
    rows=[('battery',r) for rr in bat['sweeps'].values() for r in rr]+[('payload',r) for r in payload['rows']]
    for kind,row in rows:
        b=synthetic(row['I'],row['geometry'],row['seed'])
        value=row['battery_budget'] if kind=='battery' else row['payload_capacity_Ck']
        if kind=='battery':b.B[:]=value
        else:b.C[:]=value
        b.name+=f'_{kind}{value}'
        opened=row['solution']['opened_fields'];field=_facility_transport_lp(as_original(b),opened)
        r={'instance_id':b.name,'task':'legacy_open_reconstruction','budget_type':kind,'budget':value,
            'input_sha256':b.fingerprint(),'code_sha256':codehash(),'status':'reconstruction_failed',
            'historical_aggregate_bound':row['aggregate_relaxation']['lower_bound']}
        if field:
            q=field[0][:,:,None];y=np.zeros(len(b.fixed));y[opened]=1
            n=np.maximum(0,np.ceil(np.maximum(q.sum(axis=0)/b.C,(b.energy*q).sum(axis=0)/b.B)-1e-8))
            rep=audit(b,q,y,n);r['audit']=rep
            if rep['feasible']:r.update(status='audited_feasible',upper_bound=rep['objective'],_q=q,_y=y,_n=n)
        (out/'instances').mkdir(exist_ok=True);b.save(out/'instances'/(b.name+'.json'))
        save_result(r,out/(b.name+'.json'))
    print('Reconstructed',len(rows),'reported open sets')
if __name__=='__main__':main()
