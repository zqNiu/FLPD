"""Audit stored solution arrays against saved inputs and inspect certificate order."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from flpd_revision import Instance,audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root)
    report={'array_solutions_checked':0,'audit_failures':[],'input_hash_failures':[],'duality_failures':[],'files_by_folder':{}}
    for folder in sorted((root/'results').iterdir()):
        if not folder.is_dir():continue
        count=0
        for path in folder.glob('*.json'):
            r=json.loads(path.read_text())
            if not isinstance(r,dict) or 'task' not in r:continue
            count+=1;npz=path.with_suffix('.npz')
            ip=folder/'instances'/(r.get('instance_id','')+'.json')
            if not ip.exists() or not npz.exists():continue
            inst=Instance.load(ip)
            # Hash the saved numerical representation before dataclass float casts.
            raw_hash=hashlib.sha256(json.dumps(json.loads(ip.read_text()),sort_keys=True).encode()).hexdigest()
            if r.get('input_sha256') and raw_hash!=r['input_sha256']:report['input_hash_failures'].append(str(path))
            with np.load(npz) as z:
                if not all(k in z for k in ('q','y','n')):continue
                rep=audit(inst,z['q'],z['y'],z['n'])
            report['array_solutions_checked']+=1
            if not rep['feasible']:report['audit_failures'].append({'file':str(path),'violations':rep['max_violation']})
            ub,lb=r.get('upper_bound'),r.get('lower_bound')
            if ub is not None and abs(rep['objective']-ub)>1e-5*max(1,abs(ub)):
                report['audit_failures'].append({'file':str(path),'objective_mismatch':rep['objective']-ub})
            if ub is not None and lb is not None and lb>ub+1e-5*max(1,abs(ub)):report['duality_failures'].append(str(path))
        report['files_by_folder'][folder.name]=count
    report['passed']=not any(report[k] for k in ('audit_failures','input_hash_failures','duality_failures'))
    (root/'results/package_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));assert report['passed'],'See results/package_audit.json'
if __name__=='__main__':main()
