"""Verify saved sparse LR allocations at the individual-drone level."""
import argparse
import json
from pathlib import Path
import numpy as np
from flpd_revision import Instance


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root)
    report={'checked':0,'failures':[],'tolerance':1e-6}
    for folder in (root/'results').iterdir():
        if not folder.is_dir():continue
        for fp in folder.glob('*.npz'):
            with np.load(fp) as z:
                if not all(k in z for k in ['drone_index','drone_fraction','q','n']):continue
                rr=json.loads(fp.with_suffix('.json').read_text())
                inst=Instance.load(folder/'instances'/(rr['instance_id']+'.json'))
                ix=z['drone_index'];v=z['drone_fraction'];K=int(inst.K[0]);I,J,T=inst.shape
                if T!=1:raise ValueError('Sparse LR allocations require a homogeneous fleet')
                assert ix.shape==(len(v),3)
                assert np.all(ix>=0) and np.all(ix[:,0]<I) and np.all(ix[:,1]<J) and np.all(ix[:,2]<K)
                q=np.zeros_like(z['q']);np.add.at(q[:,:,0],(ix[:,0],ix[:,1]),v*inst.w[ix[:,0]])
                payload=np.bincount(ix[:,2],weights=v*inst.w[ix[:,0]],minlength=K)
                energy=np.bincount(ix[:,2],weights=v*inst.w[ix[:,0]]*inst.energy[ix[:,0],ix[:,1],0],minlength=K)
                homes=[set()for _ in range(K)]
                for (_,j,k),flow in zip(ix,v):
                    if flow>1e-9:homes[k].add(int(j))
                field_use=np.zeros(J)
                for h in homes:
                    for j in h:field_use[j]+=1
                violations={'flow_consistency':float(np.max(abs(q-z['q']))),
                    'negative_fraction':float(max(0,-v.min())) if len(v)else 0.,
                    'payload':float(max(0,np.max(payload-inst.C[0]))),
                    'energy':float(max(0,np.max(energy-inst.B[0]))),
                    'multiple_home_fields':sum(len(h)>1 for h in homes),
                    'field_drone_count':float(max(0,np.max(field_use-z['n'][:,0])))}
                report['checked']+=1
                if max(violations.values())>report['tolerance']:
                    report['failures'].append({'file':str(fp.relative_to(root)),'violations':violations})
    report['passed']=not report['failures']
    (root/'results/drone_allocation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()
