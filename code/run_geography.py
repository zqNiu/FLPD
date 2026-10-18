"""Evaluate supplied, source-documented customer and candidate coordinates.

customers.csv: id,latitude,longitude,demand
sites.csv: id,latitude,longitude,capacity,fixed_cost
Distances use a local equirectangular projection in km (district-scale only).
No customer demand or service legality is inferred from map coordinates.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from flpd_revision import Instance,solve,save_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--customers',required=True);p.add_argument('--sites',required=True)
    p.add_argument('--source',required=True,help='Actual dataset/source description and demand provenance')
    p.add_argument('--output',required=True);p.add_argument('--backend',choices=['highs','gurobi'],default='highs')
    p.add_argument('--battery',type=float,default=50);p.add_argument('--payload',type=float,default=5)
    p.add_argument('--fleet',type=int,required=True);p.add_argument('--field-limit',type=int,required=True)
    p.add_argument('--drone-cost',type=float,default=0);p.add_argument('--cost-per-km',type=float,default=5)
    p.add_argument('--alpha',type=float,default=1.43);p.add_argument('--seconds',type=float,default=300)
    a=p.parse_args()
    def read(path):
        with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    cust,sites=read(a.customers),read(a.sites)
    coords=np.array([[float(r['latitude']),float(r['longitude'])] for r in cust+sites])
    if np.any(abs(coords[:,0])>90) or np.any(abs(coords[:,1])>180):raise ValueError('Invalid latitude/longitude')
    lat0,lon0=coords.mean(axis=0)
    xy=np.column_stack([6371.0088*np.cos(np.deg2rad(lat0))*np.deg2rad(coords[:,1]-lon0),6371.0088*np.deg2rad(coords[:,0]-lat0)])
    x,y=xy[:len(cust)],xy[len(cust):]
    if np.max(np.ptp(xy,axis=0))>100:raise ValueError('Extent exceeds local-projection limit; supply a projected-distance workflow')
    inst=Instance(np.array([float(r['demand']) for r in cust]),x,y,np.linalg.norm(x[:,None,:]-y[None,:,:],axis=2),
        np.array([float(r['capacity']) for r in sites]),np.array([float(r['fixed_cost']) for r in sites]),
        np.array([a.payload]),np.array([a.battery]),np.array([a.alpha]),np.array([a.drone_cost]),np.array([a.fleet]),a.field_limit,
        c=a.cost_per_km,name='geographic_case',provenance=a.source)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);inst.save(out/'instance.json')
    save_result(solve(inst,backend=a.backend,time_limit=a.seconds),out/'result.json')
if __name__=='__main__':main()
