#!/usr/bin/env python3
"""Convergence en resolution du reduit ExB cartesien (mesure paper-faithful T3).
Pour chaque n, fitte la fenetre papier MAPPEE (t_sim=2pi/rhobar*t_paper) et reporte
gamma_paper = gamma_raw*2pi vs la cible. Montre que l'erreur -> 0 avec n (l=3 : 13.7%->0.6%).
Lancer : PYTHONPATH=<adc_cpp>/build-master/python python diag/convergence_reduced.py"""
import math, sys
import numpy as np
import adc
R0,R1,RW=6.,8.,16.; RHO_MIN,RHO_MAX,DELTA=1e-6,1.,.1; TWO_PI=2*math.pi
PW={3:(.40,.70),4:(.60,.75),5:(1.15,1.35)}; PAP={3:.772,4:.911,5:.683}
def run(l,n,cfl=.4,t_end=None):
    t_end=t_end or TWO_PI*9.5
    s=adc.System(n=n,L=2*RW,periodic=False)
    s.set_poisson(rhs="charge_density",solver="geometric_mg",bc="dirichlet",wall="circle",wall_radius=RW)
    s.add_block("ne",model=adc.Model(state=adc.Scalar(),transport=adc.ExB(B0=1.),source=adc.NoSource(),elliptic=adc.ChargeDensity(charge=1.)),spatial=adc.Spatial(weno5=True),time=adc.Explicit(method="ssprk3"))
    h=2*RW/n; x=(np.arange(n)+.5)*h-RW; X,Y=np.meshgrid(x,x,indexing="xy"); r=np.hypot(X,Y); th=np.arctan2(Y,X)
    d=RHO_MAX*(1-DELTA+DELTA*np.sin(l*th)); s.set_density("ne",np.where((r>=R0)&(r<=R1),d,RHO_MIN).reshape(-1))
    thc=np.linspace(0,TWO_PI,1024,endpoint=False); xs=RW+R0*np.cos(thc); ys=RW+R0*np.sin(thc)
    fi=xs/h-.5; fj=ys/h-.5; i0=np.clip(np.floor(fi).astype(int),0,n-2); j0=np.clip(np.floor(fj).astype(int),0,n-2); tx,ty=fi-i0,fj-j0
    ts,cs=[],[]
    while True:
        t=float(s.time()); phi=np.asarray(s.potential(),float).reshape(n,n)
        if not np.isfinite(phi).all(): break
        v=(phi[j0,i0]*(1-tx)*(1-ty)+phi[j0,i0+1]*tx*(1-ty)+phi[j0+1,i0]*(1-tx)*ty+phi[j0+1,i0+1]*tx*ty)
        ts.append(t); cs.append(abs((np.fft.rfft(v)/v.size)[l]))
        if t>=t_end: break
        s.step_cfl(cfl)
    ts=np.array(ts); cs=np.array(cs); lo,hi=PW[l]; m=(ts>=lo*TWO_PI)&(ts<=hi*TWO_PI)&(cs>0)
    g=float(np.polyfit(ts[m],np.log(cs[m]),1)[0]) if m.sum()>4 else float("nan")
    return g*TWO_PI
print("reduit ExB -- convergence gamma_paper (x2pi, fenetre mappee) vs cible")
print(" n  | l=3            | l=4            | l=5")
for n in (64,96,128,192,256):
    row=[]
    for l in (3,4,5):
        gp=run(l,n); row.append("%.3f(%+.1f%%)"%(gp,100*(gp-PAP[l])/PAP[l]))
    print(" %3d| %s | %s | %s"%(n,row[0],row[1],row[2]))
print("cible          | %.3f          | %.3f          | %.3f"%(PAP[3],PAP[4],PAP[5]))
