"""Validated classical benchmark rerun, separate from quarantined Bayesian outputs.

Run from project root: python -m src.validated_baselines
Only M0--M2 are accepted here. M3--M5 remain under scientific validation.
"""
from pathlib import Path
from datetime import date, datetime, timezone
import csv, json, hashlib, time
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.stats import norm
from src.models import ns_loadings, fit_lambda, project_covariance
from src.experiment import read_panel, select_origins, write_csv
from src.scores import dss, energy_score, variogram_score, hac_dm

ROOT=Path(__file__).resolve().parents[1]
TAU=np.array([.25,.5,1,2,5,7,10,30])
MODELS=['M0','M1','M2']


def gaussian_moments(mean,cov):
    return np.asarray(mean),project_covariance(cov,1e-6)


def forecast_baseline(y,model,h,load):
    if model=='M0':
        return gaussian_moments(y[-1].copy(),np.cov(y[h:]-y[:-h],rowvar=False,ddof=1))
    if model=='M1':
        center=y.mean(axis=0)
        _,_,vt=np.linalg.svd(y-center,full_matrices=False)
        basis=vt[:3].T; f=(y-center)@basis
        X=np.column_stack([np.ones(len(f)-1),f[:-1]])
        beta=np.linalg.lstsq(X,f[1:],rcond=None)[0]
        intercept=beta[0]; A=beta[1:].T
        residual=f[1:]-X@beta
        q=np.cov(residual,rowvar=False,ddof=1)
        m=f[-1].copy(); c=np.zeros((3,3))
        for _ in range(h): m=intercept+A@m; c=A@c@A.T+q
        r=np.maximum(np.mean(((y-center)-f@basis.T)**2,axis=0),1e-6)
        return gaussian_moments(center+basis@m,basis@c@basis.T+np.diag(r))
    if model=='M2':
        f=np.linalg.lstsq(load,y.T,rcond=None)[0].T
        r=np.maximum(np.mean((y-f@load.T)**2,axis=0),1e-6)
        mf=[]; vf=[]
        for j in range(3):
            X=np.column_stack([np.ones(len(f)-1),f[:-1,j]])
            a,b=np.linalg.lstsq(X,f[1:,j],rcond=None)[0]
            # Classical AR(1) is unrestricted OLS, not silently clipped.
            q=np.mean((f[1:,j]-X@np.array([a,b]))**2)
            mf.append(b**h*f[-1,j]+a*sum(b**k for k in range(h)))
            vf.append(q*sum(b**(2*k) for k in range(h)))
        return gaussian_moments(load@mf,load@np.diag(vf)@load.T+np.diag(r))
    raise ValueError(model)


def run():
    start=time.perf_counter();out=ROOT/'results'/'validated_baselines';out.mkdir(exist_ok=True)
    config=json.loads((ROOT/'config/experiment.json').read_text())
    path=ROOT/'data/processed/canada_weekly.csv'
    dates,y=read_panel(path)
    first=next(i for i,d in enumerate(dates) if d>=date(2000,1,1))
    initial=[i for i,d in enumerate(dates) if date(2000,1,1)<=d<=date(2015,12,31)]
    lam,_,sse=fit_lambda(y[initial],TAU,np.linspace(*config['lambda_grid_per_year']))
    load=ns_loadings(TAU,lam)
    selected=select_origins(dates,date(2016,1,1),date(2025,12,31),[1,4],4)
    rng=np.random.default_rng(20261027)
    rows=[];maturity_rows=[];risk=[];audit=[];draw_store=[];mean_store=[];cov_store=[];target_store=[];key_store=[]
    for t,h,target in selected:
        train=y[first:t+1]
        assert dates[t]<dates[target] and dates[target]<=date(2025,12,31)
        audit.append({'origin_date':dates[t].isoformat(),'target_date':dates[target].isoformat(),'horizon':h,'n_training':len(train),'latest_fit_date':dates[t].isoformat(),'elapsed_calendar_days':(dates[target]-dates[t]).days})
        for model in MODELS:
            m,c=forecast_baseline(train,model,h,load)
            draws=rng.multivariate_normal(m,c,size=1024)
            observed=y[target];e=observed-m
            sd=np.sqrt(np.diag(c)); z=norm.ppf(.95)
            lo=m-z*sd;hi=m+z*sd;hits=(observed>=lo)&(observed<=hi)
            row={'origin_date':dates[t].isoformat(),'target_date':dates[target].isoformat(),'horizon':h,'model':model,
                 'dss':dss(observed,m,c),'gaussian_negative_log_score':float(.5*(8*np.log(2*np.pi)+dss(observed,m,c))),
                 'energy_score':energy_score(draws,observed),'variogram_score':variogram_score(draws,observed),
                 'squared_error_average':float(np.mean(e**2)),'mae':float(np.mean(abs(e))),
                 'coverage90':float(hits.mean()),'width90':float(np.mean(hi-lo))}
            rows.append(row)
            for j,tau in enumerate(TAU):
                maturity_rows.append({'origin_date':dates[t].isoformat(),'horizon':h,'model':model,'maturity':tau,
                                      'squared_error':e[j]**2,'absolute_error':abs(e[j]),'coverage90':int(hits[j]),'pit':norm.cdf(e[j]/sd[j]),'width90':hi[j]-lo[j]})
            # Clearly labelled first-order equal-market-value duration shock, not realized bond return.
            delta=((draws-y[t])/100)@(TAU/8)
            loss=float(((observed-y[t])/100)@(TAU/8))
            q95=np.quantile(delta,.95);q99=np.quantile(delta,.99)
            risk.append({'origin_date':dates[t].isoformat(),'horizon':h,'model':model,'duration_shock_loss':loss,'VaR95':q95,'ES95':delta[delta>=q95].mean(),'VaR99':q99,'ES99':delta[delta>=q99].mean()})
            draw_store.append(draws);mean_store.append(m);cov_store.append(c);target_store.append(observed);key_store.append([dates[t].isoformat(),str(h),model])
    write_csv(out/'scores.csv',rows);write_csv(out/'maturity_by_origin.csv',maturity_rows);write_csv(out/'origin_audit.csv',audit);write_csv(out/'risk_scenarios.csv',risk)
    summary=[]
    for h in [1,4]:
        for model in MODELS:
            s=[r for r in rows if r['horizon']==h and r['model']==model]
            summary.append({'horizon':h,'model':model,'n':len(s),
                            **{k:float(np.mean([r[k] for r in s])) for k in ['dss','gaussian_negative_log_score','energy_score','variogram_score','mae','coverage90','width90']},
                            'pooled_rmse':float(np.sqrt(np.mean([r['squared_error_average'] for r in s])))})
    write_csv(out/'summary.csv',summary)
    np.savez_compressed(out/'forecasts.npz',scenarios=np.array(draw_store),means=np.array(mean_store),covariances=np.array(cov_store),targets=np.array(target_store),keys=np.array(key_store))
    metadata={'status':'classical_baselines_only; main Bayesian question unresolved','seed':20261027,'lambda_per_year':lam,'initial_sse':sse,'n_origins':len(set(t for t,h,s in selected)),'forecast_records':len(rows),'models':MODELS,'energy_ensemble':1024,'initial_training_n':len(initial),'first_origin':audit[0]['origin_date'],'last_origin':audit[-1]['origin_date'],'elapsed_s':time.perf_counter()-start,'utc':datetime.now(timezone.utc).isoformat(),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'limitations':['historical/revised vintage; not publication-time real-time','BoC publication lag means same-week curve unavailable live','diagonal reconstruction covariance','Gaussian model conditional on fitted parameters']}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2));print(summary)

if __name__=='__main__':run()
