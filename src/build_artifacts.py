"""Build manuscript tables, plots and numerical audit from accepted saved outputs."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.scores import hac_dm, var_backtests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/generated';OUT.mkdir(exist_ok=True)
FIG=ROOT/'figures/validated';FIG.mkdir(exist_ok=True)
R=ROOT/'results/validated_baselines'


def table(df,name,caption,label,formats=None):
    # Plain Markdown rather than executable placeholders or manually copied values.
    formats=formats or {}
    lines=['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']
    for _,row in df.iterrows():
        cells=[]
        for c,x in row.items():
            cells.append(formats.get(c,lambda v:f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v))(x))
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['',f': {caption} {{#{label}}}','']
    (OUT/name).write_text('\n'.join(lines),encoding='utf-8')


def main():
    s=pd.read_csv(R/'summary.csv'); detail=pd.read_csv(R/'maturity_by_origin.csv'); risk=pd.read_csv(R/'risk_scenarios.csv')
    old_diag=pd.read_csv(ROOT/'results/archive_before_validation/full_130_origin/mcmc_diagnostics.csv')
    raw=pd.read_csv(ROOT/'data/processed/canada_weekly.csv',parse_dates=['Date'])
    cols=['y0.25','y0.5','y1','y2','y5','y7','y10','y30']; tau=[.25,.5,1,2,5,7,10,30]
    sample=raw[(raw.Date>='2000-01-01')&(raw.Date<='2025-12-31')]
    train=sample[sample.Date<='2015-12-31'];test=sample[sample.Date>='2016-01-01']
    stats=sample[cols].agg(['mean','std','min','max']).T.reset_index()
    stats.columns=['Maturity','Mean','SD','Min','Max'];stats.Maturity=tau
    table(stats,'data_summary.md','Weekly Canadian yields, 2000–2025, in percentage points.','tbl-data')
    for h in [1,4]:
        v=s[s.horizon==h][['model','dss','energy_score','pooled_rmse','mae','coverage90','width90']].copy()
        v.columns=['Model','DSS','Energy','RMSE','MAE','Coverage','Width']
        table(v,f'scores_h{h}.md',f'Accepted classical forecasts at {h}-week horizon; 130 origins per model. RMSE is pooled across origins and maturities; yields and width are in percentage points.','tbl-h'+str(h))
    matur=[]
    for (model,t),d in detail[detail.horizon==4].groupby(['model','maturity'],sort=True):
        matur.append({'Model':model,'Maturity':t,'RMSE':np.sqrt(d.squared_error.mean()),'Coverage':d.coverage90.mean()})
    mat=pd.DataFrame(matur);table(mat,'maturity_h4.md','Four-week point accuracy and marginal 90% interval coverage.','tbl-maturity')
    diag=old_diag[['factor','parameter','rhat','ess']].copy();diag.columns=['Factor','Parameter','Split R-hat','ESS estimate']
    table(diag,'failed_diagnostics.md','Failed two-chain diagnostics at 28 November 2025 from the quarantined run. ESS is an approximate legacy estimate, not rank-normalized bulk/tail ESS.','tbl-failed-diag')
    risk_rows=[]
    for (h,model),d in risk.groupby(['horizon','model']):
        if h!=4:continue
        bt=var_backtests(d.duration_shock_loss.to_numpy(),d.VaR95.to_numpy(),.05)
        risk_rows.append({'Model':model,'Mean VaR95':100*d.VaR95.mean(),'Mean ES95':100*d.ES95.mean(),'Exceptions':bt['exceptions'],'UC p':bt['p_unconditional_coverage']})
    table(pd.DataFrame(risk_rows),'risk_h4.md','Four-week first-order duration-shock risk for an equal-market-value maturity basket. VaR/ES are percentages of initial value, not total-return risk measures. UC tests are exploratory.','tbl-risk')
    score=pd.read_csv(R/'scores.csv');comparisons=[]
    for h in [1,4]:
        for loss in ['dss','energy_score']:
            p=score[score.horizon==h].pivot(index='origin_date',columns='model',values=loss)
            dm=hac_dm((p.M2-p.M0).to_numpy(),max_lag=3)
            comparisons.append({'Horizon':h,'Loss':loss,'M2 minus M0':dm['mean_difference'],'HAC SE':dm['hac_se'],'95% lower':dm['mean_difference']-1.96*dm['hac_se'],'95% upper':dm['mean_difference']+1.96*dm['hac_se']})
    table(pd.DataFrame(comparisons),'comparison.md','Descriptive paired loss differences with Bartlett HAC standard errors (three origin lags). Normal intervals are approximate; no claim about Bayesian volatility follows.','tbl-comparison')
    fig,ax=plt.subplots(figsize=(6.6,3.4))
    for c in ['y0.25','y2','y10','y30']:ax.plot(sample.Date,sample[c],lw=.8,label=c[1:]+' years')
    ax.axvline(pd.Timestamp('2016-01-01'),color='black',ls='--',lw=.8);ax.set(ylabel='Yield (percentage points)',xlabel='Observation date');ax.legend(ncol=4,fontsize=8,frameon=False);fig.tight_layout();fig.savefig(FIG/'yield_history.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,3.4))
    ax.plot(tau,train[cols].mean(),'-o',label='2000–2015');ax.plot(tau,test[cols].mean(),'-s',label='2016–2025');ax.set(xlabel='Maturity (years)',ylabel='Yield (percentage points)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'average_curve.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7,3.5))
    for h in [1,4]:
        sub=s[s.horizon==h];axs[0].plot(sub.model,sub.dss,'-o',label=f'{h} weeks');axs[1].plot(sub.model,sub.energy_score,'-o',label=f'{h} weeks')
    axs[0].set(ylabel='Mean DSS (lower is better)');axs[1].set(ylabel='Energy score (lower is better)');axs[0].legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'scores.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7,3.5))
    for model,d in mat.groupby('Model'):
        axs[0].plot(d.Maturity,d.RMSE,'-o',label=model);axs[1].plot(d.Maturity,d.Coverage,'-o',label=model)
    axs[1].axhline(.9,color='black',ls='--',lw=.8);axs[0].set(xlabel='Maturity',ylabel='RMSE (percentage points)');axs[1].set(xlabel='Maturity',ylabel='90% interval coverage',ylim=(.65,1));axs[0].legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'maturity.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,3.3))
    for model,d in detail[detail.horizon==4].groupby('model'):
        ax.hist(d.pit,bins=np.linspace(0,1,11),histtype='step',density=True,label=model)
    ax.axhline(1,color='black',ls='--',lw=.8);ax.set(xlabel='Marginal Gaussian PIT',ylabel='Density');ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'pit.png',dpi=180);plt.close(fig)
    # Audit saved scores by recalculating from archived accepted moments.
    a=np.load(R/'forecasts.npz',allow_pickle=False)
    from src.scores import dss
    recomputed=np.array([dss(y,m,c) for y,m,c in zip(a['targets'],a['means'],a['covariances'])])
    audit={'score_rows':len(score),'unique_keys':len(score.drop_duplicates(['origin_date','horizon','model'])),'n_origins':score.origin_date.nunique(),'max_abs_dss_reconstruction_error':float(np.max(abs(recomputed-score.dss.to_numpy()))),'positive_definite_min_eigenvalue':float(np.linalg.eigvalsh(a['covariances']).min()),'initial_train_n':len(train),'oos_weekly_n':len(test),'sample_n':len(sample),'abstract_limit':250,'models_accepted':['M0','M1','M2'],'models_unvalidated':['M3','M4','M5'],'old_max_rhat':float(old_diag.rhat.max()),'old_min_ess':float(old_diag.ess.min())}
    (R/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
