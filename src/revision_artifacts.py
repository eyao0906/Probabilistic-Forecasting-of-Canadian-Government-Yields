"""Regenerate supplementary tables/traces from recorded computations, never refit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def build(out=None,fig=None):
    out=Path(out) if out is not None else ROOT/'paper/generated'
    fig=Path(fig) if fig is not None else ROOT/'figures/validated'
    out.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
    lag=pd.read_csv(ROOT/'results/availability_robustness/summary.csv')
    lag=lag[lag.horizon==4][['model','dss','energy_score','pooled_rmse','coverage90']]
    lag.columns=['Model','DSS','Energy','RMSE','Coverage']
    text=['| '+' | '.join(lag.columns)+' |','|'+'|'.join(['---']*len(lag.columns))+'|']
    text+=['| '+' | '.join(f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in lag.itertuples(index=False,name=None)]
    text+=['',': Four-week forecasts with an assumed 14-calendar-day availability lag; 130 common origins. Historical vintages and exact release dates are not recovered. {#tbl-availability}','']
    (out/'availability_h4.md').write_text('\n'.join(text),encoding='utf-8')
    d=pd.read_csv(ROOT/'results/bayesian_revised_pilot/extended/diagnostics.csv')
    text=['| Model | Factor | Max. rank R-hat | Min. bulk ESS | Min. tail ESS |','|---|---|---|---|---|']
    for (model,factor),group in d.groupby(['model','factor'],sort=False):
        text.append(f'| {model} | {factor} | {group.rhat.max():.3f} | {group.ess_bulk.min():.1f} | {group.ess_tail.min():.1f} |')
    text+=['',': Revised initial-window pilot: four chains, 3,000 warm-up and 6,000 retained draws per chain. Acceptance requires all monitored rank R-hat at most 1.01 and bulk/tail ESS at least 400. These are diagnostics, not forecasting results. {#tbl-revised-pilot}','']
    (out/'revised_pilot.md').write_text('\n'.join(text),encoding='utf-8')
    with np.load(ROOT/'results/bayesian_revised_pilot/extended/chains.npz',allow_pickle=False) as a:
        image,axes=plt.subplots(3,3,figsize=(8,6),sharex=True)
        for j,factor in enumerate(['level','slope','curvature']):
            vals=a['M5_'+factor]
            for k,(idx,param) in enumerate([(2,'omega'),(3,'phi'),(4,'q')]):
                for chain in range(vals.shape[0]): axes[j,k].plot(vals[chain,:,idx],lw=.35,alpha=.7)
                axes[j,k].set_title(factor+': '+param,fontsize=9)
        image.tight_layout();image.savefig(fig/'revised_pilot_traces.png',dpi=160);plt.close(image)
    reports=json.loads((ROOT/'results/mcs_baselines/mcs_arch.json').read_text())
    text=['| Horizon | Loss | Retained models |','|---|---|---|']
    for h in [1,4]:
        for loss in ['dss','energy_score']:
            report=reports[f'h{h}_{loss}']
            text.append(f'| {h} | {loss} | '+', '.join(report['included'])+' |')
    text+=['',': Restricted 90% Model Confidence Sets, range statistic, stationary bootstrap with expected block length four origins, 1,999 replicates. Universe is M0–M2 only. {#tbl-mcs}','']
    (out/'mcs_baselines.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':build()
