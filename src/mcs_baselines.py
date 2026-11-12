"""Range-statistic MCS for ACCEPTED M0--M2 only, via arch's tested implementation.

This replaces no legacy output and says nothing about unaccepted M3--M5.
"""
import numpy as np
from arch.bootstrap import MCS
import arch


def compute(losses,names,reps=1999,seed=20261030,block_size=4):
    data=np.asarray(losses,float)
    if data.ndim!=2 or data.shape[1]!=len(names) or not np.isfinite(data).all():
        raise ValueError('Finite common-date loss matrix required')
    procedure=MCS(data,size=.10,reps=reps,block_size=block_size,method='R',bootstrap='stationary',seed=seed)
    procedure.compute()
    return {'model_universe':list(names),'included':[names[int(i)] for i in procedure.included],
            'pvalues':{names[int(i)]:float(row.iloc[0]) for i,row in procedure.pvalues.iterrows()},
            'method':'R','confidence_level':.90,'bootstrap':'stationary','expected_block_length_origins':block_size,
            'repetitions':reps,'seed':seed,'n_origins':len(data),'arch_version':arch.__version__,
            'limitation':'Restricted to accepted M0-M2; not a six-model confidence set. No finite-sample confidence guarantee.'}


if __name__=='__main__':
    import json,hashlib
    from pathlib import Path
    import pandas as pd
    root=Path(__file__).resolve().parents[1]
    source=root/'results/validated_baselines/scores.csv'
    scores=pd.read_csv(source)
    reports={}
    for horizon in [1,4]:
        for loss in ['dss','energy_score']:
            matrix=scores[scores.horizon==horizon].pivot(index='origin_date',columns='model',values=loss)
            if matrix.isna().any().any(): raise ValueError('Noncommon origins')
            reports[f'h{horizon}_{loss}']=compute(matrix.to_numpy(),list(matrix.columns))
    reports['input_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    output=root/'results/mcs_baselines/mcs_arch.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))
