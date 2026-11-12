"""Recompute official-source agreement over common complete research-period dates."""
from pathlib import Path
import csv,hashlib,math,json
from datetime import date
ROOT=Path(__file__).resolve().parents[1]
COLS=['ZC025YR','ZC050YR','ZC100YR','ZC200YR','ZC500YR','ZC700YR','ZC1000YR','ZC3000YR']


def compare(original,fresh,start='2000-01-01',end='2025-12-31',kind='boc_csv'):
    from .acquisition import read_yield_csv
    date.fromisoformat(start);date.fromisoformat(end)
    if start>end: raise ValueError('Invalid comparison date range')
    columns=COLS if kind=='boc_csv' else [f'SVENY{i:02d}' for i in range(1,31)]
    scale=1. if kind=='boc_csv' else .01  # BoC decimals versus Fed percentage points.
    raw_a=Path(original).read_bytes();raw_b=Path(fresh).read_bytes()
    _,a,_=read_yield_csv(raw_a,kind);_,b,_=read_yield_csv(raw_b,kind)
    ar={d for d in a if start<=d<=end};br={d for d in b if start<=d<=end}
    def complete(rows,days):
        return {d for d in days if all(rows[d][c].lower() not in ('','na','n/a') for c in columns)}
    ac=complete(a,ar);bc=complete(b,br);common=sorted(ac&bc)
    if not common: raise ValueError('No comparable complete dates')
    diffs=[];cells=[];per_column={c:0 for c in columns}
    for day in common:
        day_diffs=[]
        for c in columns:
            av=float(a[day][c]);bv=float(b[day][c]);delta=abs(av-bv);day_diffs.append(delta)
            if delta:
                per_column[c]+=1
                cells.append({'date':day,'column':c,'original':av,'fresh':bv,'abs_native_difference':delta})
        diffs.append(max(day_diffs))
    max_native=max(diffs)
    return {'source_kind':kind,'requested_start':start,'requested_end':end,'columns':columns,
            'native_units':'decimal yield' if kind=='boc_csv' else 'percentage points',
            'compared_complete_dates':len(common),'compared_cells':len(common)*len(columns),
            'dates_with_difference':sum(v>0 for v in diffs),'differing_cells':len(cells),
            'differing_cells_by_column':per_column,
            'maximum_abs_native_yield_difference':max_native,
            'maximum_abs_decimal_yield_difference':max_native*scale,
            'maximum_abs_basis_point_difference':max_native*scale*10000,
            'largest_differences':sorted(cells,key=lambda x:(-x['abs_native_difference'],x['date'],x['column']))[:10],
            'local_raw_rows':len(a),'fresh_raw_rows':len(b),'original_min_date':min(a),'original_max_date':max(a),
            'fresh_min_date':min(b),'fresh_max_date':max(b),'compared_first_date':common[0],'compared_last_date':common[-1],
            'original_raw_dates_in_range':len(ar),'fresh_raw_dates_in_range':len(br),
            'original_complete_dates_in_range':len(ac),'fresh_complete_dates_in_range':len(bc),
            'original_incomplete_dates_in_range':len(ar-ac),'fresh_incomplete_dates_in_range':len(br-bc),
            'original_only_raw_dates_in_range':sorted(ar-br),'fresh_only_raw_dates_in_range':sorted(br-ar),
            'original_only_complete_dates_in_range':sorted(ac-bc),'fresh_only_complete_dates_in_range':sorted(bc-ac),
            'research_panel_equal':ac==bc and not cells,'bytes_equal':raw_a==raw_b,
            'original_sha256':hashlib.sha256(raw_a).hexdigest(),'fresh_sha256':hashlib.sha256(raw_b).hexdigest(),
            'limitation':'Dated numeric equality is only for the selected maturities and requested range. A new retrieval neither reconstructs historical releases nor authenticates the original transaction.'}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,default=ROOT/'data/raw/boc_daily_2000_2026.csv')
    parser.add_argument('--fresh',type=Path,default=ROOT/'docs/sources/boc_source_check_2015.txt')
    parser.add_argument('--report',type=Path,default=ROOT/'logs/source_comparison.json')
    parser.add_argument('--kind',choices=['boc_csv','fed_csv'],default='boc_csv')
    parser.add_argument('--start',default='2000-01-01')
    parser.add_argument('--end',default='2025-12-31')
    args=parser.parse_args();report=compare(args.original,args.fresh,start=args.start,end=args.end,kind=args.kind)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
