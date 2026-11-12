"""Verify preserved evidence and collect this audit's executed-code snapshot.

Read-only with respect to canonical data/results; new evidence uses exclusive
creation. This is an audit replay helper, not recovered historical commands.
Run once after tests: .venv/Scripts/python.exe -B reproduction/acquisition_audit_20261027T181110Z/finalize_audit.py
"""
from pathlib import Path
import hashlib
import json
import re
import sys
from collections import Counter

AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[1]
sys.path.insert(0,str(ROOT))
from src.acquisition import atomic_new,validate_payload
from src.source_catalog import build_catalog


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    payload=path.read_bytes()
    return {'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}


def save(name,value):
    atomic_new(AUDIT/name,json.dumps(value,indent=2).encode('utf-8'))


before=load(AUDIT/'preserved_before.json')
missing=[name for name in before if not (ROOT/name).is_file()]
after={name:digest(ROOT/name) for name in before if (ROOT/name).is_file()}
changed=[name for name in after if after[name]!=before[name]]
save('preserved_after.json',after)
preservation={'original_files':len(before),'verified_unchanged_files':len(after)-len(changed),
              'missing':missing,'changed':changed,'passed':not missing and not changed,
              'raw_and_source_cache_unchanged':not any(p.startswith(('data/raw/','docs/sources/')) for p in missing+changed),
              'nonfixture_results_unchanged':not any(p.startswith('results/') and '/test_fixture/' not in p and '/independent_test_fixture/' not in p for p in missing+changed),
              'issue':'Full-suite pre-existing M3 tests write timing metadata directly into canonical test_fixture directories. Original hashes are retained, but previous bytes are not recoverable from the repository; preservation gate remains FAILED.' if changed else None}
for name in changed:
    atomic_new(AUDIT/'changed_test_fixtures'/name,(ROOT/name).read_bytes())
save('preservation_check.json',preservation)

catalog=load(ROOT/'config/sources.json')
rebuilt=build_catalog(ROOT)
save('rebuilt_catalog.json',rebuilt)
assert rebuilt==catalog,'Rebuilt catalog differs; inspect instead of repinning'
offline=load(AUDIT/'offline_pinned_replay_final.json')
assert offline['source_count']==len(offline['records'])==len(catalog['sources'])
assert offline['passed'] and not offline['robots_transactions']
assert not any(r['status']=='error' and r.get('required',True) for r in offline['records'])

live_reports=[load(AUDIT/name) for name in ['live_transactions.json','live_literature_alternative.json']]
records=[r for report in live_reports for r in report['records']]
assert sum(r['source_count'] for r in live_reports)==len(records)
assert len({r['id'] for r in records})==len(records)
live_root=AUDIT/'live'
specs={s['id']:s for s in catalog['sources']}
verified=[]
for record in records:
    if record['status']!='downloaded': continue
    path=live_root/record['path'];actual=digest(path)
    assert actual=={k:record[k] for k in ['sha256','bytes']},record['id']
    assert actual['sha256']==load(live_root/'_acquisition'/f"{record['id']}.json")['sha256']
    spec=specs[record['id']]
    content=validate_payload(spec['kind'],path.read_bytes(),spec.get('doi'))
    assert content==record['content'],record['id']
    verified.append(record['id'])
for required in ['boc_original','fed_original','lit_omori_uci','metadata_omori2007stochastic']:
    assert required in verified,required
text=live_root/'docs/sources/extracted_utf8/lit_omori_uci.txt'
text.read_text(encoding='utf-8',errors='strict')
assert digest(text)['sha256']==load(text.with_suffix('.txt.json'))['text_sha256']

transactions=[load(p) for p in (live_root/'_transactions').glob('*.json')]
robots=[load(p) for p in (live_root/'_robots').glob('*.json')]
assert len(transactions)==len(records)
for event in robots:
    assert digest(live_root/event['body_path'])=={k:event[k] for k in ['sha256','bytes']}
assert len(robots)==sum(len(r['robots_transactions']) for r in live_reports)

log=(AUDIT/'tests_full_green.txt').read_text(encoding='utf-8')
match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
assert match and log.rstrip().endswith('OK')
test_count=int(match[1]);assert len(re.findall(r'^test_',log,re.M))==test_count

code_paths=[
    'src/acquisition.py','src/source_catalog.py','src/verify_data_sources.py','src/prepare_data.py',
    'src/source_data_check.py','src/verify_sources.py','src/legacy_guard.py','fetch_raw.py',
    'docs/sources/literature_verified/retrieve.py',
    'docs/archive_pre_resumption/docs__sources__literature_verified__retrieve.py.txt',
    'docs/archive_pre_resumption/src__verify_sources.py.txt',
    'docs/archive_pre_resumption/src__source_data_check.py.txt',
    'config/sources.json','requirements-lock.txt',
    'tests/test_acquisition.py','tests/test_source_catalog.py','tests/test_source_comparison.py','tests/test_prepare_safety.py',
]
code_manifest={}
for relative in code_paths:
    source=ROOT/relative;target=AUDIT/'executed_code'/relative
    atomic_new(target,source.read_bytes());code_manifest[relative]=digest(source)
save('executed_code_manifest.json',code_manifest)

prefix='export TMPDIR=C:/Users/ethan/AppData/Local/hermes/cache/scratch; '
python='.venv/Scripts/python.exe -B'
commands=[
    {'label':'pinned_offline','command':f'{python} -m src.acquisition --report <NEW_REPORT>.json'},
    {'label':'live_primary','command':f'{python} -m src.acquisition --live --root <NEW_EMPTY_ROOT> --only boc_original fed_original lit_diebold_li_ucm metadata_diebold2006forecasting --extract --report <NEW_REPORT>.json','observed_exit_code':1},
    {'label':'live_representative_literature','command':f'{python} -m src.acquisition --live --root <NEW_EMPTY_ROOT> --only lit_omori_uci metadata_omori2007stochastic --extract --report <NEW_REPORT>.json','observed_exit_code':0},
    {'label':'compare_boc','command':f'{python} -m src.verify_data_sources --original data/raw/boc_daily_2000_2026.csv --fresh <LIVE_ROOT>/data/raw/boc_daily_2000_2026.csv --report <NEW_REPORT>.json'},
    {'label':'compare_fed','command':f'{python} -m src.verify_data_sources --kind fed_csv --original data/raw/feds200628.csv --fresh <LIVE_ROOT>/data/raw/feds200628.csv --report <NEW_REPORT>.json'},
    {'label':'regressions','command':f'{python} -m unittest discover -s tests -v','observed_exit_code':0},
]
for command in commands: command['command']=prefix+command['command']
save('replay_commands.json',{'provenance':'Reconstructed runnable equivalents; placeholders must be replaced with NEW paths. Not a recovered transcript of original historical interactive commands. Current actual executions are evidenced by the saved logs and transaction timestamps.','working_directory':str(ROOT),'commands':commands})
comparisons={name:load(AUDIT/(name+'.json')) for name in ['boc_research_period_comparison','fed_research_period_comparison','boc_full_comparison','fed_full_comparison']}
summary={
    'audit_directory':str(AUDIT),'overall_audit_passed':preservation['passed'],'preservation':preservation,
    'catalog_equal_after_rebuild':rebuilt==catalog,'historical_pins_changed':False,
    'offline':{'passed_required':offline['passed'],'source_count':offline['source_count'],'status_counts':dict(Counter(r['status'] for r in offline['records'])),'errors':[r for r in offline['records'] if r['status']=='error']},
    'live':{'attempted_sources':len(records),'status_counts':dict(Counter(r['status'] for r in records)),'readback_verified_downloads':verified,'source_transactions':len(transactions),'robots_transactions':len(robots),'representative_pdf_pages':next(r['content']['pages'] for r in records if r['id']=='lit_omori_uci'),'errors':[r for r in records if r['status']=='error']},
    'comparisons':comparisons,'tests':{'passed':True,'count':test_count,'seconds':float(match[2]),'log':'tests_full_green.txt'},
    'executed_code_snapshot_files':len(code_manifest),
    'provenance_limitation':'Pinned replay verifies supplied evidence. Reconstructed code/commands and present live transactions do not recover missing exact historical interactive commands or point-in-time data releases.'
}
save('audit_summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},indent=2))
raise SystemExit(0 if preservation['passed'] else 1)
