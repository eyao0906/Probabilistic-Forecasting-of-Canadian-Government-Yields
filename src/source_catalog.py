"""Build a replayable public-source catalog from preserved transaction evidence.

This reconstructs acquisition inputs, not the exact historical interactive commands.
The committed JSON catalog is sufficient to replay acquisition without this generator.
"""
from pathlib import Path
import json,hashlib,re,urllib.parse
ROOT=Path(__file__).resolve().parents[1]


def build_catalog(root=ROOT):
    from .acquisition import safe_target
    root=Path(root).resolve();sources=[]
    # Existing committed pins are an authority, not inputs to be regenerated
    # from whichever mutable response happens to occupy a historical path.
    committed=root/'config/sources.json';pins={}
    if committed.exists():
        for row in json.loads(committed.read_text(encoding='utf-8'))['sources']:
            if not row.get('sha256'): continue
            path=row['path'];pins[path]=row['sha256'];body=safe_target(root,path)
            if body.exists() and hashlib.sha256(body.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Historical source digest differs from committed pin: '+row['id'])
            if not body.exists() and row.get('required',True):
                raise ValueError('Required pinned historical body missing: '+row['id'])
    def add(key,path,url,kind,required=True,**extra):
        path=path.replace('\\','/')
        row={'id':key,'path':path,'url':url,'kind':kind,'required':required,**extra}
        p=safe_target(root,path)
        if path in pins: row['sha256']=pins[path]
        elif p.exists(): row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        sources.append(row)
    add('boc_original','data/raw/boc_daily_2000_2026.csv',
        'https://www.bankofcanada.ca/stats/results/csv?lookupPage=lookup_yield_curve.php&dFrom=2000-01-01&dTo=2026-10-09','boc_csv',
        provenance='Supplied raw vintage. URL reconstructed from supplied downloader/template; original transaction not authenticated.')
    add('fed_original','data/raw/feds200628.csv','https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv','fed_csv',
        provenance='Supplied US raw file, retained for historical completeness; not the Canadian experiment.')
    for record in json.loads((root/'docs/sources/retrieval_log.json').read_text(encoding='utf-8')):
        key=record['key'];path='docs/sources/'+key+'.txt'
        kind='boc_csv' if key=='boc_source_check_2015' else 'text'
        add(key,path,record['url'],kind,retrieved_utc=record.get('utc'),provenance='Preserved source-check log; cache verification is not a new retrieval.')
    manifest=root/'docs/sources/literature_verified/retrieval_manifest.json'
    seen=set()
    for record in json.loads(manifest.read_text(encoding='utf-8')):
        key='lit_'+record['id'];url=record['requested_url']
        if key in seen: continue
        seen.add(key)
        exists=bool(record.get('original'))
        path='docs/sources/literature_verified/'+record.get('original','unavailable_'+record['id']+'.body')
        success=record.get('pdf_magic',False)
        failure=not exists or record.get('status',0)>=400 or record['id'] in ['dawid_sebastiani','dawid_euclid_pdf']
        kind='pdf' if success else ('failure_evidence' if failure and exists else 'html')
        add(key,path,url,kind,required=exists,retrieved_utc=record.get('accessed_utc'),
            historical_status=record.get('status'),historical_error=record.get('error'),
            availability='retrieved_pdf' if success else ('unavailable_full_text' if failure else 'html_record'))
    for key,url in [('bolder_failed','https://publications.gc.ca/collections/Collection/FB3-2-104-48E.pdf')]:
        add(key,'docs/sources/bolder2004_download_returned_html.txt',url,'failure_evidence',availability='unavailable_full_text')
    bib=(root/'references.bib').read_text(encoding='utf-8')
    for entry in re.split(r'(?=@\w+\{)',bib):
        match=re.search(r'@\w+\{([^,]+),',entry);doi=re.search(r'\bdoi\s*=\s*\{([^}]+)\}',entry,re.I)
        if not match or not doi: continue
        key=match[1];value=doi[1]
        old=root/'docs/sources'/f'{key}.json'
        if old.exists():
            actual=json.loads(old.read_text(encoding='utf-8')).get('message',{}).get('DOI')
            if actual and actual.lower()==value.lower(): path=old.relative_to(root).as_posix()
            else:
                if actual:
                    add('historical_metadata_'+key,old.relative_to(root).as_posix(),'https://api.crossref.org/works/'+urllib.parse.quote(actual,safe=''),'crossref_json',doi=actual,provenance='Historical metadata does not verify the current bibliography entry.')
                path='docs/sources/replayed_metadata/'+key+'.json'
        else: path='docs/sources/replayed_metadata/'+key+'.json'
        add('metadata_'+key,path,'https://api.crossref.org/works/'+urllib.parse.quote(value,safe=''),'crossref_json',doi=value)
    return {'schema_version':1,'sources':sources,'notes':['Pinned bytes reproduce the supplied vintage; a new live retrieval may differ.','Failure evidence is not full-text access. Unavailable records remain explicit.','No credentials or access-control bypass are used.']}


if __name__=='__main__':
    target=ROOT/'config/sources.json'
    target.write_text(json.dumps(build_catalog(),indent=2),encoding='utf-8')
    print(target)
