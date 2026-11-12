"""Auditable public acquisition and pinned-cache verification. No credentials needed."""
from __future__ import annotations
import csv
import io
import json


def read_yield_csv(payload, kind):
    """Parse dated numeric CSV, allowing only documented blank/NA missing cells.

    The Fed file has a staff-product preamble. BoC's preserved CSV has a
    trailing empty column. Neither is silently treated as an HTML download.
    """
    import math
    from datetime import date
    text=payload.decode('utf-8-sig',errors='strict')
    if '<html' in text.lower() or '<!doctype html' in text.lower():
        raise ValueError('Expected yield CSV, received HTML')
    lines=text.splitlines()
    if kind=='fed_csv':
        required={'Date',*[f'SVENY{i:02d}' for i in range(1,31)],'BETA0','BETA1','BETA2','BETA3','TAU1','TAU2'}
        starts=[i for i,line in enumerate(lines) if line.startswith('Date,')]
        if len(starts)!=1: raise ValueError('Expected Federal Reserve dated CSV header')
        lines=lines[starts[0]:]
        selected=[f'SVENY{i:02d}' for i in range(1,31)]
    elif kind=='boc_csv':
        selected=['ZC025YR','ZC050YR','ZC100YR','ZC200YR','ZC500YR','ZC700YR','ZC1000YR','ZC3000YR']
        required={'Date',*selected}
    else: raise ValueError('Unsupported yield CSV kind')
    reader=csv.reader(lines,strict=True)
    fields=[x.strip() for x in next(reader,[])]
    named=[x for x in fields if x]
    if len(named)!=len(set(named)) or not required.issubset(fields):
        raise ValueError('Missing or duplicate yield CSV columns')
    rows={};complete=0
    for values in reader:
        if not values: continue
        if len(values)!=len(fields): raise ValueError('Ragged yield CSV row')
        row={k:v.strip() for k,v in zip(fields,values) if k}
        day=row['Date']
        if len(day)!=10 or date.fromisoformat(day).isoformat()!=day:
            raise ValueError('Invalid source date')
        if day in rows: raise ValueError('Duplicate source date')
        for column,value in row.items():
            if column=='Date' or value.lower() in ('','na','n/a'): continue
            try: numeric=float(value)
            except ValueError as error: raise ValueError('Non-numeric yield CSV cell') from error
            if not math.isfinite(numeric): raise ValueError('Non-finite yield CSV cell')
        if all(row[c].lower() not in ('','na','n/a') for c in selected): complete+=1
        rows[day]=row
    if not rows or not complete: raise ValueError('No complete yield curves in CSV')
    return fields,rows,{'rows':len(rows),'columns':len(fields),'complete_rows':complete,
                       'min_date':min(rows),'max_date':max(rows),'type':kind}


def validate_payload(kind, payload, doi=None):
    if kind == 'pdf':
        if not payload.startswith(b'%PDF-'):
            raise ValueError('Expected PDF, received non-PDF content')
        from pypdf import PdfReader
        try:
            reader=PdfReader(io.BytesIO(payload))
            return {'pages':len(reader.pages),'type':'pdf'}
        except Exception as error:
            raise ValueError('Unparseable PDF') from error
    if kind in ('boc_csv','fed_csv'):
        return read_yield_csv(payload,kind)[2]
    if kind == 'crossref_json':
        try:
            message=json.loads(payload)['message']
            actual=message['DOI']
        except Exception as error:
            raise ValueError('Invalid Crossref metadata') from error
        if doi and actual.lower()!=doi.lower():
            raise ValueError('Crossref DOI mismatch')
        return {'doi':actual,'type':'crossref_json'}
    if kind in ('text','html','failure_evidence','fed_csv'):
        if not payload: raise ValueError('Empty source response')
        if kind=='fed_csv' and b'SVENY01' not in payload:
            raise ValueError('Expected Federal Reserve Svensson data')
        return {'type':kind,'bytes':len(payload)}
    raise ValueError('Unsupported content kind: '+str(kind))


def safe_target(root, relative):
    from pathlib import Path, PureWindowsPath
    p=Path(relative)
    if p.is_absolute() or PureWindowsPath(relative).is_absolute() or '..' in p.parts or '..' in PureWindowsPath(relative).parts:
        raise ValueError('Source path must be relative and confined to the cache root')
    base=Path(root).resolve(); target=(base/p).resolve()
    if not target.is_relative_to(base): raise ValueError('Source path escapes cache root')
    return target


def atomic_new(path, payload):
    """Publish validated bytes without replacing another process's existing file."""
    import os, uuid
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.part')
    try:
        with temporary.open('xb') as handle:
            handle.write(payload);handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)  # atomic, fails if destination already exists
    finally:
        temporary.unlink(missing_ok=True)


USER_AGENT='MQF-reproducible-public-research/2.0'
ROBOT_AGENT='MQF-reproducible-public-research'


def public_url(url):
    import urllib.parse
    parsed=urllib.parse.urlparse(url)
    if (parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.fragment or any(ord(c)<33 for c in url)):
        raise ValueError('Only credential-free public HTTPS source URLs are supported')
    return parsed


class PublicTransport:
    """Robots-aware default transport shared by every public acquisition entry.

    Redirects are refused rather than fetching a target whose robots policy
    has not been checked. No browser challenges or authentication are bypassed.
    """
    def __init__(self, root, delay=1.0):
        import urllib.request,urllib.error
        self.root=root;self.delay=max(0,delay);self.policies={};self.previous={};self.events=[]
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise urllib.error.HTTPError(req.full_url,code,'Redirect refused: '+newurl,headers,fp)
        self.opener=urllib.request.build_opener(NoRedirect())

    def __call__(self, request, timeout=120):
        import hashlib,time,uuid,urllib.request,urllib.error,urllib.robotparser
        from datetime import datetime,timezone
        from pathlib import Path
        parsed=public_url(request.full_url);origin=f'{parsed.scheme}://{parsed.netloc}'
        if origin not in self.policies:
            url=origin+'/robots.txt';raw=b'';headers={};status=None;final_url=url;error=None
            try:
                with self.opener.open(urllib.request.Request(url,headers={'User-Agent':USER_AGENT}),timeout=30) as response:
                    raw=response.read();headers=dict(response.headers.items());status=response.status;final_url=response.geturl()
            except urllib.error.HTTPError as exc:
                raw=exc.read();headers=dict(exc.headers.items());status=exc.code;error=exc
            except Exception as exc:
                error=exc
            event={'requested_url':url,'final_url':final_url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                   'http_status':status,'headers':headers,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            prefix=Path(self.root)/'_robots'/uuid.uuid4().hex
            atomic_new(prefix.with_suffix('.body'),raw)
            event['body_path']=prefix.with_suffix('.body').relative_to(Path(self.root)).as_posix()
            try:
                if status==404:
                    policy=urllib.robotparser.RobotFileParser(url);policy.parse(['User-agent: *','Allow: /'])
                else:
                    if error: raise error
                    if status!=200 or final_url!=url: raise ValueError('Unexpected robots response or redirect')
                    length=next((v for k,v in headers.items() if k.lower()=='content-length'),None)
                    if length is not None and len(raw)!=int(length): raise ValueError('Incomplete robots response')
                    text=raw.decode('utf-8-sig',errors='strict')
                    if '<html' in text.lower() or '<!doctype' in text.lower(): raise ValueError('HTML is not robots.txt')
                    policy=urllib.robotparser.RobotFileParser(url);policy.parse(text.splitlines())
                self.policies[origin]=policy;event['policy_status']='parsed_404_allow' if status==404 else 'parsed'
            except Exception as exc:
                event['policy_status']='error';event['error']=str(exc)
                raise
            finally:
                atomic_new(prefix.with_suffix('.json'),json.dumps(event,indent=2).encode('utf-8'));self.events.append(event)
        policy=self.policies[origin]
        if not policy.can_fetch(ROBOT_AGENT,request.full_url): raise PermissionError('robots.txt disallows source retrieval')
        interval=max(self.delay,policy.crawl_delay(ROBOT_AGENT) or 0)
        rate=policy.request_rate(ROBOT_AGENT)
        if rate: interval=max(interval,rate.seconds/rate.requests)
        time.sleep(max(0,interval-(time.monotonic()-self.previous.get(origin,0))))
        self.previous[origin]=time.monotonic()
        return self.opener.open(request,timeout=timeout)


def acquire_one(spec, root, offline=False, opener=None):
    import hashlib,re,urllib.request,urllib.parse
    from datetime import datetime,timezone
    from pathlib import Path
    root=Path(root).resolve()
    key=spec['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',key): raise ValueError('Invalid source identifier')
    url=spec['url'];public_url(url)
    target=safe_target(root,spec['path'])
    now=datetime.now(timezone.utc).isoformat()
    if target.exists():
        raw=target.read_bytes(); actual=hashlib.sha256(raw).hexdigest()
        pin=spec.get('sha256')
        sidecar=Path(root)/'_acquisition'/f'{key}.json'
        if not pin and sidecar.exists():
            record=json.loads(sidecar.read_text(encoding='utf-8'))
            if record.get('requested_url')!=url: raise ValueError('Cached URL association mismatch')
            pin=record.get('sha256')
        if not pin or actual!=pin: raise ValueError('Missing or mismatched pinned source hash: '+key)
        check=validate_payload(spec['kind'],raw,spec.get('doi'))
        return {'id':key,'status':'cached_verified','path':spec['path'],'sha256':actual,'requested_url':url,
                'cache_verified_utc':now,'original_retrieval_utc':spec.get('retrieved_utc'),'content':check}
    if offline: raise FileNotFoundError('Offline source cache missing: '+key)
    opener=opener or PublicTransport(root)
    request=urllib.request.Request(url,headers={'User-Agent':USER_AGENT})
    import uuid,urllib.error,http.client
    record={'id':key,'status':'error','path':spec['path'],'requested_url':url,'retrieved_utc':now,
            'http_status':None,'headers':{},'bytes':0}
    raw=b''
    try:
        with opener(request,timeout=120) as response:
            record.update(http_status=getattr(response,'status',200),headers=dict(response.headers.items()),final_url=response.geturl())
            try: raw=response.read()
            except http.client.IncompleteRead as error:
                raw=error.partial
                raise ValueError('Incomplete HTTP response body') from error
        if record['http_status']!=200: raise ValueError('Unexpected HTTP status '+str(record['http_status']))
        if record['final_url']!=url: raise ValueError('Redirected response refused; use a reviewed explicit source URL')
        length=next((v for k,v in record['headers'].items() if k.lower()=='content-length'),None)
        if length is not None and len(raw)!=int(length): raise ValueError('Incomplete HTTP response body')
        check=validate_payload(spec['kind'],raw,spec.get('doi'))
        actual=hashlib.sha256(raw).hexdigest()
        record.update(status='downloaded',bytes=len(raw),sha256=actual,content=check,
                      matches_historical_pin=actual==spec['sha256'] if spec.get('sha256') else None)
        atomic_new(target,raw)
        sidecar=Path(root)/'_acquisition'/f'{key}.json'
        if not sidecar.exists(): atomic_new(sidecar,json.dumps(record,indent=2).encode('utf-8'))
        return record
    except Exception as error:
        record.update(status='error',error=str(error))
        if isinstance(error,urllib.error.HTTPError):
            raw=error.read();record.update(http_status=error.code,headers=dict(error.headers.items()),final_url=error.geturl())
        record.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if raw:
            failed=safe_target(root,'_failed/'+key+'.'+record['sha256'][:12]+'.body')
            if not failed.exists(): atomic_new(failed,raw)
            record['failure_body_path']=failed.relative_to(Path(root)).as_posix()
        raise
    finally:
        record['completed_utc']=datetime.now(timezone.utc).isoformat()
        atomic_new(Path(root)/'_transactions'/f'{key}.{uuid.uuid4().hex}.json',json.dumps(record,indent=2).encode('utf-8'))


def extract_pdf(source, output):
    """Pinned PDF -> strict UTF-8 page-marked text, with extractor provenance."""
    import hashlib,importlib.metadata
    from pathlib import Path
    from pypdf import PdfReader
    source,output=Path(source),Path(output)
    payload=source.read_bytes();validate_payload('pdf',payload)
    reader=PdfReader(io.BytesIO(payload))
    text='\n\n'.join(f'===== PDF PAGE {i} =====\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages,1))
    raw=text.encode('utf-8',errors='strict')
    report={'source_sha256':hashlib.sha256(payload).hexdigest(),'text_sha256':hashlib.sha256(raw).hexdigest(),
            'pages':len(reader.pages),'encoding':'UTF-8 strict','pypdf_version':importlib.metadata.version('pypdf')}
    if output.exists():
        if output.read_bytes()!=raw: raise ValueError('Existing derived text differs; choose a new output path')
    else: atomic_new(output,raw)
    sidecar=output.with_suffix(output.suffix+'.json')
    if not sidecar.exists(): atomic_new(sidecar,json.dumps(report,indent=2).encode('utf-8'))
    return report


def run_catalog(specs, root, offline=True, delay=1.0, extract=False):
    import time,urllib.parse,urllib.request,urllib.error,urllib.robotparser,hashlib
    from pathlib import Path
    ids=[x['id'] for x in specs];paths=[x['path'] for x in specs]
    if len(ids)!=len(set(ids)) or len(paths)!=len(set(paths)):
        raise ValueError('Duplicate catalog identifiers or paths')
    records=[];transport=PublicTransport(root,delay=delay) if not offline else None
    for spec in specs:
        try:
            target=safe_target(root,spec['path'])
            record=acquire_one(spec,root,offline=offline,opener=transport)
            if spec['kind']=='failure_evidence': record['status']='documented_access_failure_cached'
            if extract and spec['kind']=='pdf':
                output=safe_target(root,'docs/sources/extracted_utf8/'+spec['id']+'.txt')
                record['extraction']=extract_pdf(target,output)
            records.append(record)
        except Exception as error:
            record={'id':spec['id'],'status':'error','error':str(error),'required':spec.get('required',True),'requested_url':spec['url']}
            if isinstance(error,urllib.error.HTTPError):
                body=error.read();record['http_status']=error.code
                if body:
                    failed=safe_target(root,'_failed/'+spec['id']+'.'+hashlib.sha256(body).hexdigest()[:12]+'.body')
                    if not failed.exists(): atomic_new(failed,body)
                    record['failure_body_sha256']=hashlib.sha256(body).hexdigest()
            records.append(record)
    return {'offline':offline,'source_count':len(records),'passed':not any(x['status']=='error' and x.get('required',True) for x in records),
            'records':records,'robots_transactions':transport.events if transport else []}


def main(argv=None):
    import argparse
    from pathlib import Path
    from datetime import datetime,timezone
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parents[1]/'config/sources.json')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--live',action='store_true',help='Fetch missing public sources; existing pinned bytes are never replaced')
    parser.add_argument('--only',nargs='*',help='Subset by exact source identifiers')
    parser.add_argument('--extract',action='store_true',help='Regenerate strict UTF-8 PDF text with pinned extraction metadata')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    specs=json.loads(args.catalog.read_text(encoding='utf-8'))['sources']
    if args.only:
        wanted=set(args.only);found={x['id'] for x in specs}
        if not wanted.issubset(found): parser.error('Unknown source identifiers: '+str(wanted-found))
        specs=[x for x in specs if x['id'] in wanted]
    report=run_catalog(specs,args.root,offline=not args.live,extract=args.extract)
    dest=args.report or args.root/'logs'/('acquisition_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    atomic_new(dest,json.dumps(report,indent=2).encode('utf-8'))
    print(json.dumps({'passed':report['passed'],'sources':report['source_count'],'report':str(dest)},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

