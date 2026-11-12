"""Fail-closed PDF/citation checks; output writes only on explicit execution."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]


def verify(root=ROOT, write_outputs=True, previews=True):
    from pypdf import PdfReader
    root=Path(root);p=root/'paper/MRP_Draft.pdf'
    with p.open('rb') as handle:
        reader=PdfReader(handle);texts=[page.extract_text() or '' for page in reader.pages]
    q=(root/'paper/MRP_Draft.qmd').read_text(encoding='utf-8')
    match=re.search(r'This study investigates.*?(?=```)',q,re.S)
    abstract=match.group(0).strip() if match else ''
    bib=(root/'references.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\{([^,]+),',bib))
    combined=q;missing=[]
    for include in re.findall(r'\{\{< include ([^>]+) >\}\}',q):
        path=(root/'paper'/include.strip()).resolve()
        if not path.is_relative_to(root.resolve()):
            missing.append(include);continue
        if path.exists():combined+='\n'+path.read_text(encoding='utf-8')
        else:missing.append(include)
    citations=set(re.findall(r'(?<![\w])@([A-Za-z][A-Za-z0-9_-]+)',combined))
    labels=re.findall(r'\{#((?:sec|eq|tbl|fig)-[^\s}]+)',combined)
    xrefs={x for x in citations if x.startswith(('sec-','eq-','tbl-','fig-'))}
    report={'pdf':str(p),'page_count':len(texts),'page_limit_ok':25<=len(texts)<=35,
            'abstract_words':len(abstract.split()),'abstract_limit_ok':0<len(abstract.split())<=250,
            'unknown_citations':sorted(citations-keys-xrefs),'unknown_crossrefs':sorted(xrefs-set(labels)),
            'bibliography_entries':len(keys),'duplicate_labels':sorted({x for x in labels if labels.count(x)>1}),
            'missing_includes':missing,'unresolved_pdf_crossref_markers':[i+1 for i,t in enumerate(texts) if '??' in t],
            'blank_pages':[i+1 for i,t in enumerate(texts) if len(t.strip())<10],
            'title_first':bool(texts and 'Probabilistic Forecasting' in texts[0]),
            'main_body_12pt_configured':bool(re.search(r'^fontsize:\s*12pt\s*$',q,re.M))}
    report['passed']=bool(report['page_limit_ok'] and report['abstract_limit_ok'] and report['title_first']
                          and report['main_body_12pt_configured'] and not any(report[k] for k in
                          ['unknown_citations','unknown_crossrefs','duplicate_labels','missing_includes','unresolved_pdf_crossref_markers','blank_pages']))
    if write_outputs:
        logs=root/'logs';logs.mkdir(exist_ok=True)
        (logs/'pdf_extracted.txt').write_text('\n\n'.join(texts),encoding='utf-8')
        (logs/'pdf_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if previews:
        if not write_outputs:raise ValueError('Preview generation requires explicit output writes')
        import pypdfium2 as pdfium
        out=root/'logs/pdf_preview';out.mkdir(exist_ok=True)
        doc=pdfium.PdfDocument(p)
        try:
            indices={0,1,2,len(doc)-1}
            for i,t in enumerate(texts):
                if any(s in t for s in ['Table 6.1','Table 7.1','Table 6.4','Figure 6.1','Analytical derivations']):indices.add(i)
            for i in sorted(indices):
                doc[i].render(scale=1.3).to_pil().save(out/f'page_{i+1:02d}.png')
        finally:doc.close()
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args()
    report=verify(args.root,previews=not args.no_previews)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)

