"""The paper verifier is a real fail-closed gate, not an import-time script."""
import unittest,subprocess,sys,tempfile,shutil
from pathlib import Path
from tests.support import scratch_root


class PaperGateTests(unittest.TestCase):
    def test_import_has_no_artifact_reads_or_writes(self):
        source=Path(__file__).resolve().parents[1]/'src/verify_paper.py'
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            dest=Path(folder)/'src/verify_paper.py';dest.parent.mkdir();shutil.copyfile(source,dest)
            code="import importlib.util; s=importlib.util.spec_from_file_location('probe',r'"+str(dest)+"'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"
            result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual({p.relative_to(folder).as_posix() for p in Path(folder).rglob('*') if p.is_file()},{'src/verify_paper.py'})

    def test_failed_crossrefs_produce_failed_acceptance(self):
        from src import verify_paper as module
        self.assertTrue(hasattr(module,'verify'),'Missing callable fail-closed PDF verifier')
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=scratch_root()) as folder:
            work=Path(folder);(work/'paper/generated').mkdir(parents=True)
            for name in ['paper/MRP_Draft.pdf','paper/MRP_Draft.qmd','references.bib']:
                shutil.copyfile(root/name,work/name)
            q=work/'paper/MRP_Draft.qmd'
            q.write_text(q.read_text(encoding='utf-8')+'\n@nonexistentReference @tbl-never-present\n',encoding='utf-8')
            report=module.verify(work,write_outputs=False,previews=False)
            self.assertFalse(report['passed'])
            self.assertIn('nonexistentReference',report['unknown_citations'])
            self.assertIn('tbl-never-present',report['unknown_crossrefs'])


if __name__=='__main__':unittest.main()
