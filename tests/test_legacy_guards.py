"""Safe regression tests; never run legacy experiments or downloads."""
import ast
import importlib.util
import io
import os
from contextlib import redirect_stderr
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PortableScratchTests(unittest.TestCase):
    def scratch_root(self):
        self.assertIsNotNone(
            importlib.util.find_spec("tests.support"),
            "tests need a shared portable scratch helper, not ambient TMPDIR",
        )
        from tests.support import scratch_root
        return scratch_root

    def test_explicit_tmpdir_is_created_and_preferred(self):
        scratch_root = self.scratch_root()
        base = scratch_root()
        with tempfile.TemporaryDirectory(dir=base) as temp:
            target = Path(temp) / "explicit" / "nested"
            with patch.dict(os.environ, {"TMPDIR": str(target)}):
                self.assertEqual(scratch_root(), target)
                self.assertTrue(target.is_dir())

    def test_windows_fallback_is_hermes_scratch_not_system_temp(self):
        scratch_root = self.scratch_root()
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            local = Path(temp) / "local"
            with patch.dict(os.environ, {"LOCALAPPDATA": str(local)}, clear=True), \
                    patch("tests.support.sys.platform", "win32"):
                expected = local / "hermes" / "cache" / "scratch"
                self.assertEqual(scratch_root(), expected)
                self.assertTrue(expected.is_dir())

    def test_windows_without_localappdata_uses_home_appdata(self):
        scratch_root = self.scratch_root()
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            home = Path(temp)
            with patch.dict(os.environ, {}, clear=True), \
                    patch("tests.support.sys.platform", "win32"), \
                    patch("tests.support.Path.home", return_value=home):
                expected = home / "AppData" / "Local" / "hermes" / "cache" / "scratch"
                self.assertEqual(scratch_root(), expected)
                self.assertTrue(expected.is_dir())

    def test_posix_fallback_is_home_hermes_scratch(self):
        scratch_root = self.scratch_root()
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            home = Path(temp)
            with patch.dict(os.environ, {"TMPDIR": ""}, clear=True), \
                    patch("tests.support.sys.platform", "linux"), \
                    patch("tests.support.Path.home", return_value=home):
                expected = home / ".hermes" / "cache" / "scratch"
                self.assertEqual(scratch_root(), expected)
                self.assertTrue(expected.is_dir())

    def test_unusable_explicit_tmpdir_is_not_silently_bypassed(self):
        scratch_root = self.scratch_root()
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            occupied = Path(temp) / "not-a-directory"
            occupied.write_text("occupied", encoding="utf-8")
            with patch.dict(os.environ, {"TMPDIR": str(occupied)}):
                with self.assertRaises(FileExistsError):
                    scratch_root()


class LegacyOptInTests(unittest.TestCase):
    def guard(self):
        self.assertIsNotNone(
            importlib.util.find_spec("src.legacy_guard"),
            "legacy CLIs need a shared explicit opt-in guard",
        )
        from src.legacy_guard import require_legacy_opt_in
        return require_legacy_opt_in

    def assert_warning(self, message):
        self.assertIn("historical/unaccepted", message)
        self.assertIn("canonical pipeline is src/reproduce.py", message)
        self.assertIn("--allow-legacy", message)

    def test_missing_opt_in_warns_and_exits_two(self):
        guard = self.guard()
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["legacy.py"]), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as stopped:
                guard()
        self.assertEqual(stopped.exception.code, 2)
        self.assert_warning(stderr.getvalue())

    def test_explicit_opt_in_warns_and_consumes_only_the_guard_flag(self):
        guard = self.guard()
        argv = ["legacy.py", "--out", "fixture", "--allow-legacy", "--help"]
        stderr = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stderr(stderr):
            guard()
        self.assertEqual(argv, ["legacy.py", "--out", "fixture", "--help"])
        self.assert_warning(stderr.getvalue())
        self.assertIn("overwrite", stderr.getvalue().lower())

    def test_similar_flags_do_not_enable_legacy(self):
        guard = self.guard()
        for flag in ("--allow-legacy=true", "--allow-leg", "--no-allow-legacy"):
            with self.subTest(flag=flag), patch.object(sys, "argv", ["legacy.py", flag]), \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    guard()
                self.assertEqual(stopped.exception.code, 2)

    def test_program_name_does_not_count_as_opt_in(self):
        guard = self.guard()
        with patch.object(sys, "argv", ["--allow-legacy"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                guard()
        self.assertEqual(stopped.exception.code, 2)

    def test_flag_after_option_terminator_does_not_count_as_opt_in(self):
        guard = self.guard()
        with patch.object(sys, "argv", ["legacy.py", "--", "--allow-legacy"]), \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                guard()
        self.assertEqual(stopped.exception.code, 2)

    def test_duplicate_opt_in_flags_are_consumed_but_literal_arguments_are_not(self):
        guard = self.guard()
        argv = ["legacy.py", "--allow-legacy", "--allow-legacy", "--", "--allow-legacy"]
        with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
            guard()
        self.assertEqual(argv, ["legacy.py", "--", "--allow-legacy"])


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPTS = (
    "fetch_raw.py", "prep_data.py", "src/data_pipeline.py", "src/experiment.py",
    "src/finalize_paper.py", "src/generate_figures.py", "src/update_paper_results.py",
    "src/summarize_results.py",
)


class LegacyCliTests(unittest.TestCase):
    def check_blocked_cli(self, relative, mode):
        from tests.support import scratch_root
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            sandbox = Path(temp) / "project"
            elsewhere = Path(temp) / "unrelated-cwd"
            elsewhere.mkdir()
            # A cwd/PYTHONPATH shadow must never replace the local guard.
            (elsewhere / "legacy_guard.py").write_text(
                "raise RuntimeError('wrong guard imported from cwd')\n", encoding="utf-8"
            )
            for name in (*LEGACY_SCRIPTS, "src/legacy_guard.py"):
                target = sandbox / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            before = {p.relative_to(sandbox): p.read_bytes()
                      for p in sandbox.rglob("*") if p.is_file()}
            module = relative.removesuffix(".py").replace("/", ".")
            invocation = [str(sandbox / relative)] if mode == "file" else ["-m", module]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join((str(elsewhere), str(sandbox)))
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            # -S denies scientific dependencies; --help keeps the historical
            # downloader harmless during RED. No results/data are copied.
            result = subprocess.run(
                [sys.executable, "-B", "-S", *invocation, "--help"],
                cwd=elsewhere, env=env, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertIn("historical/unaccepted", result.stderr)
            self.assertIn("canonical pipeline is src/reproduce.py", result.stderr)
            self.assertIn("--allow-legacy", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            after = {p.relative_to(sandbox): p.read_bytes()
                     for p in sandbox.rglob("*") if p.is_file()}
            self.assertEqual(after, before, "blocked CLI must not create or replace outputs")

    def check_prefix_opt_in_and_import(self, relative):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        prefix = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                prefix.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
                prefix.append(node)
            elif isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
                prefix.append(node)
                break
            else:
                self.fail("opt-in must run before legacy imports or main work")
        self.assertIsInstance(prefix[-1], ast.If)
        code = compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec")
        # Only the safe leading guard is executed, never historical functions.
        for module_name in ("__main__", "library_import"):
            stderr = io.StringIO()
            argv = [str(path), "--allow-legacy", "--out", "fixture"]
            with self.subTest(module_name=module_name), patch.object(sys, "argv", argv), \
                    patch.object(sys, "path", sys.path.copy()), redirect_stderr(stderr):
                exec(code, {"__file__": str(path), "__name__": module_name})
                if module_name == "__main__":
                    self.assertEqual(argv, [str(path), "--out", "fixture"])
                    self.assertIn("historical/unaccepted", stderr.getvalue())
                else:
                    self.assertEqual(argv, [str(path), "--allow-legacy", "--out", "fixture"])
                    self.assertEqual(stderr.getvalue(), "")


def _blocked_cli_test(relative, mode):
    def test(self):
        self.check_blocked_cli(relative, mode)
    return test


def _prefix_test(relative):
    def test(self):
        self.check_prefix_opt_in_and_import(relative)
    return test


for _relative in LEGACY_SCRIPTS:
    _name = _relative.removesuffix(".py").replace("/", "_")
    for _mode in ("file", "module"):
        setattr(LegacyCliTests, f"test_{_name}_{_mode}_requires_opt_in",
                _blocked_cli_test(_relative, _mode))
    setattr(LegacyCliTests, f"test_{_name}_safe_opt_in_prefix_and_import",
            _prefix_test(_relative))


if __name__ == "__main__":
    unittest.main()
