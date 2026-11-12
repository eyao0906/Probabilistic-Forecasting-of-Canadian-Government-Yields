"""Non-executing regression for M3 smoke tests' writable output paths."""
import ast
from pathlib import Path
import unittest


class M3FixtureIsolationTests(unittest.TestCase):
    def test_writing_smoke_tests_only_use_isolated_workspace(self):
        # Inspect source, never execute the unsafe historical test bodies for RED.
        source = Path(__file__).with_name('test_state_space_revised.py')
        tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
        names = {
            'test_end_to_end_smoke_saves_moments_keys_targets_and_fit_diagnostics',
            'test_independent_saved_archive_verifier_checks_real_smoke',
        }
        methods = {node.name: node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name in names}
        self.assertEqual(set(methods), names)
        for name, method in methods.items():
            with self.subTest(smoke_test=name):
                scopes = [node for node in ast.walk(method) if isinstance(node, ast.With)
                          and any(isinstance(item.context_expr, ast.Call)
                                  and isinstance(item.context_expr.func, ast.Name)
                                  and item.context_expr.func.id == 'isolated_m3_workspace'
                                  for item in node.items)]
                calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                         and ((isinstance(node.func, ast.Name) and node.func.id == 'run_experiment')
                              or (isinstance(node.func, ast.Attribute)
                                  and node.func.attr == 'run_experiment'))]
                self.assertEqual(len(calls), 1, 'Smoke coverage must still execute the real runner')
                self.assertTrue(scopes, 'Smoke output must be scoped to isolated_m3_workspace(), '
                                'not canonical results/state_space_revised fixture directories')
                self.assertTrue(any(calls[0] in list(ast.walk(scope)) for scope in scopes),
                                'The runner must execute before temporary workspace cleanup')
                output = next(keyword.value for keyword in calls[0].keywords
                              if keyword.arg == 'output_dir')
                self.assertIsInstance(output, ast.Name)
                assert isinstance(output, ast.Name)
                self.assertTrue(any(
                    isinstance(item.optional_vars, ast.Tuple)
                    and any(isinstance(binding, ast.Name) and binding.id == output.id
                            for binding in item.optional_vars.elts)
                    for scope in scopes for item in scope.items),
                    'Output must be bound by the isolated workspace context')
                for node in ast.walk(method):
                    if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == output.id
                            for target in node.targets):
                        self.fail('Output must come from the temporary workspace binding, '
                                  f'not a separately assigned path: {ast.unparse(node.value)}')


if __name__ == '__main__':
    unittest.main()
