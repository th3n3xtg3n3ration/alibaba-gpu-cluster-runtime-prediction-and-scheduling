"""Guards on the test suite itself: every test must actually run.

Two defects in this directory made guard tests disappear without a single
failure to show for it, and both were invisible from inside the suite because a
test that is never collected reports nothing at all.

1. tests/test_checkpoint_currency.py was written as pytest module functions
   with a pytest fixture. The only test command this repository documents and
   runs is ``python -m unittest discover tests`` (README.md,
   scripts/run_all_experiments.sh), which collects ``unittest.TestCase``
   subclasses and nothing else, so all 18 tests protecting the
   checkpoint/model-artifact currency guard were absent from the project's own
   QA gate: both guards could be deleted from src/tuning.py and the gate still
   printed "All unit tests passed successfully!" over 80 green tests.

2. tests/test_regression_guards.py had its ``if __name__ == "__main__":``
   block in the middle of the file, above four more TestCase classes. Running
   that file directly collected only the 11 classes defined above it and
   printed OK, hiding the EASY-backfilling, GPU-less-provisioning,
   degenerate-predictor-refusal and utilization-integral guards.

Neither is a property of any one test, so neither can be guarded from inside
one. This module reads the suite as source and asserts both.
"""
import ast
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent


def _test_modules():
    """Every file `unittest discover` would try to collect from tests/."""
    return sorted(_TESTS_DIR.glob("test*.py"))


def _is_test_function(node):
    return (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test"))


def _base_names(class_node):
    names = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _authored_test_counts(path):
    """Tests a reader of this file would believe exist, split by how they are
    written: ``(inside TestCase classes, at module level, in pytest-style
    classes)``.

    Only the first kind is visible to the documented gate. The other two are
    counted separately so a failure can say which invisibility mode reappeared.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def _is_test_case(name, seen=()):
        # Resolved through locally-defined base classes too: the shared
        # fixtures in this suite (e.g. _ProvenanceCase) sit between the
        # concrete class and unittest.TestCase.
        if name in seen:
            return False
        if name == "TestCase":
            return True
        node = classes.get(name)
        if node is None:
            return False
        return any(_is_test_case(b, (*seen, name)) for b in _base_names(node))

    def _methods(name, seen=()):
        """Test methods a class exposes, including those it inherits from a
        base defined in this same file."""
        node, found = classes.get(name), set()
        if node is None or name in seen:
            return found
        found.update(m.name for m in node.body if _is_test_function(m))
        for base in _base_names(node):
            found.update(_methods(base, (*seen, name)))
        return found

    in_test_case = module_level = pytest_class = 0
    for node in tree.body:
        if _is_test_function(node):
            module_level += 1
        elif isinstance(node, ast.ClassDef):
            if _is_test_case(node.name):
                in_test_case += len(_methods(node.name))
            elif node.name.startswith("Test"):
                pytest_class += len(_methods(node.name))
    return in_test_case, module_level, pytest_class


class TestEveryAuthoredTestIsCollectedByTheDocumentedGate(unittest.TestCase):
    """``python -m unittest discover tests`` must run the whole suite.

    The gap is silent by construction: a test the loader never sees cannot
    fail, cannot be listed, and leaves the run's "OK" looking exactly as it
    does when the suite is complete.
    """

    def test_no_test_is_written_where_unittest_cannot_see_it(self):
        offenders = []
        for path in _test_modules():
            _, module_level, pytest_class = _authored_test_counts(path)
            if module_level or pytest_class:
                offenders.append(
                    f"{path.name}: {module_level} module-level test function(s), "
                    f"{pytest_class} test(s) in classes that do not subclass "
                    "unittest.TestCase"
                )
        self.assertEqual(
            offenders, [],
            "these tests are collected by pytest but not by the documented gate "
            "(python -m unittest discover tests), so they can pass forever "
            "without ever running:\n  " + "\n  ".join(offenders),
        )

    def test_the_loader_finds_every_test_the_files_define(self):
        """Counts, not just shapes: this catches the invisibility modes the
        check above does not enumerate (a decorator that replaces a test, a
        class the loader skips, a file whose name discovery does not match).
        """
        suite = unittest.TestLoader().discover(
            start_dir=str(_TESTS_DIR), top_level_dir=str(_REPO_ROOT)
        )

        def _flatten(s):
            for item in s:
                if isinstance(item, unittest.TestSuite):
                    yield from _flatten(item)
                else:
                    yield item

        collected = list(_flatten(suite))
        # A module that fails to import is reported as a single synthetic
        # failing test, which would otherwise show up here as a count mismatch
        # with a mystifying message.
        broken = [str(t) for t in collected if "_FailedTest" in type(t).__name__]
        self.assertEqual(broken, [], f"a test module failed to import: {broken}")

        authored = sum(_authored_test_counts(p)[0] for p in _test_modules())
        self.assertEqual(
            len(collected), authored,
            "the number of tests unittest discovers differs from the number "
            "written in tests/: some are defined in a way the documented gate "
            "does not collect",
        )


class TestSelfRunCollectsTheWholeFile(unittest.TestCase):
    """``python -m tests.test_x`` must run all of test_x, or none of it.

    ``unittest.main()`` collects the module as it stands at that moment, so any
    class defined after the call simply does not exist yet. The run still
    prints OK, with a lower test count that nothing compares against.
    """

    def test_the_main_block_is_the_last_thing_in_every_test_file(self):
        offenders = []
        for path in _test_modules():
            body = ast.parse(path.read_text(encoding="utf-8")).body
            for i, node in enumerate(body):
                if not isinstance(node, ast.If):
                    continue
                if "__name__" not in ast.dump(node.test):
                    continue
                trailing = [
                    n.name for n in body[i + 1:] if isinstance(n, ast.ClassDef)
                ]
                if trailing:
                    offenders.append(
                        f"{path.name}: unittest.main() at line {node.lineno} hides "
                        f"{trailing}"
                    )
        self.assertEqual(
            offenders, [],
            "a __main__ block before a TestCase class makes direct invocation "
            "of that file report OK without running those classes:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
