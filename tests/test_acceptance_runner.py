"""The acceptance gate must not silently pass skipped or partially failed cases."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from aimara.run_acceptance import main, run_suite


class AcceptanceRunnerTests(unittest.TestCase):
    def test_missing_media_tools_write_a_failed_setup_report_and_exit_two(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with (
                patch("sys.argv", ["run_acceptance", "--report", str(path)]),
                patch("aimara.run_acceptance.shutil.which", return_value=None),
                patch(
                    "aimara.run_acceptance.unittest.defaultTestLoader.discover"
                ) as discover,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(), 2)
            report = json.loads(path.read_text())
        discover.assert_not_called()
        self.assertFalse(report["passed"])
        self.assertIn("ffmpeg", report["setup_error"])
        self.assertIn("ffprobe", report["setup_error"])

    def test_class_setup_errors_are_reported_even_without_started_tests(self):
        class Case(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("fixture could not be created")

            def test_contract(self):
                pass

        report = run_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(Case), stream=io.StringIO()
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["tests"][0]["status"], "error")
        self.assertIn("fixture could not be created", report["tests"][0]["details"][0])

    def test_success_has_a_named_timed_record(self):
        class Case(unittest.TestCase):
            def test_contract(self):
                self.assertTrue(True)

        report = run_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(Case), stream=io.StringIO()
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["tests_run"], 1)
        self.assertEqual(report["tests"][0]["status"], "passed")
        self.assertIn("test_contract", report["tests"][0]["name"])
        self.assertGreaterEqual(report["tests"][0]["duration_seconds"], 0)

    def test_skipped_and_empty_suites_are_incomplete(self):
        class Case(unittest.TestCase):
            @unittest.skip("required native library unavailable")
            def test_contract(self):
                pass

        report = run_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(Case), stream=io.StringIO()
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["skipped"], 1)
        self.assertIn("unavailable", report["tests"][0]["details"][0])
        self.assertFalse(
            run_suite(unittest.TestSuite(), stream=io.StringIO())["passed"]
        )

    def test_failed_subtest_marks_its_parent_failed(self):
        class Case(unittest.TestCase):
            def test_contract(self):
                for value in (1, 2):
                    with self.subTest(value=value):
                        self.assertEqual(value, 1)

        report = run_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(Case), stream=io.StringIO()
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["tests"][0]["status"], "failed")
        self.assertTrue(
            any("value=2" in detail for detail in report["tests"][0]["details"])
        )
