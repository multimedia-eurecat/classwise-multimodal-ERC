"""Run deterministic acceptance and regression checks without model downloads."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from .batch_resume import file_digest


class RecordingResult(unittest.TextTestResult):
    """Record each test, including failed subtests, for a machine-readable report."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = {}
        self.started = {}

    def startTest(self, test):
        super().startTest(test)
        self.started[test.id()] = time.perf_counter()
        self.records[test.id()] = {
            "name": test.id(),
            "status": "running",
            "details": [],
        }

    def stopTest(self, test):
        self.records[test.id()]["duration_seconds"] = round(
            time.perf_counter() - self.started[test.id()], 6
        )
        super().stopTest(test)

    def addSuccess(self, test):
        self.records[test.id()]["status"] = "passed"
        super().addSuccess(test)

    def _failure(self, test, status, err):
        self.records[test.id()]["status"] = status
        self.records[test.id()]["details"].append(self._exc_info_to_string(err, test))

    def addFailure(self, test, err):
        self._failure(test, "failed", err)
        super().addFailure(test, err)

    def addError(self, test, err):
        # Class/module setup errors can occur before startTest.
        self.records.setdefault(test.id(), {"name": test.id(), "details": []})
        self._failure(test, "error", err)
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._failure(test, "failed", err)
            self.records[test.id()]["details"].append(subtest.id())
        super().addSubTest(test, subtest, err)

    def addSkip(self, test, reason):
        self.records.setdefault(test.id(), {"name": test.id(), "details": []})
        self.records[test.id()]["status"] = "skipped"
        self.records[test.id()]["details"].append(reason)
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        self._failure(test, "expected_failure", err)
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.records[test.id()]["status"] = "unexpected_success"
        super().addUnexpectedSuccess(test)


def run_suite(suite, *, stream=None, verbosity=1) -> dict:
    start = time.perf_counter()
    result = unittest.TextTestRunner(
        stream=stream, verbosity=verbosity, resultclass=RecordingResult
    ).run(suite)
    records = list(result.records.values())
    return {
        "passed": result.testsRun > 0
        and result.wasSuccessful()
        and all(record["status"] == "passed" for record in records),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "duration_seconds": round(time.perf_counter() - start, 6),
        "tests": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write a JSON acceptance report")
    parser.add_argument("--verbose", action="store_true", help="Print each test name")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = root / "tests" / "fixtures" / "acceptance_cases.json"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_kind": "deterministic_contracts",
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "fixture_manifest_sha256": file_digest(manifest),
    }
    versions = {}
    for package in (
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "opencv-python-headless",
        "faster-whisper",
        "nemo-toolkit",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report["versions"] = versions
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        report.update(
            passed=False,
            setup_error=f"Missing required executables: {', '.join(missing)}",
        )
        exit_code = 2
    else:
        suite = unittest.defaultTestLoader.discover(str(root / "tests"))
        report.update(run_suite(suite, verbosity=2 if args.verbose else 1))
        exit_code = 0 if report["passed"] else 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(args.report)
    print(
        f"Acceptance {'PASSED' if report['passed'] else 'FAILED'}: "
        f"{report.get('tests_run', 0)} tests, {report.get('skipped', 0)} skipped"
    )
    if report.get("setup_error"):
        print(report["setup_error"], file=sys.stderr)
    if args.report:
        print(f"Report: {args.report.resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
