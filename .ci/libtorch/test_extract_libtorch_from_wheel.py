#!/usr/bin/env python3
"""Tests for extract_libtorch_from_wheel.py, focusing on fix_rpath."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extract_libtorch_from_wheel import fix_rpath


class TestFixRpath(unittest.TestCase):
    def test_raises_on_non_linux(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.so").write_bytes(b"fake")
            with patch("sys.platform", "darwin"):
                with self.assertRaises(RuntimeError):
                    fix_rpath(Path(d))

    def test_calls_patchelf_on_so_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libtorch.so").write_bytes(b"fake")
            Path(d, "readme.txt").write_text("not a library")

            with (
                patch("sys.platform", "linux"),
                patch(
                    "extract_libtorch_from_wheel.shutil.which",
                    return_value="/usr/local/bin/patchelf",
                ),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0
                )
                fix_rpath(Path(d))

                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                self.assertEqual(args[0], "/usr/local/bin/patchelf")
                self.assertEqual(args[1], "--set-rpath")
                self.assertEqual(args[2], "$ORIGIN")
                self.assertEqual(args[3], "--force-rpath")
                self.assertIn("libtorch.so", args[4])

    def test_calls_patchelf_on_versioned_so(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.so.1").write_bytes(b"fake")

            with (
                patch("sys.platform", "linux"),
                patch(
                    "extract_libtorch_from_wheel.shutil.which",
                    return_value="/usr/local/bin/patchelf",
                ),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0
                )
                fix_rpath(Path(d))
                mock_run.assert_called_once()

    def test_skips_non_so_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.a").write_bytes(b"fake")

            with (
                patch("sys.platform", "linux"),
                patch(
                    "extract_libtorch_from_wheel.shutil.which",
                    return_value="/usr/local/bin/patchelf",
                ),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                fix_rpath(Path(d))
                mock_run.assert_not_called()

    def test_raises_if_patchelf_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.so").write_bytes(b"fake")

            with (
                patch("sys.platform", "linux"),
                patch("extract_libtorch_from_wheel.shutil.which", return_value=None),
            ):
                with self.assertRaises(FileNotFoundError):
                    fix_rpath(Path(d))

    def test_raises_on_patchelf_failure(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.so").write_bytes(b"fake")

            with (
                patch("sys.platform", "linux"),
                patch(
                    "extract_libtorch_from_wheel.shutil.which",
                    return_value="/usr/local/bin/patchelf",
                ),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=1, stderr="cannot find section"
                )
                with self.assertRaises(RuntimeError):
                    fix_rpath(Path(d))

    @unittest.skipUnless(sys.platform == "linux", "patchelf only supported on Linux")
    @unittest.skipUnless(shutil.which("patchelf"), "patchelf not installed")
    def test_integration_with_real_patchelf(self):
        with tempfile.TemporaryDirectory() as d:
            so = Path(d) / "libtest.so"
            gcc = shutil.which("gcc") or shutil.which("cc")
            if not gcc:
                self.skipTest("no C compiler available")

            src = Path(d) / "test.c"
            src.write_text("void foo(void) {}")
            result = subprocess.run(
                [gcc, "-shared", "-o", str(so), str(src)],
                capture_output=True,
            )
            if result.returncode != 0:
                self.skipTest("failed to compile test shared library")

            fix_rpath(Path(d))

            result = subprocess.run(
                ["patchelf", "--print-rpath", str(so)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), "$ORIGIN")


if __name__ == "__main__":
    unittest.main()
