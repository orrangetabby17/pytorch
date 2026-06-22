#!/usr/bin/env python3
"""Smoke tests for the libtorch extraction pipeline."""

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from extract_libtorch_from_wheel import (
    compute_zip_prefix,
    copy_bin,
    copy_cmake,
    copy_includes,
    copy_libraries,
    create_libtorch_zip,
    extract_wheel,
    find_wheel,
    get_git_hash,
    parse_version_from_wheel,
    write_metadata,
)


def _make_fake_wheel(wheel_dir: Path, version: str = "2.6.0") -> Path:
    name = f"torch-{version}-cp310-cp310-linux_x86_64.whl"
    wheel_path = wheel_dir / name
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("torch/lib/libtorch.so", "fake")
        zf.writestr("torch/lib/libtorch_cpu.so", "fake")
        zf.writestr("torch/lib/libtorch_python.so", "excluded")
        zf.writestr("torch/lib/_C.cpython-310-x86_64-linux-gnu.so", "excluded")
        zf.writestr("torch/include/torch/torch.h", "// header")
        zf.writestr("torch/share/cmake/Torch/TorchConfig.cmake", "# cmake")
        zf.writestr("torch/version.py", "git_version = 'abc123'\n")
    return wheel_path


class TestSmokeExtraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.wheel_dir = Path(self._tmp) / "wheels"
        self.wheel_dir.mkdir()
        self.output_dir = Path(self._tmp) / "output"
        self.output_dir.mkdir()
        _make_fake_wheel(self.wheel_dir)

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def _extract(self, platform="linux"):
        wheel_path = find_wheel(str(self.wheel_dir))
        version = parse_version_from_wheel(wheel_path)
        extract_dir = self.wheel_dir / "_tmp"
        extract_dir.mkdir()
        torch_dir = extract_wheel(wheel_path, extract_dir)

        libtorch_dir = extract_dir / "libtorch"
        libtorch_dir.mkdir()
        for sub in ["lib", "bin", "include", "share"]:
            (libtorch_dir / sub).mkdir()

        copy_libraries(torch_dir, libtorch_dir / "lib", platform)
        copy_includes(torch_dir, libtorch_dir / "include")
        copy_cmake(torch_dir, libtorch_dir / "share")
        copy_bin(torch_dir, libtorch_dir / "bin", platform)
        write_metadata(libtorch_dir, version, get_git_hash(torch_dir))

        prefix = compute_zip_prefix(platform, "cu126", "shared-with-deps")
        return create_libtorch_zip(libtorch_dir, self.output_dir, prefix, version)

    def test_output_zip_has_expected_files(self):
        zip_path = self._extract()
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        self.assertIn("libtorch/lib/libtorch.so", names)
        self.assertIn("libtorch/include/torch/torch.h", names)
        self.assertTrue(any("TorchConfig.cmake" in n for n in names))
        self.assertIn("libtorch/build-version", names)
        self.assertIn("libtorch/build-hash", names)

    def test_excluded_libs_not_in_zip(self):
        zip_path = self._extract()
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        self.assertNotIn("libtorch/lib/libtorch_python.so", names)
        self.assertFalse(any("_C.cpython" in n for n in names))

    def test_metadata_content(self):
        zip_path = self._extract()
        with zipfile.ZipFile(zip_path) as zf:
            self.assertEqual(zf.read("libtorch/build-version").decode().strip(), "2.6.0")
            self.assertEqual(zf.read("libtorch/build-hash").decode().strip(), "abc123")

    def test_latest_symlink_created(self):
        self._extract()
        symlinks = list(self.output_dir.glob("*-latest.zip"))
        self.assertEqual(len(symlinks), 1)
        self.assertTrue(symlinks[0].is_symlink())

    def test_zip_prefix_naming(self):
        self.assertEqual(compute_zip_prefix("linux", "cu126", "shared-with-deps"), "libtorch-shared-with-deps")
        self.assertEqual(compute_zip_prefix("macos", "cpu", "shared-with-deps"), "libtorch-macos-arm64")
        self.assertEqual(compute_zip_prefix("windows", "cu126", "shared-with-deps"), "libtorch-win-shared-with-deps")

    def test_find_wheel_errors(self):
        empty = Path(self._tmp) / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            find_wheel(str(empty))
        _make_fake_wheel(self.wheel_dir, "2.7.0")  # now two wheels
        with self.assertRaises(RuntimeError):
            find_wheel(str(self.wheel_dir))


if __name__ == "__main__":
    unittest.main()
