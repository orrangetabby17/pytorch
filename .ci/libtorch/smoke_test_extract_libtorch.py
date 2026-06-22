#!/usr/bin/env python3
"""Smoke tests for the end-to-end libtorch extraction pipeline.

Builds a minimal fake wheel, runs the full extraction (fix_rpath mocked on
Linux), and verifies the output zip contains the right files/structure.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from extract_libtorch_from_wheel import (
    compute_zip_prefix,
    copy_includes,
    copy_libraries,
    create_libtorch_zip,
    extract_wheel,
    find_wheel,
    get_git_hash,
    parse_version_from_wheel,
    write_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_wheel(wheel_dir: Path, version: str = "2.6.0") -> Path:
    """Create a minimal fake wheel zip with a torch/ directory tree."""
    name = f"torch-{version}-cp310-cp310-linux_x86_64.whl"
    wheel_path = wheel_dir / name
    with zipfile.ZipFile(wheel_path, "w") as zf:
        # Libraries: some should be included, some excluded
        zf.writestr("torch/lib/libtorch.so", "fake_so")
        zf.writestr("torch/lib/libtorch_cpu.so", "fake_so")
        zf.writestr("torch/lib/libc10.so", "fake_so")
        zf.writestr("torch/lib/libstatic.a", "fake_a")
        # Excluded: python binding and cpython extension
        zf.writestr("torch/lib/libtorch_python.so", "should_be_excluded")
        zf.writestr("torch/lib/_C.cpython-310-x86_64-linux-gnu.so", "should_be_excluded")
        # Include headers
        zf.writestr("torch/include/torch/torch.h", "// fake header")
        zf.writestr("torch/include/torch/csrc/api/include/torch/nn.h", "// fake")
        # CMake files
        zf.writestr("torch/share/cmake/Torch/TorchConfig.cmake", "# fake cmake")
        zf.writestr("torch/share/cmake/Torch/TorchConfigVersion.cmake", "# fake cmake")
        # version.py
        zf.writestr(
            "torch/version.py",
            "git_version = 'abc123def456'\n__version__ = '2.6.0'\n",
        )
    return wheel_path


def _run_extraction(
    wheel_dir: Path,
    output_dir: Path,
    platform: str,
    desired_cuda: str = "cu126",
    libtorch_variant: str = "shared-with-deps",
) -> Path:
    """Run the extraction pipeline and return the output zip path."""
    wheel_path = find_wheel(str(wheel_dir))
    version = parse_version_from_wheel(wheel_path)

    extract_dir = wheel_dir / "_extract_tmp"
    extract_dir.mkdir()
    torch_dir = extract_wheel(wheel_path, extract_dir)

    libtorch_dir = extract_dir / "libtorch"
    libtorch_dir.mkdir()
    for subdir in ["lib", "bin", "include", "share"]:
        (libtorch_dir / subdir).mkdir()

    copy_libraries(torch_dir, libtorch_dir / "lib", platform)
    # fix_rpath skipped: requires patchelf; tested separately in TestFixRpath
    copy_includes(torch_dir, libtorch_dir / "include")
    from extract_libtorch_from_wheel import copy_cmake, copy_bin
    copy_cmake(torch_dir, libtorch_dir / "share")
    copy_bin(torch_dir, libtorch_dir / "bin", platform)

    git_hash = get_git_hash(torch_dir)
    write_metadata(libtorch_dir, version, git_hash)

    zip_prefix = compute_zip_prefix(platform, desired_cuda, libtorch_variant)
    zip_path = create_libtorch_zip(libtorch_dir, output_dir, zip_prefix, version)
    shutil.rmtree(extract_dir)
    return zip_path


def _zip_names(zip_path: Path) -> set:
    with zipfile.ZipFile(zip_path) as zf:
        return set(zf.namelist())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindAndParse(unittest.TestCase):
    def test_find_wheel_success(self):
        with tempfile.TemporaryDirectory() as d:
            wheel_path = _make_fake_wheel(Path(d))
            found = find_wheel(d)
            self.assertEqual(found, wheel_path)

    def test_find_wheel_no_wheel_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                find_wheel(d)

    def test_find_wheel_multiple_raises(self):
        with tempfile.TemporaryDirectory() as d:
            _make_fake_wheel(Path(d), "2.6.0")
            _make_fake_wheel(Path(d), "2.7.0")
            with self.assertRaises(RuntimeError):
                find_wheel(d)

    def test_parse_version(self):
        with tempfile.TemporaryDirectory() as d:
            wheel_path = _make_fake_wheel(Path(d), "2.6.0+cu126")
            self.assertEqual(parse_version_from_wheel(wheel_path), "2.6.0+cu126")


class TestLibraryFiltering(unittest.TestCase):
    """Verify that copy_libraries includes/excludes the right files."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.torch_dir = Path(self._tmp) / "torch"
        (self.torch_dir / "lib").mkdir(parents=True)
        self.dest = Path(self._tmp) / "libtorch" / "lib"
        self.dest.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def _put(self, name: str, content: bytes = b"fake"):
        (self.torch_dir / "lib" / name).write_bytes(content)

    def test_includes_so_files(self):
        self._put("libtorch.so")
        self._put("libc10.so")
        copy_libraries(self.torch_dir, self.dest, "linux")
        self.assertTrue((self.dest / "libtorch.so").exists())
        self.assertTrue((self.dest / "libc10.so").exists())

    def test_includes_static_lib(self):
        self._put("libtorch.a")
        copy_libraries(self.torch_dir, self.dest, "linux")
        self.assertTrue((self.dest / "libtorch.a").exists())

    def test_excludes_libtorch_python(self):
        self._put("libtorch_python.so")
        copy_libraries(self.torch_dir, self.dest, "linux")
        self.assertFalse((self.dest / "libtorch_python.so").exists())

    def test_excludes_cpython_extension(self):
        self._put("_C.cpython-310-x86_64-linux-gnu.so")
        copy_libraries(self.torch_dir, self.dest, "linux")
        names = [f.name for f in self.dest.iterdir()]
        self.assertFalse(any("_C.cpython" in n for n in names))

    def test_excludes_py_files(self):
        self._put("torch_version.py")
        copy_libraries(self.torch_dir, self.dest, "linux")
        self.assertFalse((self.dest / "torch_version.py").exists())

    def test_macos_includes_dylib(self):
        self._put("libtorch.dylib")
        self._put("libtorch.so")  # should be excluded on macOS
        copy_libraries(self.torch_dir, self.dest, "macos")
        self.assertTrue((self.dest / "libtorch.dylib").exists())
        self.assertFalse((self.dest / "libtorch.so").exists())

    def test_windows_includes_dll_and_lib(self):
        self._put("torch.dll")
        self._put("torch.lib")
        self._put("torch.pdb")
        self._put("torch.so")  # excluded on windows
        copy_libraries(self.torch_dir, self.dest, "windows")
        self.assertTrue((self.dest / "torch.dll").exists())
        self.assertTrue((self.dest / "torch.lib").exists())
        self.assertTrue((self.dest / "torch.pdb").exists())
        self.assertFalse((self.dest / "torch.so").exists())


class TestEndToEndLinux(unittest.TestCase):
    def test_output_zip_structure(self):
        with tempfile.TemporaryDirectory() as d:
            tmpdir = Path(d)
            wheel_dir = tmpdir / "wheels"
            wheel_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            _make_fake_wheel(wheel_dir, "2.6.0")
            zip_path = _run_extraction(wheel_dir, output_dir, "linux")

            names = _zip_names(zip_path)

            # Libraries included
            self.assertIn("libtorch/lib/libtorch.so", names)
            self.assertIn("libtorch/lib/libtorch_cpu.so", names)
            self.assertIn("libtorch/lib/libc10.so", names)

            # Libraries excluded
            self.assertNotIn("libtorch/lib/libtorch_python.so", names)
            self.assertFalse(any("_C.cpython" in n for n in names))

            # Headers copied
            self.assertIn("libtorch/include/torch/torch.h", names)

            # CMake files copied
            self.assertTrue(
                any("TorchConfig.cmake" in n for n in names),
                f"No CMake files in zip; got: {sorted(names)}",
            )

            # Metadata
            self.assertIn("libtorch/build-version", names)
            self.assertIn("libtorch/build-hash", names)

    def test_metadata_content(self):
        with tempfile.TemporaryDirectory() as d:
            tmpdir = Path(d)
            wheel_dir = tmpdir / "wheels"
            wheel_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            _make_fake_wheel(wheel_dir, "2.6.0")
            zip_path = _run_extraction(wheel_dir, output_dir, "linux")

            with zipfile.ZipFile(zip_path) as zf:
                version = zf.read("libtorch/build-version").decode().strip()
                git_hash = zf.read("libtorch/build-hash").decode().strip()

            self.assertEqual(version, "2.6.0")
            self.assertEqual(git_hash, "abc123def456")

    def test_latest_symlink_created(self):
        with tempfile.TemporaryDirectory() as d:
            tmpdir = Path(d)
            wheel_dir = tmpdir / "wheels"
            wheel_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            _make_fake_wheel(wheel_dir, "2.6.0")
            _run_extraction(wheel_dir, output_dir, "linux")

            symlinks = list(output_dir.glob("*-latest.zip"))
            self.assertEqual(len(symlinks), 1)
            self.assertTrue(symlinks[0].is_symlink())

    def test_zip_prefix_linux(self):
        self.assertEqual(
            compute_zip_prefix("linux", "cu126", "shared-with-deps"),
            "libtorch-shared-with-deps",
        )

    def test_zip_prefix_macos(self):
        self.assertEqual(
            compute_zip_prefix("macos", "cpu", "shared-with-deps"),
            "libtorch-macos-arm64",
        )

    def test_zip_prefix_windows(self):
        self.assertEqual(
            compute_zip_prefix("windows", "cu126", "shared-with-deps"),
            "libtorch-win-shared-with-deps",
        )


class TestEndToEndMacOS(unittest.TestCase):
    def test_output_zip_excludes_so_includes_dylib(self):
        with tempfile.TemporaryDirectory() as d:
            tmpdir = Path(d)
            wheel_dir = tmpdir / "wheels"
            wheel_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            # Build a macOS-style fake wheel
            wheel_path = wheel_dir / "torch-2.6.0-cp310-cp310-macosx_11_0_arm64.whl"
            with zipfile.ZipFile(wheel_path, "w") as zf:
                zf.writestr("torch/lib/libtorch.dylib", "fake")
                zf.writestr("torch/lib/libc10.dylib", "fake")
                zf.writestr("torch/lib/libtorch_python.dylib", "excluded")
                zf.writestr("torch/include/torch/torch.h", "// header")
                zf.writestr("torch/share/cmake/Torch/TorchConfig.cmake", "# cmake")
                zf.writestr("torch/version.py", "git_version = 'deadbeef'\n")

            zip_path = _run_extraction(wheel_dir, output_dir, "macos")
            names = _zip_names(zip_path)

            self.assertIn("libtorch/lib/libtorch.dylib", names)
            self.assertNotIn("libtorch/lib/libtorch_python.dylib", names)
            self.assertIn("libtorch/include/torch/torch.h", names)


class TestGetGitHash(unittest.TestCase):
    def test_reads_git_version(self):
        with tempfile.TemporaryDirectory() as d:
            version_py = Path(d) / "version.py"
            version_py.write_text("git_version = 'deadbeef12345'\n__version__ = '2.6.0'\n")
            self.assertEqual(get_git_hash(Path(d)), "deadbeef12345")

    def test_returns_unknown_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(get_git_hash(Path(d)), "unknown")

    def test_returns_unknown_on_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            # literal_eval raises on an unquoted bare word (not a valid literal)
            (Path(d) / "version.py").write_text("git_version = not_a_literal\n")
            self.assertEqual(get_git_hash(Path(d)), "unknown")


if __name__ == "__main__":
    unittest.main()
