import os
import re
import shutil
import subprocess
import sys
import sysconfig
import warnings

from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import torch
from torch.utils.cpp_extension import include_paths, library_paths


REPO_ROOT = Path(__file__).resolve().parent
THUNDERKITTENS_ROOT = REPO_ROOT / "third_party" / "ThunderKittens"
EXTENSION_NAME = "mok._C"


def parse_major_minor_versions(version: str, name: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"Unable to parse {name} version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def check_linux() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError(f"The setup cannot run on a non-Linux platform; found {sys.platform!r}.")


def check_make() -> None:
    makefile = REPO_ROOT / "Makefile"
    if not makefile.is_file():
        raise RuntimeError(f"Makefile not found: {makefile}")

    make = shutil.which("make")
    if make is None:
        raise RuntimeError("GNU Make is required, but 'make' was not found on PATH.")


def check_python() -> None:
    if sys.version_info < (3, 12):
        raise RuntimeError(f"Python >=3.12 is required; found {sys.version.split()[0]}.")

    include_dir = sysconfig.get_path("include")
    if include_dir is None or not (Path(include_dir) / "Python.h").is_file():
        raise RuntimeError("Python development headers are required, but Python.h was not found.")


def check_pytorch() -> None:
    version = parse_major_minor_versions(str(torch.__version__), "PyTorch")
    if not (2, 10) <= version < (3, 0):
        raise RuntimeError(f"PyTorch >=2.10.0,<3 is required; found PyTorch {torch.__version__}.")

    if torch.version.cuda is None:
        raise RuntimeError("A CUDA-enabled PyTorch installation is required.")

    cuda_version = parse_major_minor_versions(torch.version.cuda, "PyTorch CUDA")
    if cuda_version < (13, 0):
        raise RuntimeError(f"PyTorch built for CUDA >=13.0 is required; found CUDA {torch.version.cuda}.")
    if torch.version.cuda != "13.0":
        warnings.warn(f"MoK should run with PyTorch CUDA {torch.version.cuda}, but it is built and tested for CUDA 13.0.", RuntimeWarning, stacklevel=2)


def get_nvcc_name() -> str:
    return os.environ.get("MOK_NVCC", os.environ.get("NVCC", "nvcc"))


def check_nvcc() -> str:
    nvcc_name = get_nvcc_name()
    nvcc = shutil.which(nvcc_name)
    if nvcc is None:
        raise RuntimeError(
            f"NVCC is required, but {nvcc_name!r} was not found. "
            "Install the CUDA toolkit or ensure PATH does point to its bin directory."
        )

    output = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if match is None:
        raise RuntimeError(f"Unable to parse the NVCC version:\n{output.strip()}")
    nvcc_version = int(match.group(1)), int(match.group(2))

    if nvcc_version[0] != 13:
        raise RuntimeError(
            f"CUDA 13.x NVCC is required; found {nvcc_version[0]}.{nvcc_version[1]}. "
            "Install the correct CUDA toolkit or ensure PATH points to its bin directory."
        )

    torch_cuda_version = parse_major_minor_versions(str(torch.version.cuda), "PyTorch CUDA")
    if nvcc_version != torch_cuda_version:
        raise RuntimeError(
            f"NVCC {nvcc_version[0]}.{nvcc_version[1]} does not match PyTorch CUDA {torch.version.cuda}. "
            "Install the matching CUDA toolkit or ensure PATH points to its bin directory."
        )

    return nvcc


def check_thunderkittens() -> None:
    thunderkittens_header = THUNDERKITTENS_ROOT / "include" / "kittens.cuh"
    if thunderkittens_header.is_file():
        return

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("ThunderKittens is missing and Git was not found.")
    if not (REPO_ROOT / ".git").exists():
        raise RuntimeError("ThunderKittens headers are missing from this source distribution.")

    subprocess.check_call([
        git, "submodule", "update", "--init", "--recursive", "--", "third_party/ThunderKittens"
    ], cwd=REPO_ROOT)

    if not thunderkittens_header.is_file():
        raise RuntimeError("ThunderKittens is unavailable after initialization.")


class MoKBuildExtension(build_ext):
    def build_extension(self, ext) -> None:
        check_linux()
        check_make()
        check_python()
        check_pytorch()
        nvcc = check_nvcc()
        check_thunderkittens()

        pytorch_includes = " ".join(f"-I{path}" for path in include_paths())
        pytorch_libdir = " ".join(f"-L{path}" for path in library_paths())
        output_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        subprocess.check_call([
            "make",
            f"SRC={REPO_ROOT / 'csrc' / 'bindings.cu'}",
            f"NVCC={nvcc}",
            f"ARCH={os.environ.get('MOK_ARCH', 'SM103')}",
            f"PYTHON={sys.executable}",
            f"THUNDERKITTENS_ROOT={THUNDERKITTENS_ROOT}",
            f"OUT={output_path}",
            f"PYTHON_INCLUDES=-I{sysconfig.get_path('include')}",
            f"PYTORCH_INCLUDES={pytorch_includes}",
            f"PYTORCH_LIBDIR={pytorch_libdir}",
        ], cwd=REPO_ROOT)


setup(
    ext_modules=[Extension(EXTENSION_NAME, sources=[])],
    cmdclass={"build_ext": MoKBuildExtension},
)
