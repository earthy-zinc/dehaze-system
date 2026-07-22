"""CUDA 扩展编译环境自动配置

在 app 包导入时执行（app/__init__.py 顶部导入本模块），
确保 `uvicorn app.main:app` 启动时在任何 algorithm 模块 import 之前
完成 MSVC / ninja / 编译缓存目录等环境配置。

自动处理：
- MSVC 编译器环境（cl.exe / INCLUDE / LIB）
- TORCH_EXTENSIONS_DIR（编译缓存目录）
- NVCC_PREPEND_FLAGS（允许 nvcc 使用不被官方支持的 MSVC 版本）
- ninja.exe（缺失时自动下载）
"""
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_msvc_env():
    """自动搜索并加载 MSVC 编译器环境（cl.exe / INCLUDE / LIB）

    优先选择 VS 2019（MSVC 14.29 兼容 CUDA 12.1），
    其次 VS 2022（MSVC 14.4x 需 CUDA 12.4+）。
    """
    if sys.platform != "win32":
        return

    from shutil import which
    if which("cl"):
        return

    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        return

    try:
        result = subprocess.run(
            [str(vswhere), "-all", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-sort", "-property", "installationPath"],
            capture_output=True, text=True, timeout=10,
        )
        vs_paths = [p.strip() for p in result.stdout.splitlines() if p.strip()]
    except Exception:
        return

    if not vs_paths:
        return

    # 优先选择 VS 2019（路径含 2019），兼容 CUDA 12.1
    vscmd_paths = []
    for vs_path in vs_paths:
        vcvars = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if vcvars.exists():
            priority = 0 if "2019" in vs_path else 1
            vscmd_paths.append((priority, vcvars))
    vscmd_paths.sort(key=lambda x: x[0])

    if not vscmd_paths:
        return

    vcvars = vscmd_paths[0][1]

    try:
        result = subprocess.run(
            f'"{vcvars}" && set',
            shell=True, capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return

    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ[key] = value

    print(f"[bootstrap] MSVC environment loaded: {vcvars.parent.parent.parent.parent}")


def _setup_torch_extensions_dir():
    """将 CUDA 扩展 JIT 编译缓存目录设为项目内，避免写入系统目录权限问题"""
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_PROJECT_ROOT / ".torch_extensions"))


def _setup_nvcc_flags():
    """允许 nvcc 使用不被官方支持的 MSVC 版本"""
    if sys.platform == "win32":
        os.environ.setdefault("NVCC_PREPEND_FLAGS", "-allow-unsupported-compiler")


def _ensure_ninja():
    """确保 ninja.exe 在 PATH 中（CUDA 扩展 JIT 编译必需）

    uv/pip 安装的 ninja Python 包不附带 ninja.exe 二进制，
    若 PATH 和 venv\\Scripts 中均未找到则自动下载。
    """
    if sys.platform != "win32":
        return

    from shutil import which
    if which("ninja"):
        return

    scripts_dir = _PROJECT_ROOT / ".venv" / "Scripts"
    ninja_exe = scripts_dir / "ninja.exe"

    if ninja_exe.exists():
        os.environ["PATH"] = f"{scripts_dir};{os.environ['PATH']}"
        return

    print("[bootstrap] ninja not found, downloading...")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    import tempfile

    zip_path = Path(tempfile.gettempdir()) / "ninja-win.zip"
    url = "https://github.com/ninja-build/ninja/releases/latest/download/ninja-win.zip"

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        proxy_handler = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(scripts_dir)

    zip_path.unlink(missing_ok=True)
    os.environ["PATH"] = f"{scripts_dir};{os.environ['PATH']}"
    print(f"[bootstrap] ninja installed to {ninja_exe}")


def setup_environment():
    """启动前自动配置 CUDA 扩展编译环境"""
    if sys.platform != "win32":
        _setup_torch_extensions_dir()
        return

    _load_msvc_env()
    _setup_torch_extensions_dir()
    _setup_nvcc_flags()
    _ensure_ninja()


# 模块导入时立即执行，确保在 app 子模块 import 之前完成配置
setup_environment()
