#!/usr/bin/env bash
# dehaze-python 质量门禁：ruff（lint + 格式）与 pyright（类型）必须全绿。
#
# 检查范围（哪些目录纳入、algorithm 为何排除）由仓库根 pyrightconfig.json 单点定义，
# ruff 范围由 pyproject.toml 的 [tool.ruff] 定义；两者与 IDE 显示口径一致，
# CI 判定只认本脚本，不认编辑器设置。
#
# 用法：dehaze-python/scripts/check.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/ruff ]]; then
    echo "缺少 .venv/bin/ruff：请先在 dehaze-python 下执行 uv sync" >&2
    exit 1
fi

# pyright 版本必须固定：不同版本的 typeshed 不同，会得出不同结论
# （实例：asyncio.getaddrinfo 的 sockaddr 联合类型在 1.1.414 才补上 tuple[int, bytes]，
# 旧 CLI 因此漏检 app/utils/ssrf.py，而 IDE 报错）。此处与 IDE 的 ms-pyright 扩展对齐，
# 升级扩展时同步改这里；CI 亦可用环境变量显式指定。
PYRIGHT_VERSION="${PYRIGHT_VERSION:-1.1.414}"
export PYRIGHT_PYTHON_FORCE_VERSION="$PYRIGHT_VERSION"

# pyright 优先用项目环境内的可执行文件；未安装时回退到 PATH
if [[ -x .venv/bin/pyright ]]; then
    PYRIGHT=.venv/bin/pyright
elif command -v pyright >/dev/null 2>&1; then
    PYRIGHT=pyright
else
    echo "缺少 pyright：请安装（如 uv add --dev pyright）后重试" >&2
    exit 1
fi

echo "==> ruff check"
.venv/bin/ruff check .

echo "==> ruff format --check"
.venv/bin/ruff format --check .

echo "==> pyright $PYRIGHT_VERSION（范围见仓库根 pyrightconfig.json）"
"$PYRIGHT"

echo "==> 全部通过"
