#!/usr/bin/env bash
# dehaze-python 质量门禁：ruff（lint + 格式）+ pyright（类型）必须全绿。
#
# ruff 规则取各项目自身的 pyproject.toml [tool.ruff]（三个推理服务无独立 venv，
# 统一用编排侧的 ruff 可执行文件，进入各自目录执行）；
# pyright 按 Python 子项目逐个执行，各项目自持 pyrightconfig.json（include / venv /
# extraPaths 各归各），共享口径（typeCheckingMode / pythonPlatform / pythonVersion）单点定义在仓库根
# pyright.base.json。仓库根 pyrightconfig.json 只服务"以仓库为单根打开 IDE"的场景，
# 不参与本脚本；CI 判定只认本脚本，不认编辑器设置。
#
# 用法：dehaze-python/scripts/check.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(dirname "$PROJECT_DIR")"
RUFF="$PROJECT_DIR/.venv/bin/ruff"

# 纳入 lint 检查的推理服务
RUFF_SERVICES=(dehaze-llm dehaze-voice dehaze-modal)

# 纳入类型检查的 Python 子项目：新增子项目时在此登记，并为其提供 pyrightconfig.json
PYRIGHT_PROJECTS=(
    dehaze-python
    dehaze-mcp-gateway
    dehaze-service/dehaze-llm
    dehaze-service/dehaze-voice
    dehaze-service/dehaze-modal
)

if [[ ! -x "$RUFF" ]]; then
    echo "缺少 ${RUFF}：请先在 dehaze-python 下执行 uv sync" >&2
    exit 1
fi

# pyright 版本必须固定：不同版本的 typeshed 不同，会得出不同结论
# （实例：asyncio.getaddrinfo 的 sockaddr 联合类型在 1.1.414 才补上 tuple[int, bytes]，
# 旧 CLI 因此漏检 app/utils/ssrf.py，而 IDE 报错）。此处与 IDE 的 ms-pyright 扩展对齐，
# 升级扩展时同步改这里；CI 亦可用环境变量显式指定。
PYRIGHT_VERSION="${PYRIGHT_VERSION:-1.1.414}"
export PYRIGHT_PYTHON_FORCE_VERSION="$PYRIGHT_VERSION"

# pyright 优先用项目环境内的可执行文件；未安装时回退到 PATH
if [[ -x "$PROJECT_DIR/.venv/bin/pyright" ]]; then
    PYRIGHT="$PROJECT_DIR/.venv/bin/pyright"
elif command -v pyright >/dev/null 2>&1; then
    PYRIGHT=pyright
else
    echo "缺少 pyright：请安装（如 uv add --dev pyright）后重试" >&2
    exit 1
fi

echo "==> ruff check（dehaze-python）"
(cd "$PROJECT_DIR" && "$RUFF" check .)

echo "==> ruff format --check（dehaze-python）"
(cd "$PROJECT_DIR" && "$RUFF" format --check .)

for service in "${RUFF_SERVICES[@]}"; do
    echo "==> ruff check / format --check（${service}）"
    (cd "$REPO_ROOT/dehaze-service/$service" && "$RUFF" check . && "$RUFF" format --check .)
done

for project in "${PYRIGHT_PROJECTS[@]}"; do
    config="$REPO_ROOT/$project/pyrightconfig.json"
    if [[ ! -f "$config" ]]; then
        echo "缺少 ${config}：纳入类型检查的子项目必须自持 pyright 配置" >&2
        exit 1
    fi
    echo "==> pyright ${PYRIGHT_VERSION}（${project}）"
    "$PYRIGHT" -p "$config" --warnings
done

echo "==> 全部通过"
