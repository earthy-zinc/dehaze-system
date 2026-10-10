#!/usr/bin/env bash
#
# dehaze 开发环境统一管理（WSL / Linux，含 TencentOS/CentOS/Rocky 等 RHEL 系）：
# apt 或 dnf/yum(系统/CLI) + mise(语言运行时) + uv(Python)
#
# 幂等、分阶段、可回滚。安装与清理严格分离：
#   install —— 通用，任何机器（含全新机器）可反复运行，只装不删。
#   cleanup —— 机器特定，只清理“迁移前的旧装”（读取 legacy.linux，全新机器留空即无操作）。
#
# 配置来源：同目录 tooling.config.json（与 Windows 脚本共用）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/tooling.config.json"
ENV_FILE="$HOME/.config/dehaze-tooling.sh"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_ROOT="$SCRIPT_DIR/backup/linux-$STAMP"

STEP="install"
DRY_RUN=0
RESTORE=0

log()  { printf '[%s] %s\n' "$1" "$2"; }
info() { log INFO "$1"; }
ok()   { log OK "$1"; }
warn() { log WARN "$1"; }
err()  { log ERR "$1" >&2; }

run() {
    if [ "$DRY_RUN" = "1" ]; then
        log WHAT "$*"
    else
        log STEP "$*"
        "$@"
    fi
}

usage() {
    cat <<'EOF'
用法:
  ./setup-toolchain-linux.sh                 安装（默认，通用，可反复运行）
  ./setup-toolchain-linux.sh --step cleanup  清理迁移前的旧装（机器特定）
  ./setup-toolchain-linux.sh --step all      安装 + 清理
  ./setup-toolchain-linux.sh --dry-run       预演，不产生任何改动
  ./setup-toolchain-linux.sh --restore       还原 apt 源（仅 apt 机器）/ .bashrc / 环境
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --step) STEP="${2:-}"; shift 2 ;;
        --dry-run|-n) DRY_RUN=1; shift ;;
        --restore) RESTORE=1; shift ;;
        -h|--help) usage ;;
        *) err "未知参数：$1"; exit 1 ;;
    esac
done

case "$STEP" in
    install|cleanup|all) ;;
    *) err "--step 只能是 install|cleanup|all"; exit 1 ;;
esac

[ -f "$CONFIG" ] || { err "找不到配置：$CONFIG"; exit 1; }

# ---------------------------------------------------------------- 基础环境
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# 用 python3 解析配置（Ubuntu 必备），避免依赖 jq 造成先装 jq 才能读配置的鸡生蛋问题
cfg() {
    python3 - "$CONFIG" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    val = json.load(f)
for key in sys.argv[2].lstrip('.').split('.'):
    val = val[key]
if val is None:
    print('')
elif isinstance(val, bool):
    print('true' if val else 'false')
elif isinstance(val, list):
    print('\n'.join(str(x) for x in val))
else:
    print(val)
PY
}

require_python3() {
    command -v python3 >/dev/null 2>&1 || { err '缺少 python3，无法解析配置'; exit 1; }
}

detect_distro() {
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "${ID:-unknown}"
}

# apt（Ubuntu/Debian/WSL）或 dnf/yum（TencentOS/CentOS/Rocky 等 RHEL 系）；
# 没有换源逻辑的 RHEL 系默认沿用机器现有内网源，不做额外改造。
detect_pkg_mgr() {
    if command -v apt-get >/dev/null 2>&1; then
        echo apt
    elif command -v dnf >/dev/null 2>&1; then
        echo dnf
    elif command -v yum >/dev/null 2>&1; then
        echo yum
    else
        err '未找到支持的包管理器（apt-get/dnf/yum）'
        exit 1
    fi
}

# ---------------------------------------------------------------- 备份 / 还原
save_backup() {
    info "备份到 $BACKUP_ROOT"
    if [ "$DRY_RUN" = "1" ]; then
        log WHAT "备份 apt 源 / .bashrc / 环境变量 到 $BACKUP_ROOT"
        return
    fi
    mkdir -p "$BACKUP_ROOT"
    local apt_source
    apt_source="$(cfg '.linux.aptSourceFile')"
    [ -f "$apt_source" ] && cp "$apt_source" "$BACKUP_ROOT/apt-sources.bak"
    [ -f "$HOME/.bashrc" ] && cp "$HOME/.bashrc" "$BACKUP_ROOT/bashrc.bak"
    [ -f "$ENV_FILE" ] && cp "$ENV_FILE" "$BACKUP_ROOT/dehaze-tooling.sh.bak"
    { for t in node npm pnpm go java python3 mise uv; do
        printf '%-8s %s\n' "$t" "$(command -v "$t" 2>/dev/null || echo '(none)')"
    done; } > "$BACKUP_ROOT/tools-snapshot.txt"
    ok '备份完成'
}

restore() {
    local src
    src="$(ls -1d "$SCRIPT_DIR"/backup/linux-* 2>/dev/null | sort | tail -n1 || true)"
    [ -n "$src" ] || { err '未找到 backup/linux-* 备份目录'; exit 1; }
    info "从 $src 还原"
    local apt_source; apt_source="$(cfg '.linux.aptSourceFile')"
    [ -f "$src/apt-sources.bak" ] && run $SUDO cp "$src/apt-sources.bak" "$apt_source"
    [ -f "$src/bashrc.bak" ] && run cp "$src/bashrc.bak" "$HOME/.bashrc"
    [ -f "$src/dehaze-tooling.sh.bak" ] && run cp "$src/dehaze-tooling.sh.bak" "$ENV_FILE"
    ok '还原完成（重开终端生效）'
}

# ---------------------------------------------------------------- 安装
set_apt_mirror() {
    if [ "$PKG_MGR" != "apt" ]; then
        info "包管理器为 $PKG_MGR，非 apt，跳过换源（沿用机器现有内网源）"
        return
    fi

    local mirror apt_source distro codename
    mirror="$(cfg '.mirrors.aptMirror')"
    apt_source="$(cfg '.linux.aptSourceFile')"
    distro="$(detect_distro)"
    codename="$(. /etc/os-release; echo "${VERSION_CODENAME:-}")"

    if [ "$distro" != "ubuntu" ] && [ "$distro" != "debian" ]; then
        warn "发行版 $distro 非 ubuntu/debian，跳过 apt 换源"
        return
    fi
    if [ ! -f "$apt_source" ]; then
        warn "未找到 $apt_source，跳过 apt 换源"
        return
    fi

    if [ "$DRY_RUN" = "1" ]; then
        log WHAT "规范化 $apt_source 为 $codename 套件并指向 $mirror"
        return
    fi

    if [ "$distro" = "ubuntu" ]; then
        # 重写为当前发行版的标准套件，顺带清掉残留的 EOL 套件（如 noble 机器上的 lunar），
        # 否则 apt-get update 会因那些套件 404 而失败。
        $SUDO tee "$apt_source" >/dev/null <<EOF
Types: deb
URIs: $mirror
Suites: $codename $codename-updates $codename-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: $mirror
Suites: $codename-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
    else
        $SUDO sed -i -E "s|^URIs:.*|URIs: $mirror|" "$apt_source"
    fi
    ok "apt 源已指向 ${mirror}（${codename}）"
}

install_system_packages() {
    local -a pkgs
    case "$PKG_MGR" in
        apt)
            mapfile -t pkgs < <(cfg '.linux.aptPackages')
            run $SUDO apt-get update
            run $SUDO apt-get install -y "${pkgs[@]}"
            ;;
        dnf|yum)
            mapfile -t pkgs < <(cfg '.linux.dnfPackages')
            run $SUDO "$PKG_MGR" install -y "${pkgs[@]}"
            ;;
    esac
}

install_mise() {
    if ! command -v mise >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/mise" ]; then
        run bash -c 'curl -fsSL https://mise.run | sh'
    fi
    ok 'mise 已就绪'
}

write_env_file() {
    local node_mirror go_proxy go_sumdb pip_index
    node_mirror="$(cfg '.mirrors.miseNodeMirror')"
    go_proxy="$(cfg '.mirrors.goProxy')"
    go_sumdb="$(cfg '.mirrors.goSumDb')"
    pip_index="$(cfg '.mirrors.pipIndex')"

    if [ "$DRY_RUN" = "1" ]; then
        log WHAT "写入 ${ENV_FILE}（镜像 + mise 激活 + PATH）并在 ~/.bashrc 引用"
        return
    fi
    mkdir -p "$(dirname "$ENV_FILE")"
    cat > "$ENV_FILE" <<EOF
# dehaze tooling env —— 由 scripts/tooling/setup-toolchain-linux.sh 生成，删除本文件即还原
export MISE_NODE_MIRROR="$node_mirror"
export GOPROXY="$go_proxy"
export GOSUMDB="$go_sumdb"
export UV_INDEX_URL="$pip_index"
export PIP_INDEX_URL="$pip_index"
export PATH="\$HOME/.local/bin:\$PATH"
eval "\$(\$HOME/.local/bin/mise activate bash)"
EOF

    local marker='# dehaze tooling env'
    if ! grep -qF "$marker" "$HOME/.bashrc" 2>/dev/null; then
        printf '\n%s\n[ -f "%s" ] && . "%s"\n' "$marker" "$ENV_FILE" "$ENV_FILE" >> "$HOME/.bashrc"
    fi
    ok "已写入 $ENV_FILE 并在 ~/.bashrc 中引用"
}

configure_ecosystem_mirrors() {
    local npm pip_index
    npm="$(cfg '.mirrors.npmRegistry')"
    pip_index="$(cfg '.mirrors.pipIndex')"

    if [ "$DRY_RUN" = "1" ]; then
        log WHAT "写 ~/.npmrc registry 与 ~/.config/pip/pip.conf index-url"
        return
    fi
    if ! grep -q 'registry=' "$HOME/.npmrc" 2>/dev/null; then
        printf 'registry=%s\n' "$npm" >> "$HOME/.npmrc"
    fi
    mkdir -p "$HOME/.config/pip"
    if ! grep -q 'index-url' "$HOME/.config/pip/pip.conf" 2>/dev/null; then
        printf '[global]\nindex-url = %s\n' "$pip_index" >> "$HOME/.config/pip/pip.conf"
    fi
    ok 'npm / pip 镜像已配置'
}

install_runtimes() {
    export PATH="$HOME/.local/bin:$PATH"
    run mise use -g "node@$(cfg '.runtimes.node')"
    run mise use -g "go@$(cfg '.runtimes.go')"
    run mise use -g "java@$(cfg '.runtimes.java')"
}

install_python() {
    local ver
    ver="$(cfg '.python.default')"
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        run bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    fi
    # 不同 uv 版本对 --default 的稳定性不同（旧版需 preview）：先直试，失败再回退
    if ! run uv python install "$ver" --default; then
        warn '当前 uv 的 --default 需要 preview 模式，改用 --preview 重试'
        run uv python install "$ver" --preview --default
    fi
    ok "已安装 Python ${ver}（系统 /usr/bin/python3 保持不变）"
}

invoke_install() {
    info '=== 安装阶段（通用，可反复运行）==='
    save_backup
    set_apt_mirror
    install_system_packages
    install_mise
    write_env_file
    configure_ecosystem_mirrors
    install_runtimes
    install_python
    ok '安装阶段完成。请执行 source ~/.bashrc 或重开终端使配置生效。'
}

# ---------------------------------------------------------------- 清理
cleanup_legacy() {
    info '=== 清理阶段（机器特定，删除迁移前旧装）==='
    local -a remove=() installed=() manual_dirs=() strip_patterns=()
    mapfile -t remove < <(cfg '.legacy.linux.aptRemove')
    mapfile -t manual_dirs < <(cfg '.legacy.linux.manualDirs')
    mapfile -t strip_patterns < <(cfg '.legacy.linux.shellStripPatterns')

    # 只卸载真正已安装的包，避免某个包不存在时 apt 报错中断脚本。
    # 不使用 --autoremove：它会连带清扫无关的自动安装包，曾导致整条 CUDA 被误删。
    # legacy.linux.aptRemove 是 Debian 系旧装清理专用，非 apt 机器上跳过（如需清理 dnf 包请手工处理）。
    if [ "$PKG_MGR" = "apt" ]; then
        for p in "${remove[@]}"; do
            [ -n "$p" ] || continue
            dpkg -s "$p" >/dev/null 2>&1 && installed+=("$p")
        done
        if [ "${#installed[@]}" -gt 0 ]; then
            run $SUDO apt-get remove -y "${installed[@]}"
        fi
    elif [ "${#remove[@]}" -gt 0 ] && [ -n "${remove[0]}" ]; then
        warn "包管理器为 $PKG_MGR，跳过 legacy.linux.aptRemove（${remove[*]} 需手工处理）"
    fi

    # 删除手装目录（如 /usr/local/go、~/.nvm）
    for d in "${manual_dirs[@]}"; do
        [ -n "$d" ] || continue
        [ -d "$d" ] || continue
        if [ -w "$(dirname "$d")" ]; then
            run rm -rf "$d"
        else
            run $SUDO rm -rf "$d"
        fi
    done

    # 移除 ~/.bashrc 与 ~/.profile 中残留的旧 PATH / nvm / conda 初始化行
    local shell_file p
    for shell_file in "$HOME/.bashrc" "$HOME/.profile"; do
        [ -f "$shell_file" ] || continue
        for p in "${strip_patterns[@]}"; do
            [ -n "$p" ] || continue
            if [ "$DRY_RUN" = "1" ]; then
                log WHAT "从 $shell_file 移除含 '$p' 的行"
            else
                sed -i "\|$p|d" "$shell_file" 2>/dev/null || true
            fi
        done
    done
    ok '清理完成。'
}

show_summary() {
    info '=== 结果核对 ==='
    export PATH="$HOME/.local/bin:$PATH"
    for t in mise uv node go java python3; do
        printf '  %-8s %s\n' "$t" "$(command -v "$t" 2>/dev/null || echo '(none)')"
    done
    info "备份目录：$BACKUP_ROOT"
}

# ---------------------------------------------------------------- 入口
require_python3
PKG_MGR="$(detect_pkg_mgr)"
info "包管理器：$PKG_MGR"

if [ "$RESTORE" = "1" ]; then
    restore
    exit 0
fi

case "$STEP" in
    install) invoke_install ;;
    cleanup) cleanup_legacy ;;
    all) invoke_install; cleanup_legacy ;;
esac

show_summary
