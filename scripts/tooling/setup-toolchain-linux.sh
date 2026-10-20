#!/usr/bin/env bash
#
# clarity 开发环境统一管理（WSL / Linux，含 TencentOS/CentOS/Rocky 等 RHEL 系）：
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
ENV_FILE="$HOME/.config/clarity-tooling.sh"
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
    [ -f "$ENV_FILE" ] && cp "$ENV_FILE" "$BACKUP_ROOT/clarity-tooling.sh.bak"
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
    [ -f "$src/clarity-tooling.sh.bak" ] && run cp "$src/clarity-tooling.sh.bak" "$ENV_FILE"
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
    local -a pkgs=()
    local install_cmd=""
    case "$PKG_MGR" in
        apt)
            mapfile -t pkgs < <(cfg '.linux.aptPackages')
            install_cmd="apt-get install"
            run $SUDO apt-get update
            ;;
        dnf|yum)
            mapfile -t pkgs < <(cfg '.linux.dnfPackages')
            # strict=0：个别包名在该发行版不存在时跳过，而不是让整批安装失败
            install_cmd="$PKG_MGR install --setopt=strict=0"
            ;;
    esac
    [ "${#pkgs[@]}" -gt 0 ] || return 0
    # 开发机通常已装齐；个别包名在 RHEL 系不存在时不应中断整个安装流程
    if [ "$DRY_RUN" = "1" ]; then
        log WHAT "$install_cmd -y ${pkgs[*]}"
        return
    fi
    log STEP "$install_cmd -y ${pkgs[*]}"
    if ! $SUDO $install_cmd -y "${pkgs[@]}"; then
        warn '系统包安装未全部成功，跳过继续（通常是该发行版缺少个别包名）'
    fi
}

install_mise() {
    if ! command -v mise >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/mise" ]; then
        run bash -c 'curl -fsSL https://mise.run | sh'
    fi
    ok 'mise 已就绪'
}

write_env_file() {
    local node_mirror override
    node_mirror="$(cfg '.mirrors.miseNodeMirror')"
    override="$(cfg '.mirrors.overrideExisting')"

    if [ "$DRY_RUN" = "1" ]; then
        log WHAT "写入 ${ENV_FILE}（mise 节点镜像 + mise 激活 + PATH）并在 ~/.bashrc 引用"
        return
    fi
    mkdir -p "$(dirname "$ENV_FILE")"
    {
        cat <<EOF
# clarity tooling env —— 由 scripts/tooling/setup-toolchain-linux.sh 生成，删除本文件即还原
export MISE_NODE_MIRROR="$node_mirror"
EOF
        # 仅在显式开启时才写入公网 GOPROXY/GOSUMDB/pip 镜像，避免顶掉内网源
        if [ "$override" = "true" ]; then
            cat <<EOF
export GOPROXY="$(cfg '.mirrors.goProxy')"
export GOSUMDB="$(cfg '.mirrors.goSumDb')"
export UV_INDEX_URL="$(cfg '.mirrors.pipIndex')"
export PIP_INDEX_URL="$(cfg '.mirrors.pipIndex')"
EOF
        fi
        cat <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
eval "$($HOME/.local/bin/mise activate bash)"
EOF
    } > "$ENV_FILE"

    local marker='# clarity tooling env'
    if ! grep -qF "$marker" "$HOME/.bashrc" 2>/dev/null; then
        printf '\n%s\n[ -f "%s" ] && . "%s"\n' "$marker" "$ENV_FILE" "$ENV_FILE" >> "$HOME/.bashrc"
    fi
    ok "已写入 $ENV_FILE 并在 ~/.bashrc 中引用"
}

configure_ecosystem_mirrors() {
    if [ "$(cfg '.mirrors.overrideExisting')" != "true" ]; then
        info '跳过 npm/pip 镜像写入（mirrors.overrideExisting=false，保留机器已有源）'
        return 0
    fi
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

    # 额外安装的 node/java 版本（只 install，不改变上面的全局默认版本）
    local v
    while IFS= read -r v; do
        [ -n "$v" ] || continue
        run mise install "node@$v"
    done < <(cfg '.linux.extraVersions.node')
    while IFS= read -r v; do
        [ -n "$v" ] || continue
        run mise install "java@$v"
    done < <(cfg '.linux.extraVersions.java')
}

install_global_npm_packages() {
    export PATH="$HOME/.local/bin:$PATH"
    local node_ver
    node_ver="$(cfg '.globalNpmPackages.nodeVersion')"
    [ -n "$node_ver" ] || return 0

    local -a pkgs=()
    local p
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        pkgs+=("$p")
    done < <(cfg '.globalNpmPackages.packages')
    [ "${#pkgs[@]}" -gt 0 ] || return 0

    run mise exec "node@$node_ver" -- npm install -g "${pkgs[@]}"
    ok "已为 node@$node_ver 安装全局 npm 包：${pkgs[*]}"
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
    install_global_npm_packages
    install_python
    ok '安装阶段完成。请执行 source ~/.bashrc 或重开终端使配置生效。'
}

# ---------------------------------------------------------------- 清理
# 把配置里的 `~/xxx` 展开成 $HOME/xxx（JSON 里无法直接写 shell 变量）
expand_home() {
    case "$1" in
        '~/'*) printf '%s/%s' "$HOME" "${1#\~/}" ;;
        *)     printf '%s' "$1" ;;
    esac
}

cleanup_legacy() {
    info '=== 清理阶段（机器特定，删除迁移前旧装）==='
    local -a remove=() installed=() manual_dirs=() manual_files=() strip_patterns=()
    mapfile -t remove < <(cfg '.legacy.linux.aptRemove')
    mapfile -t manual_dirs < <(cfg '.legacy.linux.manualDirs')
    mapfile -t manual_files < <(cfg '.legacy.linux.manualFiles')
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

    # 删除手装单文件（如 ~/.local/bin/go 这种独立二进制）
    for f in "${manual_files[@]}"; do
        [ -n "$f" ] || continue
        f="$(expand_home "$f")"
        [ -f "$f" ] || continue
        if [ -w "$f" ]; then
            run rm -f "$f"
        else
            run $SUDO rm -f "$f"
        fi
    done

    # 删除手装目录（如 /usr/local/go、~/.nvm、~/.goenv、/usr/local/java/jdk*）
    for d in "${manual_dirs[@]}"; do
        [ -n "$d" ] || continue
        d="$(expand_home "$d")"
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
