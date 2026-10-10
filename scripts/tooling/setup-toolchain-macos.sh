#!/usr/bin/env bash
#
# dehaze 开发环境统一管理（macOS）：brew(mise) + mise(语言运行时) + uv(Python)
#
# 幂等、分阶段、可回滚。安装与清理严格分离：
#   install —— 通用，任何机器（含全新机器）可反复运行，只装不删。
#   cleanup —— 机器特定，只清理"迁移前的旧装"（读取 legacy.macos，全新机器留空即无操作）。
#   Java 由 mise 全权管理（不复用 brew 已装的 openjdk），cleanup 阶段会卸载 brew 的 openjdk 包。
#
# 配置来源：同目录 tooling.config.json（与 Linux / Windows 脚本共用）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/tooling.config.json"
ENV_FILE="$HOME/.config/dehaze-tooling.zsh"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_ROOT="$SCRIPT_DIR/backup/macos-$STAMP"

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
  ./setup-toolchain-macos.sh                 安装（默认，通用，可反复运行）
  ./setup-toolchain-macos.sh --step cleanup  清理迁移前的旧装（机器特定，含 sudo rm /usr/local/go）
  ./setup-toolchain-macos.sh --step all      安装 + 清理
  ./setup-toolchain-macos.sh --dry-run       预演，不产生任何改动
  ./setup-toolchain-macos.sh --restore       还原 .zshrc / .zprofile / 环境
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

# 用 python3 解析配置，与 Linux 脚本共用同一份 tooling.config.json
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

ensure_brew() {
    command -v brew >/dev/null 2>&1 || { err '未检测到 Homebrew，请先手动安装 brew 后再运行本脚本'; exit 1; }
    ok "brew 已就绪（$(brew --version | head -n1)）"
}

# ---------------------------------------------------------------- 备份 / 还原
save_backup() {
    info "备份到 $BACKUP_ROOT"
    if [ "$DRY_RUN" = "1" ]; then
        log WHAT "备份 .zshrc / .zprofile / 环境变量 到 $BACKUP_ROOT"
        return
    fi
    mkdir -p "$BACKUP_ROOT"
    [ -f "$HOME/.zshrc" ] && cp "$HOME/.zshrc" "$BACKUP_ROOT/zshrc.bak"
    [ -f "$HOME/.zprofile" ] && cp "$HOME/.zprofile" "$BACKUP_ROOT/zprofile.bak"
    [ -f "$ENV_FILE" ] && cp "$ENV_FILE" "$BACKUP_ROOT/dehaze-tooling.zsh.bak"
    { for t in node npm pnpm go java python3 mise uv; do
        printf '%-8s %s\n' "$t" "$(command -v "$t" 2>/dev/null || echo '(none)')"
    done; } > "$BACKUP_ROOT/tools-snapshot.txt"
    ok '备份完成'
}

restore() {
    local src
    src="$(ls -1d "$SCRIPT_DIR"/backup/macos-* 2>/dev/null | sort | tail -n1 || true)"
    [ -n "$src" ] || { err '未找到 backup/macos-* 备份目录'; exit 1; }
    info "从 $src 还原"
    [ -f "$src/zshrc.bak" ] && run cp "$src/zshrc.bak" "$HOME/.zshrc"
    [ -f "$src/zprofile.bak" ] && run cp "$src/zprofile.bak" "$HOME/.zprofile"
    [ -f "$src/dehaze-tooling.zsh.bak" ] && run cp "$src/dehaze-tooling.zsh.bak" "$ENV_FILE"
    ok '还原完成（重开终端生效）'
}

# ---------------------------------------------------------------- 安装
install_mise() {
    if ! command -v mise >/dev/null 2>&1; then
        run brew install mise
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
        log WHAT "写入 ${ENV_FILE}（镜像 + mise 激活）并在 ~/.zshrc 引用"
        return
    fi
    mkdir -p "$(dirname "$ENV_FILE")"
    cat > "$ENV_FILE" <<EOF
# dehaze tooling env —— 由 scripts/tooling/setup-toolchain-macos.sh 生成，删除本文件即还原
export MISE_NODE_MIRROR="$node_mirror"
export GOPROXY="$go_proxy"
export GOSUMDB="$go_sumdb"
export UV_INDEX_URL="$pip_index"
export PIP_INDEX_URL="$pip_index"
eval "\$(mise activate zsh)"
EOF

    local marker='# dehaze tooling env'
    if ! grep -qF "$marker" "$HOME/.zshrc" 2>/dev/null; then
        printf '\n%s\n[ -f "%s" ] && . "%s"\n' "$marker" "$ENV_FILE" "$ENV_FILE" >> "$HOME/.zshrc"
    fi
    ok "已写入 $ENV_FILE 并在 ~/.zshrc 中引用"
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
    run mise use -g "node@$(cfg '.runtimes.node')"
    run mise use -g "go@$(cfg '.runtimes.go')"
    run mise use -g "java@$(cfg '.runtimes.java')"

    # 额外安装的 node/java 版本（只 install，不改变上面的全局默认版本）
    local v
    while IFS= read -r v; do
        [ -n "$v" ] || continue
        run mise install "node@$v"
    done < <(cfg '.macos.extraVersions.node')
    while IFS= read -r v; do
        [ -n "$v" ] || continue
        run mise install "java@$v"
    done < <(cfg '.macos.extraVersions.java')
}

install_global_npm_packages() {
    local node_ver
    node_ver="$(cfg '.macos.globalNpmPackages.nodeVersion')"
    [ -n "$node_ver" ] || return 0

    local -a pkgs=()
    local p
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        pkgs+=("$p")
    done < <(cfg '.macos.globalNpmPackages.packages')
    [ "${#pkgs[@]}" -gt 0 ] || return 0

    run mise exec "node@$node_ver" -- npm install -g "${pkgs[@]}"
    ok "已为 node@$node_ver 安装全局 npm 包：${pkgs[*]}"
}

install_python() {
    local ver
    ver="$(cfg '.python.default')"
    if ! command -v uv >/dev/null 2>&1; then
        run bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    fi
    # 不同 uv 版本对 --default 的稳定性不同（旧版需 preview）：先直试，失败再回退
    if ! run uv python install "$ver" --default; then
        warn '当前 uv 的 --default 需要 preview 模式，改用 --preview 重试'
        run uv python install "$ver" --preview --default
    fi
    ok "已安装 Python ${ver}（系统 python3 保持不变）"
}

invoke_install() {
    info '=== 安装阶段（通用，可反复运行）==='
    save_backup
    ensure_brew
    install_mise
    write_env_file
    configure_ecosystem_mirrors
    install_runtimes
    install_global_npm_packages
    install_python
    ok '安装阶段完成。请执行 source ~/.zshrc 或重开终端使配置生效。'
}

# ---------------------------------------------------------------- 清理
cleanup_legacy() {
    info '=== 清理阶段（机器特定，删除迁移前旧装）==='
    # 系统自带 /bin/bash 是 3.2（无 mapfile），用 while read 循环兼容读取
    local -a brew_remove=() manual_dirs=() sudo_dirs=() strip_patterns=()
    local line
    while IFS= read -r line; do brew_remove+=("$line"); done < <(cfg '.macos.legacy.brewRemove')
    while IFS= read -r line; do manual_dirs+=("$line"); done < <(cfg '.macos.legacy.manualDirs')
    while IFS= read -r line; do sudo_dirs+=("$line"); done < <(cfg '.macos.legacy.sudoDirs')
    while IFS= read -r line; do strip_patterns+=("$line"); done < <(cfg '.macos.legacy.shellStripPatterns')

    # 卸载旧 brew openjdk（Java 已交给 mise 全权管理）
    for p in "${brew_remove[@]}"; do
        [ -n "$p" ] || continue
        brew list --versions "$p" >/dev/null 2>&1 && run brew uninstall "$p"
    done

    # 删除手装目录（如 /usr/local/go），sudoDirs 中列出的用 sudo 删除
    for d in "${manual_dirs[@]}"; do
        [ -n "$d" ] || continue
        [ -e "$d" ] || continue
        local need_sudo=0 s
        for s in "${sudo_dirs[@]}"; do
            [ "$s" = "$d" ] && need_sudo=1
        done
        if [ "$need_sudo" = "1" ]; then
            run sudo rm -rf "$d"
        else
            run rm -rf "$d"
        fi
    done

    # 移除 ~/.zshrc 与 ~/.zprofile 中残留的旧 nvm / goenv / JAVA_HOME 初始化行
    local shell_file p
    for shell_file in "$HOME/.zshrc" "$HOME/.zprofile"; do
        [ -f "$shell_file" ] || continue
        for p in "${strip_patterns[@]}"; do
            [ -n "$p" ] || continue
            if [ "$DRY_RUN" = "1" ]; then
                log WHAT "从 $shell_file 移除含 '$p' 的行"
            else
                sed -i '' -E "\|$p|d" "$shell_file" 2>/dev/null || true
            fi
        done
    done
    ok '清理完成。'
}

show_summary() {
    info '=== 结果核对 ==='
    for t in mise uv node go java python3; do
        printf '  %-8s %s\n' "$t" "$(command -v "$t" 2>/dev/null || echo '(none)')"
    done
    info "备份目录：$BACKUP_ROOT"
}

# ---------------------------------------------------------------- 入口
require_python3

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
