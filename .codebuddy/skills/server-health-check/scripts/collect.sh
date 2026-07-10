#!/usr/bin/env bash
# 服务器健康检查 - 通用一键数据采集脚本
# 支持: Linux (RHEL/CentOS/Fedora/TencentOS/Debian/Ubuntu/Alpine), macOS
# Windows 请使用 collect.ps1
# 用法: bash collect.sh

set -o pipefail

section() { echo -e "\n====== $1 ======"; }

OS_TYPE="$(uname -s)"

# Windows (MINGW/MSYS/Cygwin) 环境检测
case "$OS_TYPE" in
    MINGW*|MSYS*|CYGWIN*)
        echo "检测到 Windows 环境 ($OS_TYPE)，请使用 collect.ps1:"
        echo "  powershell -ExecutionPolicy Bypass -File \"\$(dirname \"\$0\")/collect.ps1\""
        exit 1
        ;;
esac

# ============================================================
# macOS 采集
# ============================================================
if [ "$OS_TYPE" = "Darwin" ]; then

section "环境探测"
echo "系统: macOS $(sw_vers -productVersion 2>/dev/null)"
echo "架构: $(uname -m)"
echo "主机: $(hostname)"
echo "内核: $(uname -r)"

# Homebrew
section "Homebrew"
if command -v brew &>/dev/null; then
    echo "Homebrew 版本: $(brew --version | head -1)"
    echo "--- 手动安装的 formulae (brew leaves) ---"
    brew leaves 2>/dev/null
    echo "--- 已安装 cask ---"
    brew list --cask 2>/dev/null
    echo "--- brew 占用空间 ---"
    du -sh "$(brew --prefix)/Cellar" 2>/dev/null || true
    du -sh "$(brew --prefix)/Caskroom" 2>/dev/null || true
    echo "--- 可清理空间 ---"
    brew cleanup --dry-run 2>/dev/null | tail -5
else
    echo "未安装 Homebrew"
fi

# Node.js / nvm
section "Node.js 环境"
if [ -d "$HOME/.nvm" ]; then
    echo "--- nvm 已安装版本 ---"
    ls -1 "$HOME/.nvm/versions/node/" 2>/dev/null || echo "无"
    echo "--- nvm 占用空间 ---"
    du -sh "$HOME/.nvm" 2>/dev/null
fi
if command -v node &>/dev/null; then
    echo "当前 node: $(node -v 2>/dev/null)"
    echo "--- 全局 npm 包 ---"
    npm ls -g --depth=0 2>/dev/null || true
fi
if command -v pnpm &>/dev/null; then
    echo "--- 全局 pnpm 包 ---"
    pnpm ls -g 2>/dev/null || true
    echo "--- pnpm store 占用 ---"
    du -sh "$HOME/Library/pnpm/store" 2>/dev/null || true
fi

# Python
section "Python 环境"
if command -v python3 &>/dev/null; then
    echo "Python: $(python3 --version 2>/dev/null)"
    echo "--- pip 全局包 ---"
    pip3 list --format=columns 2>/dev/null | head -30 || true
fi
if command -v conda &>/dev/null; then
    echo "--- conda 环境 ---"
    conda env list 2>/dev/null
fi

# Java
section "Java 环境"
if command -v java &>/dev/null; then
    echo "Java: $(java -version 2>&1 | head -1)"
    echo "JAVA_HOME: ${JAVA_HOME:-未设置}"
fi
if [ -d "$HOME/.sdkman" ]; then
    echo "--- SDKMAN 已安装版本 ---"
    ls -1 "$HOME/.sdkman/candidates/java/" 2>/dev/null | head -10
    echo "--- SDKMAN 占用空间 ---"
    du -sh "$HOME/.sdkman" 2>/dev/null
fi

# 系统资源
section "系统概况"
echo "--- 内存 ---"
vm_stat 2>/dev/null | head -10
echo "--- 物理内存 ---"
sysctl -n hw.memsize 2>/dev/null | awk '{printf "总内存: %.1f GB\n", $1/1073741824}'
echo "--- 磁盘 ---"
df -h 2>/dev/null | grep -vE 'devfs|map '

section "CPU & 负载"
echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "核心: $(sysctl -n hw.ncpu 2>/dev/null)"
echo "负载: $(sysctl -n vm.loadavg 2>/dev/null)"

# 大目录
section "大目录 TOP20"
du -sh ~/Library/Caches/* ~/Library/Developer/* ~/Library/Application\ Support/* /usr/local/* /opt/* 2>/dev/null | sort -rh | head -20

section "缓存与临时文件"
echo "--- ~/Library/Caches ---"
du -sh ~/Library/Caches 2>/dev/null
echo "--- Xcode DerivedData ---"
du -sh ~/Library/Developer/Xcode/DerivedData 2>/dev/null || echo "无"
echo "--- npm cache ---"
du -sh ~/.npm 2>/dev/null || echo "无"
echo "--- pip cache ---"
du -sh ~/Library/Caches/pip 2>/dev/null || echo "无"
echo "--- gradle cache ---"
du -sh ~/.gradle 2>/dev/null || echo "无"
echo "--- maven cache ---"
du -sh ~/.m2 2>/dev/null || echo "无"

# 容器
section "容器"
CTR="none"
if command -v docker &>/dev/null && docker info &>/dev/null; then CTR="docker"
elif command -v podman &>/dev/null; then CTR="podman"
fi
if [ "$CTR" != "none" ]; then
    echo "--- 容器列表 ---"
    $CTR ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || $CTR ps -a
    echo "--- 空间占用 ---"
    $CTR system df 2>/dev/null || true
    echo "--- 镜像 ---"
    $CTR images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' 2>/dev/null || $CTR images
    echo "--- 悬空镜像 ---"
    $CTR images -f "dangling=true" -q 2>/dev/null | wc -l | xargs -I{} echo "悬空镜像数: {}"
    echo "--- 同镜像重复容器 ---"
    $CTR ps -a --format '{{.Image}}' 2>/dev/null | sort | uniq -cd | sort -rn || echo "无重复"
    echo "--- 未使用 Volume ---"
    $CTR volume ls -f dangling=true 2>/dev/null || true
else
    echo "无容器运行时"
fi

# 进程
section "进程 - 内存 TOP30"
ps aux -m 2>/dev/null | head -31

section "进程 - CPU TOP15"
ps aux -r 2>/dev/null | head -16

section "重复/遗留进程检测"
for pattern in "mcp-server" "mcp_server" "language-server"; do
    HITS=$(ps aux 2>/dev/null | grep "$pattern" | grep -v grep || true)
    if [ -n "$HITS" ]; then
        echo "[$pattern]:"
        echo "$HITS" | awk '{print $2, $9, $11, $NF}'
    fi
done
echo "--- 重复应用进程 ---"
ps -eo pid,lstart,comm 2>/dev/null | grep -E '(python|java|node|ruby|go)' | grep -v grep || echo "无"

section "僵尸进程"
ZOMBIES=$(ps aux 2>/dev/null | awk '$8~/Z/' | grep -v grep || true)
if [ -n "$ZOMBIES" ]; then
    echo "$ZOMBIES"
    echo "数量: $(echo "$ZOMBIES" | wc -l)"
else
    echo "无僵尸进程"
fi

# 端口
section "端口监听"
lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | head -50 || echo "lsof 不可用"

section "对外暴露高危端口"
lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | grep -E '\*(3306|5432|27017|6379|11211|9000|9090|9200|9300|8080|8443|15672) ' || echo "无高危端口对外暴露"

# 服务
section "LaunchDaemons & LaunchAgents"
echo "--- 系统 LaunchDaemons (非 Apple) ---"
ls /Library/LaunchDaemons/ 2>/dev/null | grep -v com.apple || echo "无"
echo "--- 用户 LaunchAgents (非 Apple) ---"
ls ~/Library/LaunchAgents/ 2>/dev/null | grep -v com.apple || echo "无"
echo "--- 系统 LaunchAgents (非 Apple) ---"
ls /Library/LaunchAgents/ 2>/dev/null | grep -v com.apple || echo "无"

# 定时任务
section "定时任务"
echo "--- crontab ---"
crontab -l 2>/dev/null || echo "无"
echo "--- launchctl 自定义 ---"
launchctl list 2>/dev/null | grep -vE '^\s*-\s+0\s+com\.apple' | head -20 || echo "无"

# 日志空间
section "日志空间"
echo "--- /var/log ---"
du -sh /var/log 2>/dev/null || true
echo "--- /var/log TOP10 ---"
du -sh /var/log/* 2>/dev/null | sort -rh | head -10
echo "--- ASL 日志 ---"
du -sh /var/log/asl 2>/dev/null || echo "无"
echo "--- 系统诊断报告 ---"
du -sh /Library/Logs/DiagnosticReports 2>/dev/null || echo "无"
du -sh ~/Library/Logs/DiagnosticReports 2>/dev/null || echo "无"

# 防火墙
section "防火墙"
echo "--- macOS 应用防火墙 ---"
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null || echo "无法检测"
echo "--- pf 状态 ---"
sudo pfctl -s info 2>/dev/null | head -5 || echo "pf 未启用或需要 sudo"

# Go 环境
section "Go 环境"
if command -v go &>/dev/null; then
    echo "Go: $(go version 2>/dev/null)"
    echo "GOPATH: ${GOPATH:-$(go env GOPATH 2>/dev/null)}"
    echo "--- Go 占用空间 ---"
    du -sh "$(go env GOPATH 2>/dev/null)" 2>/dev/null || true
    du -sh "$(go env GOMODCACHE 2>/dev/null)" 2>/dev/null || true
fi

# Python 多版本补充
if [ -d "$HOME/.pyenv" ]; then
    section "pyenv 环境"
    echo "--- pyenv 已安装版本 ---"
    ls -1 "$HOME/.pyenv/versions/" 2>/dev/null || echo "无"
    echo "--- pyenv 占用空间 ---"
    du -sh "$HOME/.pyenv" 2>/dev/null
fi

section "采集完成"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "系统: macOS $(sw_vers -productVersion 2>/dev/null) ($(uname -m))"

exit 0
fi

# ============================================================
# Linux 采集
# ============================================================

section "环境探测"
echo "--- 发行版 ---"
cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION|ID)=' || echo "未知"
echo "--- 架构 ---"
uname -m
echo "--- 虚拟化 ---"
systemd-detect-virt 2>/dev/null || ([ -f /sys/class/dmi/id/product_name ] && cat /sys/class/dmi/id/product_name) || echo "未知"

# 探测包管理器
PKG="unknown"
if command -v rpm &>/dev/null; then PKG="rpm"
elif command -v dpkg &>/dev/null; then PKG="deb"
elif command -v apk &>/dev/null; then PKG="apk"
fi
echo "包管理器: $PKG"

# 探测容器运行时
CTR="none"
if command -v docker &>/dev/null && docker info &>/dev/null; then CTR="docker"
elif command -v podman &>/dev/null; then CTR="podman"
fi
echo "容器运行时: $CTR"

# --- 系统概况 ---
section "系统概况"
echo "--- 主机 ---"
hostname 2>/dev/null; uname -r; uptime
echo "--- 内存 ---"
free -h 2>/dev/null || cat /proc/meminfo | head -5
echo "--- 磁盘 ---"
df -h 2>/dev/null | grep -vE 'tmpfs|devtmpfs|overlay' || df -h
echo "--- Swap ---"
swapon --show 2>/dev/null || echo "无 Swap"

section "CPU & 负载"
top -bn1 2>/dev/null | head -5 || { echo "CPU核心: $(nproc)"; cat /proc/loadavg; }

# --- 大目录 ---
section "大目录 TOP20"
du -sh /opt/* /usr/local/* /home/* /root/* /var/lib/docker /var/lib/containers /srv/* 2>/dev/null | sort -rh | head -20

# --- 日志空间 ---
section "日志空间 TOP10"
du -sh /var/log/* 2>/dev/null | sort -rh | head -10
journalctl --disk-usage 2>/dev/null || true

# --- 软件包 ---
section "软件包统计"
case "$PKG" in
  rpm)
    echo "总数: $(rpm -qa | wc -l)"
    echo "--- 按大小 TOP30 ---"
    rpm -qa --qf '%{NAME}\t%{SIZE}\n' | sort -t$'\t' -k2 -rn | head -30
    ;;
  deb)
    echo "总数: $(dpkg -l | grep '^ii' | wc -l)"
    echo "--- 按大小 TOP30 (KB) ---"
    dpkg-query -W -f '${Package}\t${Installed-Size}\n' | sort -t$'\t' -k2 -rn | head -30
    ;;
  apk)
    echo "总数: $(apk list --installed 2>/dev/null | wc -l)"
    echo "--- 已安装 ---"
    apk list --installed 2>/dev/null | head -30
    ;;
  *) echo "未知包管理器，跳过" ;;
esac

# --- 内核 ---
section "内核版本"
echo "运行中: $(uname -r)"
echo "已安装:"
case "$PKG" in
  rpm) rpm -qa | grep -E '^kernel-(core|devel|modules|debuginfo)' | sort ;;
  deb) dpkg -l | grep -E 'linux-(image|headers|modules)' | awk '{print $2, $3}' ;;
  *) echo "N/A" ;;
esac

# --- 固件 ---
section "硬件固件包"
case "$PKG" in
  rpm) rpm -qa | grep -iE 'firmware' | sort || echo "无" ;;
  deb) dpkg -l | grep -i firmware | awk '{print $2}' | sort || echo "无" ;;
  *) echo "N/A" ;;
esac

# --- 开发工具链 ---
section "开发工具链"
case "$PKG" in
  rpm) rpm -qa | grep -E '^(gcc|gcc-c\+\+|gcc-gfortran|cmake|llvm|clang|buildah|bison|autoconf|automake|make)-' | sort || echo "无" ;;
  deb) dpkg -l | grep -E '^ii.*(gcc|g\+\+|gfortran|cmake|llvm|clang|buildah|bison|autoconf|automake|build-essential)' | awk '{print $2}' | sort || echo "无" ;;
  *) echo "N/A" ;;
esac

# --- 容器 ---
section "容器"
if [ "$CTR" != "none" ]; then
    echo "--- 容器列表 ---"
    $CTR ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || $CTR ps -a
    echo "--- 空间占用 ---"
    $CTR system df 2>/dev/null || true
    echo "--- 镜像 ---"
    $CTR images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' 2>/dev/null || $CTR images
    echo "--- 同镜像重复容器 ---"
    $CTR ps -a --format '{{.Image}}' 2>/dev/null | sort | uniq -cd | sort -rn || echo "无重复"
    echo "--- 悬空镜像 ---"
    $CTR images -f "dangling=true" -q 2>/dev/null | wc -l | xargs -I{} echo "悬空镜像数: {}"
    echo "--- 未使用 Volume ---"
    $CTR volume ls -f dangling=true 2>/dev/null || true
else
    echo "无容器运行时"
fi

# --- 进程 ---
section "进程 - 内存 TOP30"
ps aux --sort=-%mem 2>/dev/null | head -31 || ps aux | head -31

section "进程 - CPU TOP15"
ps aux --sort=-%cpu 2>/dev/null | head -16 || true

section "僵尸进程"
ZOMBIES=$(ps aux 2>/dev/null | awk '$8~/Z/' | grep -v grep || true)
if [ -n "$ZOMBIES" ]; then
    echo "$ZOMBIES"
    echo "数量: $(echo "$ZOMBIES" | wc -l)"
else
    echo "无僵尸进程"
fi

section "重复/遗留进程检测"
echo "--- 同名进程多实例 (按启动时间) ---"
for pattern in "mcp-server" "mcp_server" "language-server"; do
    HITS=$(ps aux 2>/dev/null | grep "$pattern" | grep -v grep || true)
    if [ -n "$HITS" ]; then
        echo "[$pattern]:"
        echo "$HITS" | awk '{print $2, $9, $11, $NF}'
    fi
done
echo "--- 重复应用进程 ---"
ps -eo pid,lstart,args --sort=lstart 2>/dev/null | grep -E '(run\.py|java -jar|node server|main\.go)' | grep -v grep || echo "无"

# --- 端口 ---
section "端口监听"
ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null || echo "ss/netstat 不可用"

section "对外暴露高危端口"
ss -tlnp 2>/dev/null | grep -E '(0\.0\.0\.0|\[::\]|\*):[ ]*(3306|5432|27017|6379|11211|9000|9090|9200|9300|8080|8443|15672)\b' || echo "无高危端口对外暴露"

# --- 系统服务 ---
section "运行中的服务"
if command -v systemctl &>/dev/null; then
    systemctl list-units --type=service --state=running --no-pager 2>/dev/null | head -50
elif command -v rc-status &>/dev/null; then
    rc-status 2>/dev/null
else
    echo "无 systemd/openrc"
fi

# --- 云厂商 Agent 检测 ---
section "云厂商 Agent"
CLOUD="unknown"
if [ -d /usr/local/qcloud ] || pgrep -f tat_agent &>/dev/null; then CLOUD="tencent"
elif [ -d /usr/local/aegis ] || pgrep -f AliYunDun &>/dev/null; then CLOUD="aliyun"
elif [ -d /opt/aws ] || pgrep -f amazon-ssm-agent &>/dev/null; then CLOUD="aws"
elif [ -f /var/lib/waagent/WaAgent ] || pgrep -f waagent &>/dev/null; then CLOUD="azure"
elif pgrep -f google-guest-agent &>/dev/null; then CLOUD="gcp"
elif [ -d /usr/local/telescope ] || pgrep -f hostguard &>/dev/null; then CLOUD="huawei"
fi
echo "检测到云环境: $CLOUD"

case "$CLOUD" in
  tencent)
    ps aux | grep -iE '(sap1|barad|sgagent|tmanager|tdsp-|TsysProxy|TsysAgent|tat_agent|tagent)' | grep -v grep | awk '{printf "%-6s %-4s%% %-4s%% %s\n", $2, $3, $4, $11}' || echo "无"
    ;;
  aliyun)
    ps aux | grep -iE '(AliYunDun|aegis|CmsGoAgent|aliyun-service|assist-daemon|cloud-init)' | grep -v grep | awk '{printf "%-6s %-4s%% %-4s%% %s\n", $2, $3, $4, $11}' || echo "无"
    ;;
  aws)
    ps aux | grep -iE '(amazon-ssm-agent|amazon-cloudwatch|cloud-init)' | grep -v grep | awk '{printf "%-6s %-4s%% %-4s%% %s\n", $2, $3, $4, $11}' || echo "无"
    ;;
  azure)
    ps aux | grep -iE '(waagent|OMSAgent|mdsd)' | grep -v grep | awk '{printf "%-6s %-4s%% %-4s%% %s\n", $2, $3, $4, $11}' || echo "无"
    ;;
  gcp)
    ps aux | grep -iE '(google-guest-agent|google-osconfig)' | grep -v grep | awk '{printf "%-6s %-4s%% %-4s%% %s\n", $2, $3, $4, $11}' || echo "无"
    ;;
  huawei)
    ps aux | grep -iE '(telescope|hostguard|hostwatch|uvp-monitor)' | grep -v grep | awk '{printf "%-6s %-4s%% %-4s%% %s\n", $2, $3, $4, $11}' || echo "无"
    ;;
  *) echo "非云环境或未识别" ;;
esac

# --- 定时任务 ---
section "定时任务"
echo "--- root crontab ---"
crontab -l 2>/dev/null || echo "无"
echo "--- /etc/cron.d/ ---"
ls /etc/cron.d/ 2>/dev/null || echo "无"

# --- 防火墙 ---
section "防火墙"
if command -v firewall-cmd &>/dev/null; then
    echo "--- firewalld ---"
    firewall-cmd --state 2>/dev/null || echo "未运行"
    firewall-cmd --list-all 2>/dev/null || true
elif command -v ufw &>/dev/null; then
    echo "--- ufw ---"
    ufw status verbose 2>/dev/null || echo "ufw 不可用"
elif command -v iptables &>/dev/null; then
    echo "--- iptables ---"
    iptables -L -n --line-numbers 2>/dev/null | head -40 || echo "需要 root 权限"
else
    echo "未检测到防火墙工具"
fi

# --- 开发环境多版本 ---
section "Node.js 环境"
if [ -d "$HOME/.nvm" ]; then
    echo "--- nvm 已安装版本 ---"
    ls -1 "$HOME/.nvm/versions/node/" 2>/dev/null || echo "无"
    echo "--- nvm 占用空间 ---"
    du -sh "$HOME/.nvm" 2>/dev/null
fi
if command -v node &>/dev/null; then
    echo "当前 node: $(node -v 2>/dev/null)"
    echo "--- 全局 npm 包 ---"
    npm ls -g --depth=0 2>/dev/null || true
fi
if command -v pnpm &>/dev/null; then
    echo "--- 全局 pnpm 包 ---"
    pnpm ls -g 2>/dev/null || true
    echo "--- pnpm store 占用 ---"
    pnpm store path 2>/dev/null | xargs du -sh 2>/dev/null || true
fi

section "Python 环境"
if command -v python3 &>/dev/null; then
    echo "Python: $(python3 --version 2>/dev/null)"
    echo "--- pip 全局包 ---"
    pip3 list --format=columns 2>/dev/null | head -30 || true
fi
if [ -d "$HOME/.pyenv" ]; then
    echo "--- pyenv 已安装版本 ---"
    ls -1 "$HOME/.pyenv/versions/" 2>/dev/null || echo "无"
    echo "--- pyenv 占用空间 ---"
    du -sh "$HOME/.pyenv" 2>/dev/null
fi
if command -v conda &>/dev/null; then
    echo "--- conda 环境 ---"
    conda env list 2>/dev/null
    echo "--- conda 占用空间 ---"
    du -sh "$(conda info --base 2>/dev/null)" 2>/dev/null || true
fi

section "Java 环境"
if command -v java &>/dev/null; then
    echo "Java: $(java -version 2>&1 | head -1)"
    echo "JAVA_HOME: ${JAVA_HOME:-未设置}"
fi
if [ -d "$HOME/.sdkman" ]; then
    echo "--- SDKMAN 已安装版本 ---"
    ls -1 "$HOME/.sdkman/candidates/java/" 2>/dev/null | head -10
    echo "--- SDKMAN 占用空间 ---"
    du -sh "$HOME/.sdkman" 2>/dev/null
fi

section "Go 环境"
if command -v go &>/dev/null; then
    echo "Go: $(go version 2>/dev/null)"
    echo "GOPATH: ${GOPATH:-$(go env GOPATH 2>/dev/null)}"
    echo "--- Go 占用空间 ---"
    du -sh "$(go env GOPATH 2>/dev/null)" 2>/dev/null || true
    du -sh "$(go env GOMODCACHE 2>/dev/null)" 2>/dev/null || true
fi

# --- 自行安装应用探测 ---
section "自行安装的应用 (/usr/local, /opt)"
for d in /usr/local/*/ /opt/*/; do
    [ -d "$d" ] && SIZE=$(du -sh "$d" 2>/dev/null | cut -f1) && [ "$SIZE" != "0" ] && printf "%-45s %s\n" "$d" "$SIZE"
done 2>/dev/null

section "采集完成"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "系统: $(cat /etc/os-release 2>/dev/null | grep '^PRETTY_NAME=' | cut -d= -f2 | tr -d '"')"
echo "包管理: $PKG | 容器: $CTR | 云环境: $CLOUD"
