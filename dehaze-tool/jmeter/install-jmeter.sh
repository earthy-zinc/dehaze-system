#!/bin/bash

################################################################################
# JMeter安装脚本 - 适用于Linux TencentOS
# 版本: 1.0
# 用途: 自动下载、安装和配置Apache JMeter
################################################################################

set -e  # 遇到错误立即退出

# 配置变量
JMETER_VERSION="5.6.3"
JMETER_INSTALL_DIR="/usr/local/jmeter"
JMETER_DOWNLOAD_URL="https://downloads.apache.org//jmeter/binaries/apache-jmeter-${JMETER_VERSION}.tgz"
JMETER_BACKUP_URL="https://mirrors.tuna.tsinghua.edu.cn/apache/jmeter/binaries/apache-jmeter-${JMETER_VERSION}.tgz"
TEMP_DIR="/tmp/jmeter-install"
JMETER_HEAP_SIZE="-Xms1g -Xmx2g"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "请使用root权限或sudo运行此脚本"
        exit 1
    fi
}

# 检查系统信息
check_system() {
    log_info "检查系统信息..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log_info "操作系统: $NAME $VERSION"
    else
        log_warn "无法识别操作系统版本"
    fi

    # 检查Java是否已安装
    if ! command -v java &> /dev/null; then
        log_error "未检测到Java环境，请先安装Java 8或更高版本"
        exit 1
    else
        JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
        log_info "Java版本: $JAVA_VERSION"
    fi
}

# 下载JMeter
download_jmeter() {
    log_info "开始下载JMeter ${JMETER_VERSION}..."
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"

    # 尝试从主镜像下载
    if wget -O "apache-jmeter-${JMETER_VERSION}.tgz" "$JMETER_DOWNLOAD_URL" 2>&1 | grep -q "saved"; then
        log_info "从Apache镜像下载成功"
    else
        log_warn "主镜像下载失败，尝试使用清华大学镜像..."
        wget -O "apache-jmeter-${JMETER_VERSION}.tgz" "$JMETER_BACKUP_URL"
        if [ $? -eq 0 ]; then
            log_info "从清华大学镜像下载成功"
        else
            log_error "所有镜像下载均失败"
            exit 1
        fi
    fi
}

# 安装JMeter
install_jmeter() {
    log_info "开始安装JMeter..."

    # 创建安装目录
    mkdir -p "$JMETER_INSTALL_DIR"

    # 解压文件
    log_info "解压JMeter到 ${JMETER_INSTALL_DIR}..."
    tar -xzf "$TEMP_DIR/apache-jmeter-${JMETER_VERSION}.tgz" -C "$JMETER_INSTALL_DIR"

    # 创建软链接
    ln -sf "$JMETER_INSTALL_DIR/apache-jmeter-${JMETER_VERSION}" "$JMETER_INSTALL_DIR/current"

    # 设置JVM堆内存
    log_info "配置JVM堆内存参数..."
    sed -i "s|^HEAP=\".*\"|HEAP=\"${JMETER_HEAP_SIZE}\"|g" "$JMETER_INSTALL_DIR/current/bin/jmeter"

    # 设置执行权限
    chmod +x "$JMETER_INSTALL_DIR/current/bin/jmeter"
    chmod +x "$JMETER_INSTALL_DIR/current/bin/jmeter.sh"
}

# 配置环境变量
configure_env() {
    log_info "配置环境变量..."

    cat > /etc/profile.d/jmeter.sh << 'EOF'
# JMeter Environment Variables
export JMETER_HOME=/usr/local/jmeter/current
export PATH=$JMETER_HOME/bin:$PATH
EOF

    chmod +x /etc/profile.d/jmeter.sh
    log_info "环境变量配置文件已创建: /etc/profile.d/jmeter.sh"
    log_info "请运行 'source /etc/profile.d/jmeter.sh' 使环境变量生效"
}

# 验证安装
verify_installation() {
    log_info "验证JMeter安装..."

    if [ -f "$JMETER_INSTALL_DIR/current/bin/jmeter" ]; then
        log_info "JMeter文件存在"
        "$JMETER_INSTALL_DIR/current/bin/jmeter" --version
        if [ $? -eq 0 ]; then
            log_info "✓ JMeter安装成功！"
            return 0
        else
            log_error "JMeter执行失败"
            return 1
        fi
    else
        log_error "JMeter文件不存在"
        return 1
    fi
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."
    rm -rf "$TEMP_DIR"
}

# 主函数
main() {
    log_info "=========================================="
    log_info "   JMeter自动安装脚本"
    log_info "   版本: ${JMETER_VERSION}"
    log_info "=========================================="

    check_root
    check_system
    download_jmeter
    install_jmeter
    configure_env
    verify_installation
    cleanup

    log_info "=========================================="
    log_info "   安装完成！"
    log_info "=========================================="
    log_info "下一步操作:"
    log_info "1. 使环境变量生效: source /etc/profile.d/jmeter.sh"
    log_info "2. 验证安装: jmeter --version"
    log_info "3. 运行JMeter GUI: jmeter"
    log_info "4. 运行命令行测试: jmeter -n -t test-plan.jmx -l result.jtl"
    log_info "=========================================="
}

# 执行主函数
main "$@"
