#!/bin/bash

################################################################################
# JMeter快速启动脚本
# 用途: 快速检查环境并启动测试
################################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 显示横幅
show_banner() {
    echo -e "${GREEN}"
    echo "=========================================="
    echo "   dehaze-java JMeter 测试快速启动"
    echo "=========================================="
    echo -e "${NC}"
}

# 检查Java环境
check_java() {
    log_step "检查Java环境..."

    if ! command -v java &> /dev/null; then
        log_error "Java未安装！"
        echo "请先安装Java 8或更高版本："
        echo "  TencentOS/CentOS: sudo yum install -y java-17-openjdk java-17-openjdk-devel"
        echo "  Ubuntu: sudo apt-get install -y openjdk-17-jdk"
        return 1
    fi

    JAVA_VERSION=$(java -version 2>&1 | head -n 1)
    log_info "✓ ${JAVA_VERSION}"
    return 0
}

# 检查JMeter是否安装
check_jmeter() {
    log_step "检查JMeter安装..."

    if [ -f "/usr/local/jmeter/current/bin/jmeter" ]; then
        JMETER_VERSION=$(/usr/local/jmeter/current/bin/jmeter --version 2>&1 | head -n 1)
        log_info "✓ ${JMETER_VERSION}"
        return 0
    else {
        log_warn "JMeter未安装"
        return 1
    fi
}

# 检查dehaze-java服务是否运行
check_service() {
    log_step "检查dehaze-java服务..."

    # 检查端口8080是否开放
    if command -v nc &> /dev/null; then
        if nc -z localhost 8080 2>/dev/null; then
            log_info "✓ dehaze-java服务正在运行（端口8080）"
            return 0
        else
            log_warn "dehaze-java服务未运行或端口8080未开放"
            return 1
        fi
    elif command -v curl &> /dev/null; then
        if curl -s http://localhost:8080/actuator/health &> /dev/null; then
            log_info "✓ dehaze-java服务正在运行"
            return 0
        else
            log_warn "dehaze-java服务未运行"
            return 1
        fi
    else
        log_warn "无法检查服务状态（缺少nc和curl命令）"
        return 1
    fi
}

# 检查测试文件
check_test_files() {
    log_step "检查测试文件..."

    local missing_files=()

    if [ ! -f "configs/test-env.properties" ]; then
        missing_files+=("configs/test-env.properties")
    fi

    if [ ! -f "test-plans/dehaze-api-test.jmx" ]; then
        missing_files+=("test-plans/dehaze-api-test.jmx")
    fi

    if [ ${#missing_files[@]} -gt 0 ]; then
        log_error "缺少以下测试文件："
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        return 1
    fi

    log_info "✓ 测试文件完整"
    return 0
}

# 创建测试数据
prepare_test_data() {
    log_step "准备测试数据..."

    # 创建测试文件
    if [ ! -f "/tmp/test-file.txt" ]; then
        echo "This is a test file for JMeter upload test." > /tmp/test-file.txt
        log_info "✓ 已创建测试文件: /tmp/test-file.txt"
    else
        log_info "✓ 测试文件已存在"
    fi

    # 创建报告目录
    mkdir -p reports
    mkdir -p performance-reports
    log_info "✓ 已创建报告目录"
}

# 显示菜单
show_menu() {
    echo ""
    echo "=========================================="
    echo "请选择操作："
    echo "=========================================="
    echo "  1. 安装JMeter"
    echo "  2. 运行接口测试"
    echo "  3. 运行性能测试"
    echo "  4. 使用GUI模式调试"
    echo "  5. 查看测试报告"
    echo "  6. 环境健康检查"
    echo "  7. 退出"
    echo "=========================================="
    echo -n "请输入选项 [1-7]: "
}

# 安装JMeter
install_jmeter() {
    log_step "开始安装JMeter..."

    if [ ! -f "install-jmeter.sh" ]; then
        log_error "找不到install-jmeter.sh安装脚本"
        return 1
    fi

    chmod +x install-jmeter.sh
    sudo ./install-jmeter.sh

    if [ $? -eq 0 ]; then
        log_info "✓ JMeter安装成功"
        source /etc/profile.d/jmeter.sh
    else
        log_error "JMeter安装失败"
        return 1
    fi
}

# 运行接口测试
run_api_test() {
    log_step "运行接口测试..."

    if [ ! -f "run-test.sh" ]; then
        log_error "找不到run-test.sh脚本"
        return 1
    fi

    chmod +x run-test.sh
    ./run-test.sh
}

# 运行性能测试
run_performance_test() {
    log_step "运行性能测试..."

    if [ ! -f "run-performance-test.sh" ]; then
        log_error "找不到run-performance-test.sh脚本"
        return 1
    fi

    echo ""
    echo "请输入性能测试参数（直接回车使用默认值）："
    read -p "并发用户数 [默认: 10]: " thread_count
    read -p "循环次数 [默认: 100]: " loop_count
    read -p "启动时间-秒 [默认: 10]: " ramp_up

    thread_count=${thread_count:-10}
    loop_count=${loop_count:-100}
    ramp_up=${ramp_up:-10}

    chmod +x run-performance-test.sh
    ./run-performance-test.sh "$thread_count" "$loop_count" "$ramp_up"
}

# 使用GUI模式调试
run_gui_mode() {
    log_step "启动JMeter GUI模式..."

    if ! command -v jmeter &> /dev/null; then
        log_error "JMeter未安装或未配置环境变量"
        echo "请先安装JMeter或运行: source /etc/profile.d/jmeter.sh"
        return 1
    fi

    jmeter -t test-plans/dehaze-api-test.jmx
}

# 查看测试报告
view_reports() {
    log_step "查看测试报告..."

    echo ""
    echo "可用的测试报告："
    echo "=========================================="

    if [ -d "reports" ]; then
        echo "接口测试报告:"
        ls -lt reports/html-report-* 2>/dev/null | head -5 | while read line; do
            echo "  $line"
        done
    fi

    if [ -d "performance-reports" ]; then
        echo ""
        echo "性能测试报告:"
        ls -lt performance-reports/html-report-* 2>/dev/null | head -5 | while read line; do
            echo "  $line"
        done
    fi

    echo "=========================================="
    echo ""

    # 查找最新的报告
    local latest_report=$(find . -name "index.html" -path "*/html-report-*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")

    if [ -n "$latest_report" ]; then
        echo "最新报告: $latest_report"
        read -p "是否在浏览器中打开? [y/N]: " open_browser

        if [ "$open_browser" = "y" ] || [ "$open_browser" = "Y" ]; then
            if command -v xdg-open &> /dev/null; then
                xdg-open "$latest_report"
            elif command -v open &> /dev/null; then
                open "$latest_report"
            else
                log_warn "无法自动打开浏览器，请手动打开: $latest_report"
            fi
        fi
    else
        log_warn "未找到测试报告，请先运行测试"
    fi
}

# 环境健康检查
health_check() {
    log_step "执行环境健康检查..."
    echo ""

    local checks_passed=0
    local checks_failed=0

    # 检查Java
    if check_java; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi

    echo ""

    # 检查JMeter
    if check_jmeter; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi

    echo ""

    # 检查服务
    if check_service; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi

    echo ""

    # 检查测试文件
    if check_test_files; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi

    echo ""
    echo "=========================================="
    echo "健康检查结果："
    echo "=========================================="
    echo "  通过: ${checks_passed}"
    echo "  失败: ${checks_failed}"
    echo "=========================================="

    if [ $checks_failed -eq 0 ]; then
        log_info "✓ 所有检查通过，可以开始测试"
    else
        log_warn "存在 $checks_failed 个问题，请解决后重试"
    fi
}

# 主函数
main() {
    show_banner

    # 检查是否在正确的目录
    if [ ! -f "install-jmeter.sh" ]; then
        log_error "请在jmeter目录下运行此脚本"
        echo "当前目录: $(pwd)"
        echo "正确目录: /data/workspace/dehaze-system/jmeter"
        exit 1
    fi

    # 准备测试数据
    prepare_test_data

    # 主循环
    while true; do
        show_menu
        read -r choice

        case "$choice" in
            1)
                install_jmeter
                ;;
            2)
                run_api_test
                ;;
            3)
                run_performance_test
                ;;
            4)
                run_gui_mode
                ;;
            5)
                view_reports
                ;;
            6)
                health_check
                ;;
            7)
                log_info "退出"
                exit 0
                ;;
            *)
                log_error "无效选项，请重新输入"
                ;;
        esac

        echo ""
        read -p "按回车键继续..."
        clear
        show_banner
    done
}

# 检查命令行参数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h    显示帮助信息"
    echo "  --check       仅执行环境检查"
    echo "  --test        直接运行接口测试"
    echo "  --perf        直接运行性能测试"
    echo ""
    echo "示例:"
    echo "  $0              # 启动交互式菜单"
    echo "  $0 --check      # 环境检查"
    echo "  $0 --test       # 运行接口测试"
    exit 0
fi

# 处理命令行参数
case "${1:-}" in
    --check)
        health_check
        ;;
    --test)
        run_api_test
        ;;
    --perf)
        run_performance_test
        ;;
    *)
        main
        ;;
esac
