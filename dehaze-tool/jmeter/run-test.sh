#!/bin/bash

################################################################################
# JMeter测试执行脚本
# 用途: 执行JMeter测试计划并生成报告
################################################################################

set -e

# 配置变量
JMETER_HOME="/usr/local/jmeter/current"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PLAN="${SCRIPT_DIR}/test-plans/dehaze-api-test.jmx"
RESULT_DIR="${SCRIPT_DIR}/reports"
RESULT_FILE="${RESULT_DIR}/result-$(date +%Y%m%d-%H%M%S).jtl"
REPORT_DIR="${RESULT_DIR}/html-report-$(date +%Y%m%d-%H%M%S)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查JMeter是否安装
check_jmeter() {
    if [ ! -f "${JMETER_HOME}/bin/jmeter" ]; then
        log_error "JMeter未安装或路径不正确: ${JMETER_HOME}"
        log_error "请先运行 ./install-jmeter.sh 安装JMeter"
        exit 1
    fi
    log_info "JMeter已安装: ${JMETER_HOME}"
}

# 检查测试计划文件
check_test_plan() {
    if [ ! -f "${TEST_PLAN}" ]; then
        log_error "测试计划文件不存在: ${TEST_PLAN}"
        exit 1
    fi
    log_info "测试计划文件: ${TEST_PLAN}"
}

# 创建结果目录
create_result_dir() {
    mkdir -p "${RESULT_DIR}"
    log_info "结果目录: ${RESULT_DIR}"
}

# 执行测试
run_test() {
    log_info "开始执行测试..."
    log_info "结果文件: ${RESULT_FILE}"

    # 加载配置文件
    if [ -f "${SCRIPT_DIR}/configs/test-env.properties" ]; then
        log_info "加载环境配置: ${SCRIPT_DIR}/configs/test-env.properties"
        source "${SCRIPT_DIR}/configs/test-env.properties"
    fi

    # 执行JMeter非GUI测试
    "${JMETER_HOME}/bin/jmeter" -n -t "${TEST_PLAN}" -l "${RESULT_FILE}" -e -o "${REPORT_DIR}"

    if [ $? -eq 0 ]; then
        log_info "✓ 测试执行成功！"
        log_info "HTML报告目录: ${REPORT_DIR}"
    else
        log_error "测试执行失败！"
        exit 1
    fi
}

# 显示测试结果摘要
show_summary() {
    log_info "=========================================="
    log_info "   测试完成"
    log_info "=========================================="
    log_info "测试计划: ${TEST_PLAN}"
    log_info "结果文件: ${RESULT_FILE}"
    log_info "HTML报告: ${REPORT_DIR}"
    log_info "=========================================="

    # 检查是否有失败的请求
    if [ -f "${RESULT_FILE}" ]; then
        FAILED_COUNT=$(grep -c "false" "${RESULT_FILE}" || echo 0)
        TOTAL_COUNT=$(wc -l < "${RESULT_FILE}")
        SUCCESS_COUNT=$((TOTAL_COUNT - FAILED_COUNT))

        log_info "总请求数: ${TOTAL_COUNT}"
        log_info "成功请求: ${SUCCESS_COUNT}"
        log_info "失败请求: ${FAILED_COUNT}"

        if [ "${FAILED_COUNT}" -gt 0 ]; then
            log_warn "存在失败请求，请查看报告详情"
        fi
    fi
}

# 主函数
main() {
    log_info "=========================================="
    log_info "   dehaze-java 接口测试"
    log_info "=========================================="

    check_jmeter
    check_test_plan
    create_result_dir
    run_test
    show_summary
}

# 参数处理
case "${1:-}" in
    --help|-h)
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  --help, -h       显示帮助信息"
        echo "  --no-report      不生成HTML报告"
        echo ""
        echo "示例:"
        echo "  $0              执行测试并生成报告"
        echo "  $0 --no-report  执行测试不生成报告"
        exit 0
        ;;
    --no-report)
        log_info "不生成HTML报告模式"
        JMETER_CMD="${JMETER_HOME}/bin/jmeter -n -t ${TEST_PLAN} -l ${RESULT_FILE}"
        ${JMETER_CMD}
        ;;
    *)
        main
        ;;
esac
