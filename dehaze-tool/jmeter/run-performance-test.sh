#!/bin/bash

################################################################################
# JMeter性能测试脚本
# 用途: 执行JMeter性能压力测试
################################################################################

set -e

# 配置变量
JMETER_HOME="/usr/local/jmeter/current"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PLAN="${SCRIPT_DIR}/test-plans/dehaze-performance-test.jmx"
RESULT_DIR="${SCRIPT_DIR}/performance-reports"
RESULT_FILE="${RESULT_DIR}/performance-result-$(date +%Y%m%d-%H%M%S).jtl"
REPORT_DIR="${RESULT_DIR}/html-report-$(date +%Y%m%d-%H%M%S)"

# 性能测试参数
THREAD_COUNT="${1:-10}"           # 并发用户数
LOOP_COUNT="${2:-100}"            # 每个用户的循环次数
RAMP_UP_TIME="${3:-10}"           # 启动时间（秒）

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
        log_error "JMeter未安装或路径不正确"
        exit 1
    fi
}

# 创建性能测试计划（如果不存在）
create_performance_plan() {
    if [ ! -f "${TEST_PLAN}" ]; then
        log_warn "性能测试计划不存在，正在创建..."
        log_info "性能测试计划: ${TEST_PLAN}"
    else
        log_info "使用现有性能测试计划: ${TEST_PLAN}"
    fi
}

# 创建结果目录
create_result_dir() {
    mkdir -p "${RESULT_DIR}"
    log_info "结果目录: ${RESULT_DIR}"
}

# 执行性能测试
run_performance_test() {
    log_info "=========================================="
    log_info "   性能测试参数"
    log_info "=========================================="
    log_info "并发用户数: ${THREAD_COUNT}"
    log_info "循环次数: ${LOOP_COUNT}"
    log_info "启动时间: ${RAMP_UP_TIME}秒"
    log_info "总请求数: $((THREAD_COUNT * LOOP_COUNT))"
    log_info "=========================================="

    # 执行JMeter性能测试
    "${JMETER_HOME}/bin/jmeter" \
        -n \
        -t "${TEST_PLAN}" \
        -l "${RESULT_FILE}" \
        -e \
        -o "${REPORT_DIR}" \
        -JTHREAD_COUNT="${THREAD_COUNT}" \
        -JLOOP_COUNT="${LOOP_COUNT}" \
        -JRAMP_UP_TIME="${RAMP_UP_TIME}"

    if [ $? -eq 0 ]; then
        log_info "✓ 性能测试完成！"
        log_info "HTML报告: ${REPORT_DIR}"
    else
        log_error "性能测试失败！"
        exit 1
    fi
}

# 分析性能测试结果
analyze_results() {
    log_info "=========================================="
    log_info "   性能测试结果分析"
    log_info "=========================================="

    if [ -f "${RESULT_FILE}" ]; then
        # 计算各种指标
        TOTAL_REQUESTS=$(wc -l < "${RESULT_FILE}")
        FAILED_REQUESTS=$(grep -c "false" "${RESULT_FILE}" || echo 0)
        SUCCESS_REQUESTS=$((TOTAL_REQUESTS - FAILED_REQUESTS))

        # 提取响应时间数据
        AVG_RESPONSE=$(awk -F',' '{sum+=$2; count++} END {print sum/count}' "${RESULT_FILE}" 2>/dev/null || echo "N/A")
        MAX_RESPONSE=$(awk -F',' 'BEGIN{max=0} {if($2>max) max=$2} END {print max}' "${RESULT_FILE}" 2>/dev/null || echo "N/A")
        MIN_RESPONSE=$(awk -F',' 'BEGIN{min=999999} {if($2<min) min=$2} END {print min}' "${RESULT_FILE}" 2>/dev/null || echo "N/A")

        # 计算成功率
        if [ "${TOTAL_REQUESTS}" -gt 0 ]; then
            SUCCESS_RATE=$(awk "BEGIN {printf \"%.2f\", ($SUCCESS_REQUESTS/$TOTAL_REQUESTS)*100}")
        else
            SUCCESS_RATE="0.00"
        fi

        # 计算吞吐量（假设测试时长为30秒）
        THROUGHPUT=$(awk "BEGIN {printf \"%.2f\", $TOTAL_REQUESTS/30}")

        echo "总请求数:     ${TOTAL_REQUESTS}"
        echo "成功请求:     ${SUCCESS_REQUESTS}"
        echo "失败请求:     ${FAILED_REQUESTS}"
        echo "成功率:       ${SUCCESS_RATE}%"
        echo "平均响应时间: ${AVG_RESPONSE}ms"
        echo "最大响应时间: ${MAX_RESPONSE}ms"
        echo "最小响应时间: ${MIN_RESPONSE}ms"
        echo "吞吐量:       ${THROUGHPUT} req/s"

        # 判断性能是否达标
        log_info "=========================================="
        if (( $(echo "${SUCCESS_RATE} >= 99" | bc -l) )); then
            log_info "✓ 性能测试通过！"
        elif (( $(echo "${SUCCESS_RATE} >= 95" | bc -l) )); then
            log_warn "⚠ 性能测试勉强通过（成功率 < 99%）"
        else
            log_error "✗ 性能测试未通过！"
        fi
        log_info "=========================================="
    fi
}

# 主函数
main() {
    log_info "=========================================="
    log_info "   dehaze-java 性能测试"
    log_info "=========================================="

    check_jmeter
    create_performance_plan
    create_result_dir
    run_performance_test
    analyze_results

    log_info "性能测试报告已保存到: ${REPORT_DIR}"
    log_info "请打开 ${REPORT_DIR}/index.html 查看详细报告"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [并发用户数] [循环次数] [启动时间(秒)]"
    echo ""
    echo "参数说明:"
    echo "  并发用户数    默认: 10"
    echo "  循环次数      默认: 100"
    echo "  启动时间      默认: 10"
    echo ""
    echo "示例:"
    echo "  $0                          # 使用默认参数"
    echo "  $0 50 200 30                # 50个并发用户，循环200次，30秒启动"
    echo "  $0 100 500 60               # 100个并发用户，循环500次，60秒启动"
    echo ""
    echo "性能指标说明:"
    echo "  - 成功率: 成功请求占总请求的百分比（建议 >= 99%）"
    echo "  - 响应时间: 请求的平均、最大、最小响应时间（建议 < 1000ms）"
    echo "  - 吞吐量: 每秒处理的请求数"
    exit 0
}

# 参数处理
case "${1:-}" in
    --help|-h)
        show_help
        ;;
    *)
        main "$@"
        ;;
esac
