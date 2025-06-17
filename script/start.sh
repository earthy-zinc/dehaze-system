#!/bin/bash

# 获取当前所在文件夹路径
current_dir="$(pwd)"

# ANSI 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# 日志打印函数
log_info() {
    echo -e "${GREEN}【INFO】$*${NC}"
}

log_warn() {
    echo -e "${YELLOW}【WARN】$*${NC}"
}

log_error() {
    echo -e "${RED}【ERROR】$*${NC}" >&2
}

# Step 1: 使用 Maven 清理并打包所有模块
log_info "开始执行 Maven 打包..."
if ! mvn clean install -DskipTests -T 2C -q; then
    log_error "Maven 打包失败，请检查构建错误"
    exit 1
fi

# 定义要构建的Dockerfile所在目录键值对
declare -A MODULE_MAP=(
    ["pei-gateway"]="pei-gateway"
    ["pei-module-system/pei-module-system-server"]="pei-system"
    ["pei-module-infra/pei-module-infra-server"]="pei-infra"

    ["pei-module-ai/pei-module-ai-server"]="pei-ai"
    ["pei-module-bpm/pei-module-bpm-server"]="pei-bpm"
    ["pei-module-crm/pei-module-crm-server"]="pei-crm"
    ["pei-module-erp/pei-module-erp-server"]="pei-erp"
    ["pei-module-member/pei-module-member-server"]="pei-member"
    ["pei-module-mp/pei-module-mp-server"]="pei-mp"
    ["pei-module-pay/pei-module-pay-server"]="pei-pay"
    ["pei-module-report/pei-module-report-server"]="pei-report"

    ["pei-module-mall/pei-module-product-server"]="pei-mall-product"
    ["pei-module-mall/pei-module-promotion-server"]="pei-mall-promotion"
    ["pei-module-mall/pei-module-statistics-server"]="pei-mall-statistics"
    ["pei-module-mall/pei-module-trade-server"]="pei-mall-trade"
)
# 构建单个模块
build_module() {
    local dir="$1"
    local image_name="$2"

    log_info "构建 Docker 镜像: $image_name"
    local module_path="$current_dir/$dir"

    if ! docker build -t "${image_name}:latest" "$module_path"; then
        log_error "构建 Docker 镜像 ${image_name} 失败"
        return 1
    fi
    log_info "构建成功: $image_name"
}

# 运行单个模块
run_all_with_compose() {
    log_info "使用 docker-compose 启动所有服务"
    cd "$current_dir/script/docker" || exit

    if [ -f "docker-compose.yml" ]; then
        docker compose down > /dev/null 2>&1
        docker compose up -d
        log_info "服务启动完成"
    else
        log_error "未找到 docker-compose.yml 文件"
        return 1
    fi
}

# 主流程：构建 & 启动每个模块
for dir in "${!MODULE_MAP[@]}"; do
    image_name="${MODULE_MAP[$dir]}"
    build_module "$dir" "$image_name" || exit 1
done

run_all_with_compose || exit 1

log_info "✅ 所有模块构建和运行成功！"
