#!/bin/bash
# 安全初始化脚本：ES 系统用户密码 + Alertmanager Basic Auth 凭证
# 适用场景：首次部署 / 重置 ES 数据卷 / 修改 DEHAZE_PASSWORD 后
#
# 前置条件：
#   1. .env 中已配置 DEHAZE_PASSWORD
#   2. ES 容器已启动（docker compose up -d elasticsearch）
#
# 执行：bash scripts/init-security.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source .env

if [ -z "${DEHAZE_PASSWORD:-}" ]; then
  echo "ERROR: .env 中未配置 DEHAZE_PASSWORD"
  exit 1
fi

ES_URL="http://localhost:9200"
ES_USER="elastic"
ES_PASS="$DEHAZE_PASSWORD"

# ---------- 1. 等待 ES 就绪 ----------
echo "等待 Elasticsearch 就绪..."
for i in $(seq 1 60); do
  if curl -s -u "$ES_USER:$ES_PASS" "$ES_URL/_cluster/health" >/dev/null 2>&1; then
    echo "Elasticsearch 已就绪"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "ERROR: Elasticsearch 60s 内未就绪，请检查容器状态：docker logs elasticsearch"
    exit 1
  fi
  sleep 1
done

# ---------- 2. 设置 ES 内置系统用户密码 ----------
# kibana_system / logstash_system 是 ES 内置保留用户，启用 xpack.security 后自动存在但密码为禁用状态
# 仅 elastic 首次启动时由 ELASTIC_PASSWORD 环境变量自动设置密码
# 其他系统用户需用 elastic 账号手动设置密码才能被 kibana/logstash 使用
SYSTEM_USERS=("kibana_system" "logstash_system" "beats_system")

for user in "${SYSTEM_USERS[@]}"; do
  echo "设置 $user 密码..."
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -u "$ES_USER:$ES_PASS" -X POST "$ES_URL/_security/user/$user/_password" \
    -H 'Content-Type: application/json' \
    -d "{\"password\":\"$ES_PASS\"}")

  if [ "$HTTP_CODE" != "200" ]; then
    echo "  WARN: $user 密码设置失败 (HTTP $HTTP_CODE)，可能已设置或用户不存在"
  else
    echo "  OK: $user 密码已设置"
  fi
done

# ---------- 3. 验证系统用户可登录 ----------
echo ""
echo "验证系统用户登录："
for user in "${SYSTEM_USERS[@]}"; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -u "$user:$ES_PASS" "$ES_URL")
  if [ "$CODE" = "200" ]; then
    echo "  OK: $user 登录成功"
  else
    echo "  FAIL: $user 登录失败 (HTTP $CODE)"
  fi
done

# ---------- 4. 生成 Alertmanager Basic Auth 凭证 ----------
echo ""
echo "生成 Alertmanager Basic Auth 凭证..."
HASH=$(docker run --rm httpd:alpine htpasswd -nbB admin "$DEHAZE_PASSWORD" | sed 's/^admin://')

cat > config/alertmanager/web.yml <<EOF
basic_auth_users:
  admin: "$HASH"
EOF
echo "  OK: config/alertmanager/web.yml 已生成（账号 admin / 密码 \$DEHAZE_PASSWORD）"

# ---------- 5. 提示后续操作 ----------
cat <<EOF

安全初始化完成。账号汇总：
  Elasticsearch:
    elastic          (超级用户，k8s/admin 操作用)
    kibana_system    (kibana 连接 ES 用)
    logstash_system  (logstash 连接 ES 用)
    beats_system     (filebeat 连接 ES 用，当前架构 filebeat → logstash，未直接写 ES，预留)
  Alertmanager:
    admin            (Basic Auth，Prometheus 推送告警 / 访问 Web UI)

如修改了 DEHAZE_PASSWORD，重新执行本脚本后重启相关服务：
  docker compose restart elasticsearch
  bash scripts/init-security.sh
  docker compose up -d kibana logstash alertmanager prometheus
EOF
