#!/bin/bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 从 .env 加载密码，未设置时回退默认值
if [ -f "$ROOT/.env" ]; then
  DEHAZE_PASSWORD=$(grep -E '^DEHAZE_PASSWORD=' "$ROOT/.env" | cut -d'=' -f2- | tr -d '\r\n')
fi
DEHAZE_PASSWORD="${DEHAZE_PASSWORD:-12345678}"

MYSQL_CONTAINER="${MYSQL_CONTAINER:-mysql}"
REDIS_CONTAINER="${REDIS_CONTAINER:-redis}"

docker exec "$MYSQL_CONTAINER" mysql -uroot -p"$DEHAZE_PASSWORD" -e "DROP DATABASE IF EXISTS dehaze; CREATE DATABASE dehaze CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci; DROP DATABASE IF EXISTS dehaze_test; CREATE DATABASE dehaze_test CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"

for f in "$ROOT"/config/sql/schema/sys_*.sql; do
  echo "schema: $(basename "$f")"
  docker exec -i "$MYSQL_CONTAINER" mysql -uroot -p"$DEHAZE_PASSWORD" dehaze < "$f"
  docker exec -i "$MYSQL_CONTAINER" mysql -uroot -p"$DEHAZE_PASSWORD" dehaze_test < "$f"
done

for f in "$ROOT"/config/sql/data/sys_*.sql; do
  echo "data: $(basename "$f")"
  docker exec -i "$MYSQL_CONTAINER" mysql -uroot -p"$DEHAZE_PASSWORD" dehaze < "$f"
  docker exec -i "$MYSQL_CONTAINER" mysql -uroot -p"$DEHAZE_PASSWORD" dehaze_test < "$f"
done

for pattern in "msg:unread:*" "role:perms:*" "user:auth:*" "session:*"; do
  docker exec "$REDIS_CONTAINER" redis-cli -a "$DEHAZE_PASSWORD" --scan --pattern "$pattern" | while read -r key; do
    docker exec "$REDIS_CONTAINER" redis-cli -a "$DEHAZE_PASSWORD" DEL "$key"
  done
done

echo "Done"
