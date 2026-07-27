#!/bin/bash
# 在所有 schema/*.sql 执行完毕后，按文件名顺序加载 data/*.sql 初始化数据
for f in /docker-init-data/*.sql; do
  [ -f "$f" ] || continue
  mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" "${MYSQL_DATABASE}" < "$f"
done
