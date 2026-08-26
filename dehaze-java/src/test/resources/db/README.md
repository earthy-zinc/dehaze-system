# 测试数据库 SQL 文件说明

## 文件来源

| 文件 | 来源 | 说明 |
|------|------|------|
| `schema/*.sql` | **Maven 自动复制** from `config/sql/schema/` | 按表名拆分的建表语句，每张表一个文件，文件头部含设计思路注释 |
| `data/*.sql` | **Maven 自动复制** from `config/sql/data/` | 按表名拆分的初始化数据，文件名与 `schema/` 一一对应 |

测试统一使用真实 MySQL 测试库 `dehaze_test`（与开发同实例，零方言漂移；
2026-08-23 决策废弃 H2 内存库方案，原因同 dehaze-python：DDL/SQL 语义双层漂移）：

- `application-test.yml` 通过 `createDatabaseIfNotExist=true` 首次连接自动建库，
  再由 `spring.sql.init` 加载 `schema/sys_*.sql`（DROP TABLE IF EXISTS，幂等全量重建）+
  `data/sys_*.sql`（种子数据）；**通配必须限定 `sys_` 前缀**——`schema/` 中的
  `xxl_job.sql` 是调度中心库引导脚本（`DROP DATABASE xxl_job` + `USE xxl_job`），
  仅供 Docker 首次初始化使用，混入会切走连接当前库并误删共享实例的 xxl_job 库
- 凭证来自仓库根目录 `.env` 的按基础设施分区变量：MySQL 用 `MYSQL_HOST` / `MYSQL_PORT` /
  `MYSQL_USERNAME` / `MYSQL_PASSWORD`，Redis/MongoDB 同理（`REDIS_*` / `MONGODB_*`）
- 与 dehaze-python 测试共用同一 `dehaze_test` 库，**勿并行运行两端的数据库测试**
  （Python conftest 每次运行会 DROP + CREATE DATABASE 全量重置）

## 架构说明

- **生产/Docker**: `config/sql/schema/` 目录挂载到 `docker-entrypoint-initdb.d/` 先执行建表，`load-data.sh` 脚本随后执行 `config/sql/data/` 目录下所有数据文件
- **MySQL 测试**: `application-test.yml` 连接 `dehaze_test` 并加载 `schema/*.sql`（通配符）→ `data/*.sql`（通配符，均由 Maven 复制）
- **单一数据源**: `config/sql/schema/` 和 `config/sql/data/` 目录下每张表一个 `.sql` 文件，Docker 和测试均直接引用，无需拼接

## SQL 文件维护

- 修改表结构时，编辑 `config/sql/schema/{table_name}.sql`
- 修改初始化数据时，编辑 `config/sql/data/{table_name}.sql`

## IDE 注意事项

IntelliJ IDEA 需执行一次 `mvn process-test-resources` 以触发 SQL 文件复制，或启用 "Delegate IDE build/run actions to Maven"。
