# 测试数据库 SQL 文件说明

## 文件来源

| 文件 | 来源 | 说明 |
|------|------|------|
| `init.sql` | 本目录（测试专属） | 创建 `dehaze_test` 数据库 |
| `schema/*.sql` | **Maven 自动复制** from `config/sql/schema/` | 按表名拆分的建表语句，每张表一个文件，文件头部含设计思路注释 |
| `data/*.sql` | **Maven 自动复制** from `config/sql/data/` | 按表名拆分的初始化数据，文件名与 `schema/` 一一对应 |
| `h2/*.sql` | 本目录（H2 专属） | H2 内存数据库方言的建表语句，不与 MySQL schema/ 共用 |

## 架构说明

- **生产/Docker**: `config/sql/schema/` 目录挂载到 `docker-entrypoint-initdb.d/` 先执行建表，`load-data.sh` 脚本随后执行 `config/sql/data/` 目录下所有数据文件
- **MySQL 测试**: `application-test.yml` 加载 `init.sql` → `schema/*.sql`（通配符）→ `data/*.sql`（通配符，后两者由 Maven 复制）
- **H2 测试**: `application-test-h2.yml` 加载 `h2/*.sql`（建表）→ `data/*.sql`（数据，由 Maven 复制）
- **单一数据源**: `config/sql/schema/` 和 `config/sql/data/` 目录下每张表一个 `.sql` 文件，Docker 和测试均直接引用，无需拼接

## SQL 文件维护

- 修改表结构时，编辑 `config/sql/schema/{table_name}.sql`
- 修改初始化数据时，编辑 `config/sql/data/{table_name}.sql`

## IDE 注意事项

IntelliJ IDEA 需执行一次 `mvn process-test-resources` 以触发 SQL 文件复制，或启用 "Delegate IDE build/run actions to Maven"。
