# 测试数据库 SQL 文件说明

## 文件来源

| 文件 | 来源 | 说明 |
|------|------|------|
| `init.sql` | 本目录（测试专属） | 创建 `dehaze_test` 数据库 |
| `schema.sql` | **Maven 自动复制** from `config/sql/schema.sql` | 建表语句，构建时由 `maven-resources-plugin` 复制到 classpath |
| `data.sql` | **Maven 自动复制** from `config/sql/data.sql` | 初始化数据，同上 |
| `h2/*.sql` | 本目录（H2 专属） | H2 内存数据库方言的建表语句，不与 MySQL schema.sql 共用 |

## 架构说明

- **生产/Docker**: `config/sql/schema.sql` + `config/sql/data.sql` 通过 Docker `docker-entrypoint-initdb.d` 自动执行
- **MySQL 测试**: `application-test.yml` 加载 `init.sql` → `schema.sql` → `data.sql`（后两者由 Maven 复制）
- **H2 测试**: `application-test-h2.yml` 加载 `h2/*.sql`（建表）→ `data.sql`（数据，由 Maven 复制）
- **单一数据源**: `schema.sql` 和 `data.sql` 只在 `config/sql/` 维护，无需手动同步测试副本

## IDE 注意事项

IntelliJ IDEA 需执行一次 `mvn process-test-resources` 以触发 SQL 文件复制，或启用 "Delegate IDE build/run actions to Maven"。
