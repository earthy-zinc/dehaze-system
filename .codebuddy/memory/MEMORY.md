# DehazeSystem 项目记忆

## 项目约定

每个项目的启动方式请务必参考项目根目录的 `README.md` 文件，并严格按照其中的说明进行操作。

### Windows 开发环境注意事项
- **必须用 `127.0.0.1` 而非 `localhost`**：
- Windows 上 `localhost` 解析为 IPv6 `::1`（优先）+ IPv4 `127.0.0.1`。
- Docker Desktop 端口映射只绑定 IPv4。
- Python `socket.create_connection` 串行尝试地址，先连 `::1` 失败后等 TCP SYN 超时（约 21 秒）才回退 IPv4。
- Go 不受影响（Happy Eyeballs 并行尝试）
- Java 也不受影响（默认偏好 IPv4）

### 环境变量管理
- 统一密码环境变量 `DEHAZE_PASSWORD=12345678` 管理所有基础设施密码
- 三个后端（Go/Java/Python）共享相同 JWT 签名密钥，确保 JWT token 可互认
- `.env` 文件位于 monorepo 根目录 `DehazeSystem/.env`

### 调试辅助脚本
- **调试脚本**：`scripts/debug_helper.py <command>` - 包含登录逻辑 + API 调试
  - `compare /api/v1/xxx [METHOD] [BODY]` - 三端对比同一 API 确保一致性
  - `curl <backend> /api/v1/xxx [METHOD] [BODY]` - 单端请求
- 开发账号：admin / 12345678

### 后端生命周期管理
- **统一脚本**：`scripts/run.py <command> [args...]`
- 命令：`run|stop|restart <svc>[,svc...]|all`、`ps`、`logs <svc> [lines]`
- 支持别名（`go`/`python`/`java`）和 `all`，PID 文件落各服务目录（`.<svc>.pid`）

### pnpm workspace 依赖链接
- **必须用 `workspace:*` 而非 `link:`** 链接 workspace 内部包（如 `dehaze-sdk-js`）
- `link:` 只创建裸 symlink，不走 pnpm peer 依赖解析，导致 TypeScript `exports` 类型解析失败（报 7016/2305 错误）
- `workspace:*` 正确解析 peer 依赖，TypeScript 能正常加载 `.d.ts` 类型声明
- `pnpm-workspace.yaml` 已声明所有子项目为 workspace 成员

### Java 数据权限机制
- `MyDataPermissionHandler` 通过 `DataPermissionInterceptor` 自动在 SQL 上附加数据范围过滤
- `SysDeptMapper.selectList` 上有 `@DataPermission(deptIdColumnName = "id")`，部门查询受数据权限控制
- `SecurityUtils.isRoot()` 仅对 ROOT 角色码返回 true，ADMIN 角色不跳过数据权限
- `DataScopeEnum`: ALL=0, DEPT_AND_SUB=1, DEPT=2, SELF=3（值越小范围越大）
- `getUserAuthInfo` 有 `@Cacheable("user:auth")` 缓存，改角色 data_scope 后需清缓存或重新登录
- `SysRoleServiceImpl.saveRole` 已加 `@CacheEvict("user:auth")` 确保角色变更时缓存自动失效

### 算法生命周期状态
- 算法有 6 种状态：0=草稿, 1=测试中, 2=待审核, 3=已发布(终态), 4=已停用(终态), 5=已归档(终态)
- 终态(3/4/5)不允许通过 `PUT /api/v1/algorithms/{id}/status` 修改状态，后端返回 400 "终态算法不允许修改状态"
- 前端列表页不能用 `el-switch`/`Switch`（二态 0/1）展示算法状态，会导致无限循环：switch 归一化触发 @change → API 失败 → catch 回滚修改 row.status → 再次触发 @change
- 正确做法：列表页用 Tag 展示 6 种生命周期状态；编辑表单用 el-select/Select 下拉框展示 6 种状态
- 新增算法默认状态应为 0（草稿），不是 1（测试中）

### 文件存储双轨制
- **用户上传文件**：存入 MinIO，`objectName` 格式 `upload/yyyyMMdd/md5.ext`，`url` 指向后端 download API（如 `http://127.0.0.1:8989/api/v1/files/download/...`）
- **数据集文件**：由 nginx-dataset 容器（端口 9000）直服，不上传 MinIO；`objectName` 为相对路径（如 `Dense-Haze/clean/14_GT.png`），`url` 指向 nginx（如 `http://127.0.0.1:9000/...`）
- 数据集初始化由 Java `InitFile` 组件完成（`file.init=true` 时扫描磁盘），仅创建 DB 记录不上传 MinIO
- 三端 download 接口统一逻辑：先查 DB，若 `url` 不以当前后端 download baseUrl 开头则 302 重定向到该 URL（处理 nginx 直服的数据集文件）

以下服务已经启动，请勿重复启动，如需重启，请告知用户，切勿私自重启

- **Java**: 8989 (Spring Boot devtools 热重载)
- **Go**: 8990 (`go run` 开发模式)
- **Python**: 8991 (`uvicorn --reload` 开发模式)
- Docker 容器: MySQL 3306, Redis 6379, MongoDB 27017, MinIO 9110(API)/9190(Console), PostgreSQL 5432, RabbitMQ 5672/15672
