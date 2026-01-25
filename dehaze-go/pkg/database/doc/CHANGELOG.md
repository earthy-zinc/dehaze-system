# Changelog

## [2.0.0] - 架构重构版本

### 🎉 重大变更

#### 架构重构
- **接口抽象**：定义 `DBer` 接口，统一所有数据库操作
- **工厂模式**：实现 `Factory` 接口和驱动注册机制
- **模块化**：每个数据库独立子目录实现（`mysql/`、`postgres/`、`sqlite/`）
- **全局管理**：根目录统一管理初始化、实例获取、优雅关闭

#### 新增特性
- ✨ **主从分离支持**：MySQL/PostgreSQL 支持一主多从，随机负载均衡
- ✨ **配置统一**：通用 `Config` 结构体，包含所有数据库专属配置段
- ✨ **日志统一**：`GormLogger` 适配器，统一对接 `pkg/logger`
- ✨ **工具函数**：DSN密码隐藏、连接池配置、Gorm配置生成等
- ✨ **回调自动填充**：保留 `create_by`/`update_by` 自动填充功能
- ✨ **配置迁移**：`MigrateFromOldConfig()` 自动转换旧配置格式
- ✨ **连接验证**：启动时自动验证数据库连接可用性
- ✨ **慢查询监控**：可配置阈值，自动记录慢查询
- ✨ **密码脱敏**：日志输出时自动隐藏DSN中的密码

### 📁 新增文件

#### 核心文件
- `database.go` - DBer/Factory 接口定义，全局管理方法
- `config.go` - 通用配置结构体，参数校验
- `logger.go` - Gorm 日志适配器
- `utils.go` - 通用工具函数
- `callback.go` - Gorm 回调（自动填充 create_by/update_by）
- `migrate.go` - 配置迁移辅助函数

#### 数据库实现
- `mysql/client.go` - MySQL 客户端实现（支持主从）
- `postgres/client.go` - PostgreSQL 客户端实现（支持主从）
- `sqlite/client.go` - SQLite 客户端实现（单库）

#### 文档
- `README.md` - 使用文档
- `ARCHITECTURE.md` - 架构设计说明
- `MIGRATION_GUIDE.md` - 迁移指南
- `CHANGELOG.md` - 变更日志
- `config.example.yaml` - 配置示例

#### 测试
- `database_test.go` - 单元测试

### 🔄 兼容性

#### 向后兼容
- ✅ 保留 `gorm.go` 中的 `Init()` 函数（旧版本）
- ✅ 保留 `db/` 和 `common/` 目录（旧实现）
- ✅ 提供 `MigrateFromOldConfig()` 自动转换旧配置

#### 破坏性变更
- ⚠️ 新初始化方式：`database.Init(config)` 需要传入配置参数
- ⚠️ 全局访问方式变更：`global.DB` -> `database.DB()`
- ⚠️ 需要显式导入驱动：`import _ "pkg/database/mysql"`

### 📊 性能优化

- **反射缓存**：回调中使用字段信息缓存，减少反射开销
- **连接池管理**：统一配置连接池参数，避免连接泄漏
- **准备语句缓存**：Gorm 配置中启用 `PrepareStmt: true`

### 🔒 安全性

- **密码脱敏**：日志中自动隐藏 DSN 密码
- **连接验证**：启动时 Ping 验证连接可用
- **参数校验**：启动时全面校验配置参数
- **优雅关闭**：应用退出时正确释放资源

### 🎯 设计模式

- **工厂模式（Factory）**：动态创建数据库实例
- **单例模式（Singleton）**：全局唯一数据库连接
- **策略模式（Strategy）**：不同数据库不同策略
- **适配器模式（Adapter）**：日志适配器统一对接
- **依赖倒置（DIP）**：上层依赖接口不依赖实现

### 📦 依赖变更

无新增依赖，继续使用：
- `gorm.io/gorm`
- `gorm.io/driver/mysql`
- `gorm.io/driver/postgres`
- `github.com/glebarez/sqlite`

### 🚀 升级步骤

1. 导入驱动包（触发注册）
2. 修改初始化代码
3. 替换全局变量引用
4. 更新配置文件（可选）
5. 注册 Gorm 回调（如需自动填充）
6. 测试验证

详见 `MIGRATION_GUIDE.md`

### 🐛 已知问题

无

### 📝 待办事项

- [ ] 支持多数据库实例
- [ ] 支持数据库动态切换
- [ ] 支持读写分离策略配置（权重、健康检查）
- [ ] 支持连接池监控指标暴露
- [ ] 支持数据库迁移工具集成

---

## [1.0.0] - 初始版本

### 特性
- 支持 MySQL、PostgreSQL、SQLite
- 基础 Gorm 初始化
- 简单的数据库切换（通过 switch-case）
- Gorm 日志配置
- 回调自动填充 create_by/update_by

### 文件结构
```
pkg/database/
├── gorm.go              # 初始化入口
├── db/                  # 各数据库初始化函数
│   ├── mysql.go
│   ├── pgsql.go
│   └── sqlite.go
└── common/              # 通用配置和回调
    ├── gorm_config.go
    ├── gorm_logger.go
    ├── gorm_callback.go
    └── gorm_middleware.go
```

### 限制
- 不支持主从分离
- 配置分散在多个文件
- 扩展新数据库需修改核心代码
- 无接口抽象，直接使用 `*gorm.DB`
- 日志、工具函数各数据库重复实现
