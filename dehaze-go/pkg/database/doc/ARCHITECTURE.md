# Database 组件架构改造总结

## 改造概述

本次改造将 `pkg/database` 从简单的初始化函数改造为一个**高度抽象、模块化、可扩展**的数据库管理组件，完全符合**开闭原则**（对扩展开放，对修改封闭）。

## 架构设计核心原则

### 1. 接口抽象（DBer）

定义统一的数据库接口，屏蔽底层实现细节：

```go
type DBer interface {
    Master(ctx ...context.Context) *gorm.DB  // 主库（写）
    Slave(ctx ...context.Context) *gorm.DB   // 从库（读）
    DB(ctx ...context.Context) *gorm.DB      // 默认实例
    Close() error                             // 优雅关闭
    GetRawDB() (interface{}, error)          // 获取原生实例
}
```

**核心价值**：
- 上层代码完全不感知底层是哪种数据库
- 支持主从分离，业务层显式区分读写操作
- SQLite等单库通过 `Slave() -> Master()` 实现接口兼容

### 2. 工厂模式（Factory）

通过工厂接口和注册机制实现动态创建：

```go
type Factory interface {
    Create(config *Config) (DBer, error)
}

// 各数据库在init()中自动注册
func init() {
    database.RegisterFactory("mysql", &mysqlFactory{})
}
```

**核心价值**：
- 运行时根据配置动态选择数据库
- 新增数据库无需修改任何现有代码
- 依赖倒置：上层依赖抽象，不依赖具体实现

### 3. 模块化实现

每个数据库独立子目录，互不干扰：

```
database/
├── database.go       # 接口定义 + 全局管理
├── config.go         # 通用配置
├── logger.go         # 通用日志
├── utils.go          # 通用工具
├── mysql/            # MySQL实现（支持主从）
├── postgres/         # PostgreSQL实现（支持主从）
└── sqlite/           # SQLite实现（单库）
```

**核心价值**：
- 单一职责：每个子目录只负责一种数据库
- 高内聚低耦合：修改MySQL不影响PostgreSQL
- 易于测试：可以独立测试每个数据库实现

### 4. 全局统一管理

根目录提供全局初始化和访问接口：

```go
// 初始化（单例模式）
database.Init(config)

// 获取实例
db := database.DB()
master := database.Master()
slave := database.Slave()

// 优雅关闭
database.Close()
```

**核心价值**：
- 单例模式确保全局唯一实例
- 极简API，cmd层仅需一行代码初始化
- 线程安全的全局访问

## 关键特性

### 1. 主从分离支持

```go
// MySQL/PostgreSQL 支持一主多从
config := &Config{
    Driver: "mysql",
    MySQL: MySQLConfig{
        Master: MySQLInstanceConfig{...},
        Slaves: []MySQLInstanceConfig{
            {...}, // slave1
            {...}, // slave2
        },
    },
}

// 业务层显式区分读写
database.Master().Create(&user)   // 写操作 -> 主库
database.Slave().Find(&users)      // 读操作 -> 从库（随机负载均衡）
```

### 2. 配置通用化

```go
type Config struct {
    Driver       string        // 通用字段：驱动类型
    MaxIdleConns int          // 通用字段：连接池配置
    MySQL        MySQLConfig   // MySQL专属配置段
    Postgres     PostgresConfig // PostgreSQL专属配置段
    SQLite       SQLiteConfig   // SQLite专属配置段
}
```

**优势**：
- 一个配置结构体支持所有数据库
- 配置校验在启动时进行，避免运行时错误
- 支持从旧配置自动迁移

### 3. 日志统一

所有数据库使用统一的 `GormLogger`，对接 `pkg/logger`：

```go
type GormLogger struct {
    SlowThreshold time.Duration  // 慢查询阈值
    LogLevel      LogLevel        // 日志级别
    UseZap        bool            // 是否使用zap
}
```

**特性**：
- 慢查询自动记录
- SQL执行日志
- 密码自动脱敏（DSN中的密码被隐藏）
- 统一格式化输出

### 4. 工具函数复用

```go
// DSN密码隐藏
MaskDSN(dsn)

// 连接池配置
SetupConnectionPool(sqlDB, config)

// Gorm配置生成
GetGormConfig(config)

// 连接验证
ValidateConnection(db)
```

### 5. 回调自动填充

保留原有的 `create_by/update_by` 自动填充功能：

```go
// 注册回调
RegisterGormCallbacks(db)

// 自动从Gin上下文获取用户ID并填充
// CreateBy / UpdateBy 字段
```

## 扩展性演示

### 新增数据库仅需3步

**示例：新增 Oracle 支持**

#### 1. 创建子目录和实现

```go
// pkg/database/oracle/client.go
package oracle

import "github.com/earthyzinc/dehaze-go/pkg/database"

func init() {
    database.RegisterFactory("oracle", &oracleFactory{})
}

type oracleFactory struct{}

func (f *oracleFactory) Create(config *database.Config) (database.DBer, error) {
    return NewClient(config)
}

type Client struct {
    master *gorm.DB
    slaves []*gorm.DB
}

// 实现 DBer 接口的所有方法
func (c *Client) Master(ctx ...context.Context) *gorm.DB { ... }
func (c *Client) Slave(ctx ...context.Context) *gorm.DB { ... }
func (c *Client) DB(ctx ...context.Context) *gorm.DB { ... }
func (c *Client) Close() error { ... }
func (c *Client) GetRawDB() (interface{}, error) { ... }
```

#### 2. 扩展配置结构体

```go
// pkg/database/config.go
type Config struct {
    // ... 现有字段
    Oracle OracleConfig `mapstructure:"oracle" json:"oracle" yaml:"oracle"`
}

type OracleConfig struct {
    Master OracleInstanceConfig
    Slaves []OracleInstanceConfig
}
```

#### 3. 导入使用

```go
import (
    "github.com/earthyzinc/dehaze-go/pkg/database"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/oracle"
)

config := &database.Config{
    Driver: "oracle",
    Oracle: database.OracleConfig{...},
}

database.Init(config)
```

**完全不需要修改任何现有代码！**

## 与旧架构对比

| 特性 | 旧架构 | 新架构 |
|------|--------|--------|
| 初始化方式 | `db := database.Init()` | `database.Init(config); db := database.DB()` |
| 数据库切换 | 修改代码中的 `switch` | 仅修改配置文件 |
| 主从分离 | 不支持 | 完整支持（MySQL/PG） |
| 扩展新数据库 | 修改 `gorm.go` 添加 `case` | 新增子目录，零修改 |
| 接口抽象 | 直接返回 `*gorm.DB` | 返回 `DBer` 接口 |
| 配置管理 | 散落在各初始化函数 | 统一 `Config` 结构体 |
| 日志管理 | 各数据库独立实现 | 统一 `GormLogger` |
| 工具函数 | 重复代码 | 抽取到 `utils.go` |
| 依赖注入 | 使用全局变量 | 支持接口注入 |

## 迁移指南

### 最小改动迁移

**步骤1：更新导入**

```go
// 旧代码
import "github.com/earthyzinc/dehaze-go/pkg/database"

// 新代码：添加驱动导入
import (
    "github.com/earthyzinc/dehaze-go/pkg/database"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
)
```

**步骤2：修改初始化**

```go
// 旧代码
db := database.Init()

// 新代码（使用迁移辅助函数）
import "github.com/earthyzinc/dehaze-go/pkg/config"

oldConfig := config.GetConfig().DB
newConfig, _ := database.MigrateFromOldConfig(oldConfig)
database.Init(newConfig)
```

**步骤3：替换全局变量**

```go
// 旧代码
global.DB.Create(&user)

// 新代码
database.DB().Create(&user)

// 或显式区分读写
database.Master().Create(&user)
database.Slave().Find(&users)
```

### 推荐的重构方式

**使用依赖注入替代全局变量**：

```go
// Repository层
type UserRepository struct {
    db database.DBer
}

func NewUserRepository(db database.DBer) *UserRepository {
    return &UserRepository{db: db}
}

func (r *UserRepository) Create(user *User) error {
    return r.db.Master().Create(user).Error
}

// Service层
type UserService struct {
    userRepo *UserRepository
}

func NewUserService(db database.DBer) *UserService {
    return &UserService{
        userRepo: NewUserRepository(db),
    }
}

// 应用初始化
func main() {
    database.Init(config)
    defer database.Close()
    
    userService := NewUserService(database.GetDB())
}
```

## 测试支持

### 单元测试

```go
func TestUserService(t *testing.T) {
    // 使用SQLite内存数据库
    config := &database.Config{
        Driver: "sqlite",
        SQLite: database.SQLiteConfig{Path: ":memory:"},
    }
    
    database.ResetGlobal() // 重置单例（测试专用）
    database.Init(config)
    defer database.Close()
    
    service := NewUserService(database.GetDB())
    // ... 测试代码
}
```

### Mock测试

```go
type mockDB struct {
    database.DBer
}

func (m *mockDB) Master(ctx ...context.Context) *gorm.DB {
    // 返回mock实例
}

func TestWithMock(t *testing.T) {
    service := NewUserService(&mockDB{})
    // ... 测试代码
}
```

## 配置示例

### 开发环境（SQLite）

```yaml
database:
  driver: sqlite
  log-mode: info
  log-zap: true
  sqlite:
    path: ./data/dev.db
```

### 生产环境（MySQL主从）

```yaml
database:
  driver: mysql
  log-mode: error
  log-zap: true
  slow-threshold: 500
  max-idle-conns: 50
  max-open-conns: 200
  
  mysql:
    charset: utf8mb4
    parse-time: true
    loc: Local
    engine: InnoDB
    
    master:
      host: mysql-master.prod
      port: 3306
      database: dehaze
      username: app_user
      password: ${MYSQL_PASSWORD}
    
    slaves:
      - host: mysql-slave1.prod
        port: 3306
        database: dehaze
        username: app_user
        password: ${MYSQL_PASSWORD}
      - host: mysql-slave2.prod
        port: 3306
        database: dehaze
        username: app_user
        password: ${MYSQL_PASSWORD}
```

## 设计模式应用

1. **工厂模式（Factory）**：动态创建数据库实例
2. **单例模式（Singleton）**：全局唯一数据库连接
3. **策略模式（Strategy）**：不同数据库实现不同策略
4. **适配器模式（Adapter）**：日志适配器统一对接
5. **依赖倒置（DIP）**：上层依赖接口，不依赖实现

## 性能优化

1. **连接池管理**：统一配置，避免连接泄漏
2. **反射缓存**：回调中使用字段缓存，减少反射开销
3. **随机负载均衡**：多从库场景简单均匀分布
4. **慢查询监控**：自动记录超过阈值的SQL

## 安全性

1. **密码脱敏**：日志输出时自动隐藏密码
2. **连接验证**：启动时Ping验证连接可用
3. **优雅关闭**：应用退出时正确释放资源
4. **参数校验**：启动时全面校验配置参数

## 总结

本次改造实现了：

✅ **高度抽象**：通过 `DBer` 接口屏蔽实现细节  
✅ **模块化**：每个数据库独立实现，互不影响  
✅ **可扩展**：新增数据库零修改现有代码  
✅ **主从分离**：完整支持读写分离  
✅ **配置统一**：一个配置结构体适配所有数据库  
✅ **日志统一**：所有数据库复用同一日志适配器  
✅ **工具复用**：抽取通用工具函数避免重复  
✅ **向后兼容**：提供迁移工具支持旧配置  
✅ **易于测试**：支持依赖注入和Mock  
✅ **生产就绪**：连接池、慢查询、优雅关闭、安全性完备  

**完全符合开闭原则，扩展性极强！**
