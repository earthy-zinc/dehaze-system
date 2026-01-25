# Database 通用数据库组件

## 架构设计

采用 **「根目录抽象接口 + 全局管理 + 各数据库子目录实现 + 通用工具包」** 的结构，扩展性极强：

- **接口抽象**：定义顶层通用数据库接口（`DBer`），包含所有数据库的通用操作（主从/读写分离、关闭连接、获取原生实例等）
- **工厂模式**：通过工厂接口（`Factory`）和注册机制，根据配置中的driver自动创建对应数据库实例
- **模块化实现**：每个数据库（MySQL/PostgreSQL/SQLite）单独作为子目录，实现顶层接口，内部封装自身的驱动初始化
- **全局统一管理**：根目录封装全局初始化、实例获取、优雅关闭方法
- **配置通用化**：定义通用数据库配置结构体，适配各数据库专属配置
- **日志统一**：所有数据库实现复用通用 Gorm 日志适配器，统一对接 `pkg/logger`

## 目录结构

```
pkg/database/
├── database.go          # 核心：定义DBer/Factory接口、全局注册/获取/初始化方法
├── config.go            # 通用配置：定义Config结构体，参数校验
├── logger.go            # 通用日志：实现Gorm日志适配器，对接pkg/logger
├── utils.go             # 通用工具：DSN密码隐藏、连接池配置、Gorm通用选项等
├── mysql/               # MySQL实现：实现DBer/Factory接口，封装主从分离
│   └── client.go        # MySQL客户端实现、Gorm初始化、主从分离逻辑
├── postgres/            # PostgreSQL实现：实现DBer/Factory接口，封装主从分离
│   └── client.go        # PostgreSQL客户端实现、Gorm初始化
└── sqlite/              # SQLite实现：实现DBer/Factory接口，封装单库（兼容主从接口）
    └── client.go        # SQLite客户端实现、Gorm初始化
```

## 快速开始

### 1. 配置示例

#### MySQL 配置（支持主从分离）

```yaml
database:
  driver: mysql
  prefix: ""
  singular: true
  log-mode: info
  log-zap: true
  slow-threshold: 200
  max-idle-conns: 10
  max-open-conns: 100
  conn-max-lifetime: 3600s
  conn-max-idle-time: 600s
  
  mysql:
    charset: utf8mb4
    parse-time: true
    loc: Local
    engine: InnoDB
    default-string-size: 191
    
    # 主库配置
    master:
      host: localhost
      port: 3306
      database: dehaze
      username: root
      password: password123
      config: "charset=utf8mb4&parseTime=True&loc=Local"
    
    # 从库配置（可选）
    slaves:
      - host: slave1.example.com
        port: 3306
        database: dehaze
        username: root
        password: password123
      - host: slave2.example.com
        port: 3306
        database: dehaze
        username: root
        password: password123
```

#### PostgreSQL 配置

```yaml
database:
  driver: postgres
  prefix: ""
  singular: true
  log-mode: info
  log-zap: true
  
  postgres:
    ssl-mode: disable
    prefer-simple-protocol: false
    
    master:
      host: localhost
      port: 5432
      database: dehaze
      username: postgres
      password: password123
    
    # 从库配置（可选）
    slaves:
      - host: slave1.example.com
        port: 5432
        database: dehaze
        username: postgres
        password: password123
```

#### SQLite 配置

```yaml
database:
  driver: sqlite
  prefix: ""
  singular: true
  log-mode: info
  log-zap: true
  
  sqlite:
    path: ./data/dehaze.db
```

### 2. 初始化数据库

在 `cmd/main.go` 或应用初始化代码中：

```go
package main

import (
    "github.com/earthyzinc/dehaze-go/pkg/config"
    "github.com/earthyzinc/dehaze-go/pkg/database"
    
    // 导入数据库实现（触发 init() 注册）
    _ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/postgres"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/sqlite"
)

func main() {
    // 加载配置
    cfg := config.GetConfig()
    
    // 构建数据库配置
    dbConfig := &database.Config{
        Driver:          cfg.DB.Type,
        Prefix:          cfg.DB.Prefix,
        Singular:        cfg.DB.Singular,
        LogMode:         cfg.DB.LogMode,
        LogZap:          cfg.DB.LogZap,
        MaxIdleConns:    cfg.DB.MaxIdleConns,
        MaxOpenConns:    cfg.DB.MaxOpenConns,
        // ... 其他配置
    }
    
    // 初始化数据库
    if err := database.Init(dbConfig); err != nil {
        panic(err)
    }
    
    // 确保优雅关闭
    defer database.Close()
    
    // 使用数据库
    db := database.DB()
    // ...
}
```

### 3. 使用数据库

#### 基本用法

```go
import "github.com/earthyzinc/dehaze-go/pkg/database"

// 获取默认数据库实例（主库）
db := database.DB()
db.Create(&user)

// 显式使用主库（写操作）
master := database.Master()
master.Create(&user)

// 显式使用从库（读操作）
slave := database.Slave()
var users []User
slave.Find(&users)
```

#### 在 Repository 中使用

```go
type UserRepository struct {
    db database.DBer
}

func NewUserRepository(db database.DBer) *UserRepository {
    return &UserRepository{db: db}
}

// 写操作使用主库
func (r *UserRepository) Create(user *User) error {
    return r.db.Master().Create(user).Error
}

// 读操作使用从库
func (r *UserRepository) FindByID(id int64) (*User, error) {
    var user User
    err := r.db.Slave().Where("id = ?", id).First(&user).Error
    return &user, err
}
```

#### 依赖注入

```go
// 在 Service 层注入 DBer 接口
type UserService struct {
    db database.DBer
}

func NewUserService(db database.DBer) *UserService {
    return &UserService{db: db}
}

// 在应用初始化时注入
func main() {
    // 初始化数据库
    database.Init(config)
    
    // 注入到 Service
    userService := NewUserService(database.GetDB())
}
```

## 核心接口

### DBer 接口

```go
type DBer interface {
    // Master 获取主库实例（写操作）
    Master(ctx ...context.Context) *gorm.DB
    
    // Slave 获取从库实例（读操作）
    // SQLite等无主从数据库直接返回Master实例
    Slave(ctx ...context.Context) *gorm.DB
    
    // DB 获取默认数据库实例（等同于Master）
    DB(ctx ...context.Context) *gorm.DB
    
    // Close 优雅关闭数据库连接
    Close() error
    
    // GetRawDB 获取原生sql.DB实例
    GetRawDB() (interface{}, error)
}
```

### Factory 接口

```go
type Factory interface {
    // Create 根据配置创建数据库实例
    Create(config *Config) (DBer, error)
}
```

## 扩展新数据库

新增数据库（如 Oracle/MongoDB）仅需以下步骤，无需修改任何现有代码：

1. 创建新的子目录（如 `oracle/`）
2. 实现 `DBer` 和 `Factory` 接口
3. 在 `init()` 函数中注册工厂：

```go
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
    // 实现 DBer 接口的所有方法
}
```

4. 在 main 包中导入：`import _ "github.com/earthyzinc/dehaze-go/pkg/database/oracle"`

## 特性说明

### 主从分离

- **MySQL/PostgreSQL**：支持一主多从，自动负载均衡（随机选择从库）
- **SQLite**：单库架构，`Slave()` 直接返回 `Master()` 实例，接口兼容

### 连接池管理

所有数据库统一使用 `SetupConnectionPool` 配置连接池参数：
- `MaxIdleConns`：最大空闲连接数
- `MaxOpenConns`：最大打开连接数
- `ConnMaxLifetime`：连接最大生命周期
- `ConnMaxIdleTime`：连接最大空闲时间

### 日志统一

所有数据库使用统一的 `GormLogger`，对接 `pkg/logger`：
- 支持慢查询记录（可配置阈值）
- SQL执行日志
- 错误日志
- 密码自动脱敏（日志输出时隐藏DSN中的密码）

### 配置校验

启动时自动校验配置参数：
- 必填项检查
- 参数合法性验证
- 驱动支持检查

## 注意事项

1. **SQLite 并发限制**：SQLite使用文件锁，建议 `MaxOpenConns=1`，避免并发写入冲突
2. **密码安全**：日志输出时会自动隐藏DSN中的密码
3. **优雅关闭**：应用退出时调用 `database.Close()` 确保连接正常关闭
4. **工厂注册**：使用前必须导入对应的数据库实现包（触发 `init()` 注册）

## 迁移指南

从旧架构迁移到新架构：

### 旧代码

```go
// 旧架构
import "github.com/earthyzinc/dehaze-go/pkg/database"

db := database.Init()
db.Create(&user)
```

### 新代码

```go
// 新架构
import (
    "github.com/earthyzinc/dehaze-go/pkg/database"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"  // 注册驱动
)

// 初始化
database.Init(config)

// 使用
db := database.DB()  // 或 database.Master() / database.Slave()
db.Create(&user)
```

### 全局变量替换

```go
// 旧代码
global.DB.Create(&user)

// 新代码
database.DB().Create(&user)
// 或
database.Master().Create(&user)  // 显式写操作
database.Slave().Find(&users)     // 显式读操作
```
