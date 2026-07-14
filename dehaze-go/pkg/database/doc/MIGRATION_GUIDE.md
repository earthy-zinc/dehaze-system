# Database 组件迁移指南

## 快速迁移检查清单

- [ ] 1. 更新导入语句（添加驱动导入）
- [ ] 2. 修改初始化代码
- [ ] 3. 替换全局变量引用
- [ ] 4. 更新配置文件格式（可选）
- [ ] 5. 注册 Gorm 回调（如需自动填充功能）
- [ ] 6. 测试验证

## 详细迁移步骤

### 步骤 1：更新导入语句

**在 `cmd/main.go` 或应用入口文件中添加驱动导入：**

```go
import (
    "github.com/earthyzinc/dehaze-go/pkg/database"
    
    // 必须导入：触发驱动注册
    _ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/postgres"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/sqlite"
)
```

### 步骤 2：修改初始化代码

#### 方案A：使用迁移辅助函数（最简单）

```go
// 旧代码
import "github.com/earthyzinc/dehaze-go/pkg/database"
db := database.Init()
global.DB = db

// 新代码
import (
    "github.com/earthyzinc/dehaze-go/pkg/config"
    "github.com/earthyzinc/dehaze-go/pkg/database"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
)

// 从旧配置自动迁移
oldConfig := config.GetConfig().DB
newConfig, err := database.MigrateFromOldConfig(oldConfig)
if err != nil {
    panic(err)
}

// 初始化数据库
if err := database.Init(newConfig); err != nil {
    panic(err)
}

// 确保优雅关闭
defer database.Close()
```

#### 方案B：直接使用新配置（推荐）

**更新配置文件（如 `config.yaml`）：**

```yaml
# 新配置格式
database:
  driver: mysql
  prefix: ""
  singular: true
  log-mode: info
  log-zap: true
  max-idle-conns: 10
  max-open-conns: 100
  
  mysql:
    charset: utf8mb4
    parse-time: true
    loc: Local
    engine: InnoDB
    
    master:
      host: localhost
      port: 3306
      database: dehaze
      username: root
      password: password123
```

**修改初始化代码：**

```go
import (
    "github.com/earthyzinc/dehaze-go/pkg/config"
    "github.com/earthyzinc/dehaze-go/pkg/database"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
)

// 加载配置
cfg := config.GetConfig()

// 构建数据库配置
dbConfig := &database.Config{
    Driver:       cfg.Database.Driver,
    Prefix:       cfg.Database.Prefix,
    Singular:     cfg.Database.Singular,
    LogMode:      cfg.Database.LogMode,
    LogZap:       cfg.Database.LogZap,
    MaxIdleConns: cfg.Database.MaxIdleConns,
    MaxOpenConns: cfg.Database.MaxOpenConns,
    MySQL:        cfg.Database.MySQL,
    Postgres:     cfg.Database.Postgres,
    SQLite:       cfg.Database.SQLite,
}

// 初始化数据库
if err := database.Init(dbConfig); err != nil {
    panic(err)
}

defer database.Close()
```

### 步骤 3：替换全局变量引用

**批量查找替换（建议使用IDE全局替换功能）：**

#### 3.1 基本CRUD操作

```go
// 旧代码
global.DB.Create(&user)
global.DB.Find(&users)
global.DB.First(&user, id)
global.DB.Updates(&user)
global.DB.Delete(&user)

// 新代码（方式1：使用默认实例）
database.DB().Create(&user)
database.DB().Find(&users)
database.DB().First(&user, id)
database.DB().Updates(&user)
database.DB().Delete(&user)

// 新代码（方式2：显式区分读写）
database.Master().Create(&user)    // 写操作
database.Slave().Find(&users)       // 读操作
database.Slave().First(&user, id)   // 读操作
database.Master().Updates(&user)    // 写操作
database.Master().Delete(&user)     // 写操作
```

#### 3.2 事务操作

```go
// 旧代码
tx := global.DB.Begin()
tx.Create(&user)
tx.Commit()

// 新代码
tx := database.Master().Begin()  // 事务使用主库
tx.Create(&user)
tx.Commit()
```

#### 3.3 WithContext

```go
// 旧代码
global.DB.WithContext(ctx).Find(&users)

// 新代码
database.DB(ctx).Find(&users)
// 或
database.Master(ctx).Create(&user)
database.Slave(ctx).Find(&users)
```

#### 3.4 原生SQL

```go
// 旧代码
global.DB.Raw("SELECT * FROM users WHERE id = ?", id).Scan(&user)
global.DB.Exec("UPDATE users SET status = ? WHERE id = ?", status, id)

// 新代码
database.Slave().Raw("SELECT * FROM users WHERE id = ?", id).Scan(&user)
database.Master().Exec("UPDATE users SET status = ? WHERE id = ?", status, id)
```

### 步骤 4：Repository 层重构（推荐）

**从全局变量改为依赖注入：**

```go
// 旧代码
type UserRepository struct{}

func NewUserRepository() *UserRepository {
    return &UserRepository{}
}

func (r *UserRepository) Create(user *User) error {
    return global.DB.Create(user).Error
}

func (r *UserRepository) FindByID(id int64) (*User, error) {
    var user User
    err := global.DB.First(&user, id).Error
    return &user, err
}

// 新代码（推荐）
type UserRepository struct {
    db database.DBer
}

func NewUserRepository(db database.DBer) *UserRepository {
    return &UserRepository{db: db}
}

func (r *UserRepository) Create(user *User) error {
    return r.db.Master().Create(user).Error
}

func (r *UserRepository) FindByID(id int64) (*User, error) {
    var user User
    err := r.db.Slave().First(&user, id).Error
    return &user, err
}

// 应用初始化时注入
func main() {
    database.Init(config)
    userRepo := NewUserRepository(database.GetDB())
}
```

### 步骤 5：注册 Gorm 回调

**如果需要自动填充 `create_by`/`update_by` 字段：**

```go
import (
    "github.com/earthyzinc/dehaze-go/pkg/database"
    _ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
    "github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
)

func main() {
    // 初始化数据库
    database.Init(config)
    defer database.Close()
    
    // 注册回调（自动填充create_by/update_by）
    database.RegisterGormCallbacks(database.DB())
    
    // 注册Gin中间件（注入 userID/deptID/dataScope 到 context，供 GORM 回调和 DataScopePlugin 读取）
    router := gin.Default()
    router.Use(middleware.UserContextMiddleware())
}
```

### 步骤 6：测试验证

#### 6.1 启动应用验证

```bash
# 启动应用，检查日志
go run cmd/main.go

# 预期日志输出示例：
# [INFO] mysql: master connected | host=localhost | port=3306 | database=dehaze
# [INFO] mysql: slave connected | index=0 | host=slave1.example.com
```

#### 6.2 单元测试

```go
func TestDatabaseConnection(t *testing.T) {
    config := &database.Config{
        Driver: "sqlite",
        SQLite: database.SQLiteConfig{Path: ":memory:"},
    }
    
    database.ResetGlobal()
    err := database.Init(config)
    assert.NoError(t, err)
    
    db := database.DB()
    assert.NotNil(t, db)
    
    // 测试基本操作
    type TestUser struct {
        ID   int64
        Name string
    }
    db.AutoMigrate(&TestUser{})
    
    user := &TestUser{Name: "test"}
    err = db.Create(user).Error
    assert.NoError(t, err)
    assert.NotZero(t, user.ID)
    
    database.Close()
}
```

## 常见问题

### Q1: 为什么需要导入下划线包？

```go
import _ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
```

**A:** 下划线导入触发包的 `init()` 函数，用于注册数据库驱动工厂。不导入会导致运行时错误：`database: unknown driver "mysql"`

### Q2: Master 和 Slave 什么时候使用？

**A:**
- **Master**：写操作（Create/Update/Delete/事务）
- **Slave**：读操作（Find/First/Count/查询）
- **DB**：默认等同于 Master，不确定时使用

### Q3: SQLite 有主从分离吗？

**A:** 没有。SQLite 是单文件数据库，`Slave()` 方法直接返回 `Master()` 实例，接口兼容。

### Q4: 如何支持多数据库？

**A:** 暂不支持多数据库实例。如需多库，建议：
1. 为每个库创建独立的 `DBer` 实例（不使用全局单例）
2. 通过依赖注入分别注入到不同的 Repository

### Q5: 旧配置格式还能用吗？

**A:** 可以！使用 `MigrateFromOldConfig()` 自动转换：

```go
oldConfig := config.GetConfig().DB
newConfig, _ := database.MigrateFromOldConfig(oldConfig)
database.Init(newConfig)
```

### Q6: 如何添加主从配置？

**A:** 在配置文件中添加 `slaves` 节点：

```yaml
database:
  driver: mysql
  mysql:
    master:
      host: master.db.com
      # ...
    slaves:
      - host: slave1.db.com
        port: 3306
        database: dehaze
        username: root
        password: xxx
      - host: slave2.db.com
        port: 3306
        database: dehaze
        username: root
        password: xxx
```

### Q7: 慢查询阈值如何设置？

**A:** 配置文件中设置 `slow-threshold`（单位：毫秒）：

```yaml
database:
  slow-threshold: 500  # 超过500ms记录为慢查询
```

### Q8: 如何在日志中隐藏密码？

**A:** 自动处理！日志输出时会自动调用 `MaskDSN()` 隐藏密码：

```
[INFO] mysql: master connected | dsn=root:***@tcp(localhost:3306)/dehaze
```

## 迁移脚本示例

### 全局替换脚本（仅供参考）

```bash
#!/bin/bash
# 批量替换 global.DB 为 database.DB()

# 备份原文件
find ./internal -name "*.go" -exec cp {} {}.bak \;

# 替换
find ./internal -name "*.go" -exec sed -i 's/global\.DB/database.DB()/g' {} \;

echo "替换完成！请手动检查并根据读写操作调整为 Master()/Slave()"
```

**注意**：此脚本仅做简单替换，建议：
1. 使用 IDE 的查找替换功能
2. 手动检查每处替换
3. 根据读写操作调整为 `Master()` 或 `Slave()`

## 迁移后检查清单

- [ ] 应用能正常启动
- [ ] 数据库连接日志正常
- [ ] 基本CRUD操作正常
- [ ] 事务操作正常
- [ ] 主从切换正常（如配置了从库）
- [ ] 慢查询日志正常记录
- [ ] 密码在日志中已脱敏
- [ ] 优雅关闭正常（应用退出时连接正确释放）
- [ ] 单元测试通过
- [ ] 集成测试通过

## 回滚方案

如果迁移遇到问题，可快速回滚：

1. **恢复备份文件**：
   ```bash
   find ./internal -name "*.go.bak" -exec bash -c 'mv "$1" "${1%.bak}"' _ {} \;
   ```

2. **撤销配置文件修改**（使用git）：
   ```bash
   git checkout config.yaml
   ```

3. **使用旧的初始化方式**：
   ```go
   import "github.com/earthyzinc/dehaze-go/pkg/database"
   db := database.Init()  // 旧版本仍保留在 gorm.go
   ```

## 获得帮助

如遇到问题，请查看：
- `README.md` - 组件使用文档
- `ARCHITECTURE.md` - 架构设计说明
- `database_test.go` - 测试示例
- `config.example.yaml` - 配置示例

或提交 Issue 描述问题。
