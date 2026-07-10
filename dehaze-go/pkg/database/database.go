package database

import (
	"context"
	"fmt"
	"sync"

	"gorm.io/gorm"
)

// DBer 数据库接口，定义所有数据库的通用操作
// 支持主从分离、连接管理、原生实例获取等能力
// SQLite等无主从数据库通过Slave()返回Master实例实现接口兼容
type DBer interface {
	// Master 获取主库实例（写操作）
	Master(ctx ...context.Context) *gorm.DB
	// Slave 获取从库实例（读操作）
	// 对于无主从分离的数据库（如SQLite），直接返回Master实例
	Slave(ctx ...context.Context) *gorm.DB
	// DB 获取默认数据库实例（等同于Master）
	DB(ctx ...context.Context) *gorm.DB
	// Close 优雅关闭数据库连接
	Close() error
	// GetRawDB 获取原生sql.DB实例（用于连接池管理、Ping等）
	GetRawDB() (any, error)
}

// Factory 数据库工厂接口，用于创建具体数据库实例
type Factory interface {
	// Create 根据配置创建数据库实例
	Create(config *Config) (DBer, error)
}

// 初始化状态常量
const (
	initStateNone   = iota // 未初始化
	initStateDone          // 已成功初始化
	initStateFailed        // 初始化失败
)

var (
	factoryRegistry = make(map[string]Factory)
	registryMu      sync.RWMutex
	globalDB        DBer
	globalDBMu      sync.Mutex                 // 保护 globalDB 和 initState
	initState       int        = initStateNone // 初始化状态
)

// RegisterFactory 注册数据库工厂
// 各数据库实现在init()函数中调用此方法注册自己
func RegisterFactory(driver string, factory Factory) {
	registryMu.Lock()
	defer registryMu.Unlock()

	if factory == nil {
		panic(fmt.Sprintf("database: 为驱动 %s 注册空工厂", driver))
	}

	if _, exist := factoryRegistry[driver]; exist {
		panic(fmt.Sprintf("database: 驱动 %s 被重复注册", driver))
	}

	factoryRegistry[driver] = factory
}

// GetFactory 获取指定驱动的工厂实例
func GetFactory(driver string) (Factory, error) {
	registryMu.RLock()
	defer registryMu.RUnlock()

	factory, ok := factoryRegistry[driver]
	if !ok {
		return nil, fmt.Errorf("database: 未知驱动 %q (是否忘记导入?)", driver)
	}

	return factory, nil
}

// Init 全局初始化数据库（单例模式）
// 根据配置自动选择数据库驱动并创建实例
// 支持失败后重试：若上次初始化失败，可再次调用重试
func Init(config *Config) error {
	globalDBMu.Lock()
	defer globalDBMu.Unlock()

	// 已成功初始化，直接返回
	if initState == initStateDone && globalDB != nil {
		return nil
	}

	// 重置失败状态，允许重试
	if initState == initStateFailed {
		initState = initStateNone
	}

	if config == nil {
		initState = initStateFailed
		return fmt.Errorf("database: 配置为空")
	}

	if err := config.Validate(); err != nil {
		initState = initStateFailed
		return fmt.Errorf("database: 无效的配置: %w", err)
	}

	factory, err := GetFactory(config.Driver)
	if err != nil {
		initState = initStateFailed
		return err
	}

	db, err := factory.Create(config)
	if err != nil {
		initState = initStateFailed
		return fmt.Errorf("database: 创建 %s 实例失败: %w", config.Driver, err)
	}

	globalDB = db
	initState = initStateDone
	return nil
}

// GetDB 获取全局数据库实例
// 使用前必须先调用Init()初始化
func GetDB() DBer {
	if globalDB == nil {
		panic("database: 使用GetDB()前必须先调用Init()")
	}
	return globalDB
}

// Master 获取全局主库实例
func Master(ctx ...context.Context) *gorm.DB {
	return GetDB().Master(ctx...)
}

// Slave 获取全局从库实例
func Slave(ctx ...context.Context) *gorm.DB {
	return GetDB().Slave(ctx...)
}

// DB 获取全局默认数据库实例
func DB(ctx ...context.Context) *gorm.DB {
	return GetDB().DB(ctx...)
}

// Close 优雅关闭全局数据库连接
func Close() error {
	if globalDB != nil {
		return globalDB.Close()
	}
	return nil
}

// ResetGlobal 重置全局实例（仅用于测试）
func ResetGlobal() {
	globalDBMu.Lock()
	defer globalDBMu.Unlock()

	globalDB = nil
	initState = initStateNone
}
