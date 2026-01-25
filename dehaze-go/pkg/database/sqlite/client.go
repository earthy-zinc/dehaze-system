package sqlite

import (
	"context"
	"fmt"

	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/glebarez/sqlite"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

func init() {
	// 注册SQLite工厂
	database.RegisterFactory("sqlite", &sqliteFactory{})
}

// sqliteFactory SQLite工厂实现
type sqliteFactory struct{}

// Create 创建SQLite数据库实例
func (f *sqliteFactory) Create(config *database.Config) (database.DBer, error) {
	return NewClient(config)
}

// Client SQLite客户端实现
// SQLite是单库架构，无主从分离，Slave()方法直接返回Master实例
type Client struct {
	db     *gorm.DB // 数据库实例
	config *database.Config
}

// NewClient 创建SQLite客户端实例
func NewClient(config *database.Config) (*Client, error) {
	if config == nil {
		return nil, fmt.Errorf("sqlite: 配置为空")
	}

	client := &Client{
		config: config,
	}

	// 初始化数据库
	if err := client.init(); err != nil {
		return nil, fmt.Errorf("sqlite: 初始化失败: %w", err)
	}

	return client, nil
}

// init 初始化数据库
func (c *Client) init() error {
	// SQLite文件路径
	path := c.config.SQLite.Path

	// 创建Gorm实例
	db, err := gorm.Open(sqlite.Open(path), database.GetGormConfig(c.config))
	if err != nil {
		return fmt.Errorf("打开数据库失败: %w", err)
	}

	// 获取原生sql.DB并设置连接池
	// 注意：SQLite在并发写入时会有限制，建议MaxOpenConns设置为1
	sqlDB, err := db.DB()
	if err != nil {
		return fmt.Errorf("获取sql.DB失败: %w", err)
	}

	// SQLite特殊处理：限制并发连接数
	// SQLite使用文件锁，过多并发连接会导致性能下降
	sqlDB.SetMaxOpenConns(1)
	sqlDB.SetMaxIdleConns(1)

	// 其他连接池参数按配置设置
	if c.config.ConnMaxLifetime > 0 {
		sqlDB.SetConnMaxLifetime(c.config.ConnMaxLifetime)
	}
	if c.config.ConnMaxIdleTime > 0 {
		sqlDB.SetConnMaxIdleTime(c.config.ConnMaxIdleTime)
	}

	// 验证连接
	if err := database.ValidateConnection(db); err != nil {
		return fmt.Errorf("连接验证失败: %w", err)
	}

	c.db = db

	logger.Info("sqlite: 数据库已连接",
		zap.String("path", path),
	)

	return nil
}

// Master 获取主库实例（写操作）
// SQLite无主从分离，直接返回唯一实例
func (c *Client) Master(ctx ...context.Context) *gorm.DB {
	db := c.db
	if len(ctx) > 0 {
		db = db.WithContext(ctx[0])
	}
	return db
}

// Slave 获取从库实例（读操作）
// SQLite无主从分离，Slave方法直接返回Master实例实现接口兼容
func (c *Client) Slave(ctx ...context.Context) *gorm.DB {
	return c.Master(ctx...)
}

// DB 获取默认数据库实例（等同于Master）
func (c *Client) DB(ctx ...context.Context) *gorm.DB {
	return c.Master(ctx...)
}

// Close 优雅关闭数据库连接
func (c *Client) Close() error {
	if c.db != nil {
		if err := database.CloseDB(c.db); err != nil {
			return fmt.Errorf("sqlite: 关闭失败: %w", err)
		}
		logger.Info("sqlite: 数据库连接已关闭")
	}
	return nil
}

// GetRawDB 获取原生sql.DB实例
func (c *Client) GetRawDB() (interface{}, error) {
	if c.db == nil {
		return nil, fmt.Errorf("sqlite: 数据库为空")
	}
	return c.db.DB()
}
