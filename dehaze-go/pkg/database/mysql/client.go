package mysql

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

func init() {
	// 注册MySQL工厂
	database.RegisterFactory("mysql", &mysqlFactory{})
}

// mysqlFactory MySQL工厂实现
type mysqlFactory struct{}

// Create 创建MySQL数据库实例
func (f *mysqlFactory) Create(config *database.Config) (database.DBer, error) {
	return NewClient(config)
}

// Client MySQL客户端实现
type Client struct {
	master *gorm.DB   // 主库实例
	slaves []*gorm.DB // 从库实例列表
	config *database.Config
}

// NewClient 创建MySQL客户端实例
func NewClient(config *database.Config) (*Client, error) {
	if config == nil {
		return nil, fmt.Errorf("mysql: 配置为空")
	}

	client := &Client{
		config: config,
		slaves: make([]*gorm.DB, 0),
	}

	// 初始化主库
	if err := client.initMaster(); err != nil {
		return nil, fmt.Errorf("mysql: 初始化主库失败: %w", err)
	}

	// 初始化从库（可选）
	if err := client.initSlaves(); err != nil {
		// 从库初始化失败不影响主库使用，仅记录日志
		logger.Warn("mysql: 从库初始化失败，将使用主库进行读操作", zap.Error(err))
	}

	return client, nil
}

// initMaster 初始化主库
func (c *Client) initMaster() error {
	masterConfig := c.config.MySQL.Master

	// 构建DSN
	dsn := database.BuildMySQLDSN(
		masterConfig,
		c.config.MySQL.Charset,
		c.config.MySQL.Loc,
		c.config.MySQL.ParseTime,
	)

	// MySQL驱动配置
	mysqlConfig := mysql.Config{
		DSN:                       dsn,
		DefaultStringSize:         c.config.MySQL.DefaultStringSize,
		SkipInitializeWithVersion: c.config.MySQL.SkipInitWithVersion,
	}

	// 创建Gorm实例
	db, err := gorm.Open(mysql.New(mysqlConfig), database.GetGormConfig(c.config))
	if err != nil {
		return fmt.Errorf("打开主库失败: %w", err)
	}

	// 设置表引擎
	if c.config.MySQL.Engine != "" {
		db.InstanceSet("gorm:table_options", "ENGINE="+c.config.MySQL.Engine)
	}

	// 获取原生sql.DB并设置连接池
	sqlDB, err := db.DB()
	if err != nil {
		return fmt.Errorf("获取主库sql.DB失败: %w", err)
	}
	database.SetupConnectionPool(sqlDB, c.config)

	// 验证连接
	if err := database.ValidateConnection(db); err != nil {
		return fmt.Errorf("主库连接验证失败: %w", err)
	}

	c.master = db

	logger.Info("mysql: 主库已连接",
		zap.String("host", masterConfig.Host),
		zap.Int("port", masterConfig.Port),
		zap.String("database", masterConfig.Database),
		zap.String("dsn", database.MaskDSN(dsn)),
	)

	return nil
}

// initSlaves 初始化从库
func (c *Client) initSlaves() error {
	if len(c.config.MySQL.Slaves) == 0 {
		return nil
	}

	for i, slaveConfig := range c.config.MySQL.Slaves {
		// 构建DSN
		dsn := database.BuildMySQLDSN(
			slaveConfig,
			c.config.MySQL.Charset,
			c.config.MySQL.Loc,
			c.config.MySQL.ParseTime,
		)

		// MySQL驱动配置
		mysqlConfig := mysql.Config{
			DSN:                       dsn,
			DefaultStringSize:         c.config.MySQL.DefaultStringSize,
			SkipInitializeWithVersion: c.config.MySQL.SkipInitWithVersion,
		}

		// 创建Gorm实例
		db, err := gorm.Open(mysql.New(mysqlConfig), database.GetGormConfig(c.config))
		if err != nil {
			return fmt.Errorf("打开从库[%d]失败: %w", i, err)
		}

		// 获取原生sql.DB并设置连接池
		sqlDB, err := db.DB()
		if err != nil {
			return fmt.Errorf("获取从库[%d] sql.DB失败: %w", i, err)
		}
		database.SetupConnectionPool(sqlDB, c.config)

		// 验证连接
		if err := database.ValidateConnection(db); err != nil {
			return fmt.Errorf("从库[%d]连接验证失败: %w", i, err)
		}

		c.slaves = append(c.slaves, db)

		logger.Info("mysql: 从库已连接",
			zap.Int("index", i),
			zap.String("host", slaveConfig.Host),
			zap.Int("port", slaveConfig.Port),
			zap.String("database", slaveConfig.Database),
			zap.String("dsn", database.MaskDSN(dsn)),
		)
	}

	return nil
}

// Master 获取主库实例（写操作）
func (c *Client) Master(ctx ...context.Context) *gorm.DB {
	db := c.master
	if len(ctx) > 0 {
		db = db.WithContext(ctx[0])
	}
	return db
}

// Slave 获取从库实例（读操作）
// 如果没有从库，返回主库实例
// 如果有多个从库，随机返回一个（简单负载均衡）
func (c *Client) Slave(ctx ...context.Context) *gorm.DB {
	// 如果没有从库，返回主库
	if len(c.slaves) == 0 {
		return c.Master(ctx...)
	}

	// 随机选择一个从库（简单负载均衡）
	index := rand.IntN(len(c.slaves))
	db := c.slaves[index]

	if len(ctx) > 0 {
		db = db.WithContext(ctx[0])
	}

	return db
}

// DB 获取默认数据库实例（等同于Master）
func (c *Client) DB(ctx ...context.Context) *gorm.DB {
	return c.Master(ctx...)
}

// Close 优雅关闭所有数据库连接
func (c *Client) Close() error {
	var errs []error

	// 关闭主库
	if c.master != nil {
		if err := database.CloseDB(c.master); err != nil {
			errs = append(errs, fmt.Errorf("关闭主库失败: %w", err))
		} else {
			logger.Info("mysql: 主库连接已关闭")
		}
	}

	// 关闭从库
	for i, slave := range c.slaves {
		if err := database.CloseDB(slave); err != nil {
			errs = append(errs, fmt.Errorf("关闭从库[%d]失败: %w", i, err))
		} else {
			logger.Info("mysql: 从库连接已关闭", zap.Int("index", i))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("mysql: 关闭错误: %v", errs)
	}

	return nil
}

// GetRawDB 获取原生sql.DB实例（主库）
func (c *Client) GetRawDB() (interface{}, error) {
	if c.master == nil {
		return nil, fmt.Errorf("mysql: 主库为空")
	}
	return c.master.DB()
}
