package database

import (
	"fmt"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"gorm.io/gorm/logger"
)

// Config 通用数据库配置结构体
// 包含驱动类型、通用连接池参数、各数据库专属配置段
// 适配 Viper 配置解析
type Config struct {
	// Driver 数据库驱动类型（mysql/postgres/sqlite）
	Driver string

	// 通用配置
	Prefix        string
	Singular      bool
	LogMode       string
	LogZap        bool
	SlowThreshold int

	// 连接池配置
	MaxIdleConns    int
	MaxOpenConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration

	// MySQL专属配置
	MySQL MySQLConfig

	// PostgreSQL专属配置
	Postgres PostgresConfig

	// SQLite专属配置
	SQLite SQLiteConfig
}

// MySQLConfig MySQL专属配置
type MySQLConfig struct {
	// 主库配置
	Master MySQLInstanceConfig
	// 从库配置（可选，支持多从库）
	Slaves []MySQLInstanceConfig
	// 其他MySQL专属配置
	Charset             string
	ParseTime           bool
	Loc                 string
	Engine              string
	DefaultStringSize   uint
	SkipInitWithVersion bool
}

// MySQLInstanceConfig MySQL实例配置
type MySQLInstanceConfig struct {
	Host     string
	Port     int
	Database string
	Username string
	Password string
	Config   string
}

// PostgresConfig PostgreSQL专属配置
type PostgresConfig struct {
	// 主库配置
	Master PostgresInstanceConfig
	// 从库配置（可选，支持多从库）
	Slaves []PostgresInstanceConfig
	// 其他PostgreSQL专属配置
	SSLMode              string
	PreferSimpleProtocol bool
}

// PostgresInstanceConfig PostgreSQL实例配置
type PostgresInstanceConfig struct {
	Host     string
	Port     int
	Database string
	Username string
	Password string
	Config   string
}

// SQLiteConfig SQLite专属配置
type SQLiteConfig struct {
	Path string
}

func GetDatabaseConfig() *Config {
	cfg := config.GetConfig()
	dbCfg := cfg.DB

	// 初始化数据库配置
	result := &Config{
		Driver:          dbCfg.Driver,
		Prefix:          dbCfg.Prefix,
		Singular:        dbCfg.Singular,
		LogMode:         dbCfg.LogMode,
		LogZap:          dbCfg.LogZap,
		SlowThreshold:   dbCfg.SlowThreshold,
		MaxIdleConns:    dbCfg.MaxIdleConns,
		MaxOpenConns:    dbCfg.MaxOpenConns,
		ConnMaxLifetime: dbCfg.GetConnMaxLifetime(),
		ConnMaxIdleTime: dbCfg.GetConnMaxIdleTime(),

		MySQL: MySQLConfig{
			Slaves:            []MySQLInstanceConfig{},
			Charset:           "utf8mb4",
			ParseTime:         true,
			Loc:               "Local",
			Engine:            dbCfg.Engine,
			DefaultStringSize: dbCfg.DefaultStringSize,
		},
		Postgres: PostgresConfig{
			Slaves:               []PostgresInstanceConfig{},
			SSLMode:              "disable",
			PreferSimpleProtocol: false,
		},
		SQLite: SQLiteConfig{},
	}

	// 根据配置填充对应数据库的Master配置
	if dbCfg.MySQL != nil {
		result.MySQL.Master = MySQLInstanceConfig{
			Host:     dbCfg.MySQL.Host,
			Port:     dbCfg.MySQL.Port,
			Database: dbCfg.MySQL.Database,
			Username: dbCfg.MySQL.Username,
			Password: dbCfg.MySQL.Password,
			Config:   dbCfg.MySQL.Config,
		}
		// 覆盖默认值
		if dbCfg.MySQL.Charset != "" {
			result.MySQL.Charset = dbCfg.MySQL.Charset
		}
		result.MySQL.ParseTime = dbCfg.MySQL.ParseTime
		if dbCfg.MySQL.Loc != "" {
			result.MySQL.Loc = dbCfg.MySQL.Loc
		}
	}

	if dbCfg.Postgres != nil {
		result.Postgres.Master = PostgresInstanceConfig{
			Host:     dbCfg.Postgres.Host,
			Port:     dbCfg.Postgres.Port,
			Database: dbCfg.Postgres.Database,
			Username: dbCfg.Postgres.Username,
			Password: dbCfg.Postgres.Password,
			Config:   dbCfg.Postgres.Config,
		}
		if dbCfg.Postgres.SSLMode != "" {
			result.Postgres.SSLMode = dbCfg.Postgres.SSLMode
		}
	}

	if dbCfg.SQLite != nil {
		result.SQLite.Path = dbCfg.SQLite.Path
	}

	return result
}

// Validate 校验配置参数
func (c *Config) Validate() error {
	if c.Driver == "" {
		return fmt.Errorf("驱动不能为空")
	}

	driver := strings.ToLower(c.Driver)
	if driver != "mysql" && driver != "postgres" && driver != "sqlite" {
		return fmt.Errorf("不支持的驱动: %s，必须是以下之一: mysql, postgres, sqlite", c.Driver)
	}

	// 校验连接池参数
	if c.MaxIdleConns < 1 {
		c.MaxIdleConns = 10
	}
	if c.MaxOpenConns < 1 {
		c.MaxOpenConns = 100
	}
	if c.MaxIdleConns > c.MaxOpenConns {
		return fmt.Errorf("max-idle-conns (%d) 不能大于 max-open-conns (%d)", c.MaxIdleConns, c.MaxOpenConns)
	}

	// 根据驱动类型校验专属配置
	switch driver {
	case "mysql":
		if err := c.validateMySQL(); err != nil {
			return fmt.Errorf("mysql配置错误: %w", err)
		}
	case "postgres":
		if err := c.validatePostgres(); err != nil {
			return fmt.Errorf("postgres配置错误: %w", err)
		}
	case "sqlite":
		if err := c.validateSQLite(); err != nil {
			return fmt.Errorf("sqlite配置错误: %w", err)
		}
	}

	return nil
}

// validateMySQL 校验MySQL配置
func (c *Config) validateMySQL() error {
	if c.MySQL.Master.Host == "" {
		return fmt.Errorf("mysql主库主机不能为空")
	}
	if c.MySQL.Master.Port == 0 {
		c.MySQL.Master.Port = 3306
	}
	if c.MySQL.Master.Database == "" {
		return fmt.Errorf("mysql主库数据库不能为空")
	}
	if c.MySQL.Master.Username == "" {
		return fmt.Errorf("mysql主库用户名不能为空")
	}

	// 校验从库配置（如果有）
	for i, slave := range c.MySQL.Slaves {
		if slave.Host == "" {
			return fmt.Errorf("mysql从库[%d]主机不能为空", i)
		}
		if slave.Port == 0 {
			c.MySQL.Slaves[i].Port = 3306
		}
		if slave.Database == "" {
			return fmt.Errorf("mysql从库[%d]数据库不能为空", i)
		}
		if slave.Username == "" {
			return fmt.Errorf("mysql从库[%d]用户名不能为空", i)
		}
	}

	return nil
}

// validatePostgres 校验PostgreSQL配置
func (c *Config) validatePostgres() error {
	if c.Postgres.Master.Host == "" {
		return fmt.Errorf("postgres主库主机不能为空")
	}
	if c.Postgres.Master.Port == 0 {
		c.Postgres.Master.Port = 5432
	}
	if c.Postgres.Master.Database == "" {
		return fmt.Errorf("postgres主库数据库不能为空")
	}
	if c.Postgres.Master.Username == "" {
		return fmt.Errorf("postgres主库用户名不能为空")
	}

	// 校验从库配置（如果有）
	for i, slave := range c.Postgres.Slaves {
		if slave.Host == "" {
			return fmt.Errorf("postgres从库[%d]主机不能为空", i)
		}
		if slave.Port == 0 {
			c.Postgres.Slaves[i].Port = 5432
		}
		if slave.Database == "" {
			return fmt.Errorf("postgres从库[%d]数据库不能为空", i)
		}
		if slave.Username == "" {
			return fmt.Errorf("postgres从库[%d]用户名不能为空", i)
		}
	}

	return nil
}

// validateSQLite 校验SQLite配置
func (c *Config) validateSQLite() error {
	if c.SQLite.Path == "" {
		return fmt.Errorf("sqlite路径不能为空")
	}
	return nil
}

// LogLevel 获取日志级别
func (c *Config) LogLevel() logger.LogLevel {
	switch strings.ToLower(c.LogMode) {
	case "silent":
		return logger.Silent
	case "error":
		return logger.Error
	case "warn":
		return logger.Warn
	case "info":
		return logger.Info
	default:
		return logger.Info
	}
}

// GetSlowThreshold 获取慢查询阈值（毫秒）
func (c *Config) GetSlowThreshold() time.Duration {
	if c.SlowThreshold <= 0 {
		return 200 * time.Millisecond
	}
	return time.Duration(c.SlowThreshold) * time.Millisecond
}
