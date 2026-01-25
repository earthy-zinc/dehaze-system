package options

import (
	"time"

	"gorm.io/gorm/logger"
)

type DsnProvider interface {
	Dsn() string
}

type DB struct {
	Driver            string                  `mapstructure:"driver" json:"driver" yaml:"driver" validate:"required,oneof=mysql postgres sqlite"` // 数据库驱动
	Prefix            string                  `mapstructure:"prefix" json:"prefix" yaml:"prefix"`
	Engine            string                  `mapstructure:"engine" json:"engine" yaml:"engine" default:"InnoDB"`
	LogMode           string                  `mapstructure:"log-mode" json:"log-mode" yaml:"log-mode" validate:"oneof=silent error warn info"`
	MaxIdleConns      int                     `mapstructure:"max-idle-conns" json:"max-idle-conns" yaml:"max-idle-conns" validate:"min=1" default:"10"`
	MaxOpenConns      int                     `mapstructure:"max-open-conns" json:"max-open-conns" yaml:"max-open-conns" validate:"min=1" default:"100"`
	Singular          bool                    `mapstructure:"singular" json:"singular" yaml:"singular"`
	LogZap            bool                    `mapstructure:"log-zap" json:"log-zap" yaml:"log-zap"`
	SlowThreshold     int                     `mapstructure:"slow-threshold" json:"slow-threshold" yaml:"slow-threshold" default:"200"`                // 慢查询阈值（毫秒）
	ConnMaxLifetime   string                  `mapstructure:"conn-max-lifetime" json:"conn-max-lifetime" yaml:"conn-max-lifetime" default:"3600s"`     // 连接最大生命周期
	ConnMaxIdleTime   string                  `mapstructure:"conn-max-idle-time" json:"conn-max-idle-time" yaml:"conn-max-idle-time" default:"600s"`   // 连接最大空闲时间
	DefaultStringSize uint                    `mapstructure:"default-string-size" json:"default-string-size" yaml:"default-string-size" default:"191"` // string类型字段默认长度
	MySQL             *MySQLInstanceConfig    `mapstructure:"mysql" json:"mysql" yaml:"mysql"`
	Postgres          *PostgresInstanceConfig `mapstructure:"postgres" json:"postgres" yaml:"postgres"`
	SQLite            *SQLiteConfig           `mapstructure:"sqlite" json:"sqlite" yaml:"sqlite"`
}

type MySQLInstanceConfig struct {
	Host      string `mapstructure:"host" json:"host" yaml:"host" validate:"required"`                 // 主机地址
	Port      int    `mapstructure:"port" json:"port" yaml:"port" validate:"required,min=1,max=65535"` // 端口
	Database  string `mapstructure:"database" json:"database" yaml:"database" validate:"required"`     // 数据库名
	Username  string `mapstructure:"username" json:"username" yaml:"username" validate:"required"`     // 用户名
	Password  string `mapstructure:"password" json:"password" yaml:"password"`                         // 密码
	Config    string `mapstructure:"config" json:"config" yaml:"config"`                               // 额外配置参数
	Charset   string `mapstructure:"charset" json:"charset" yaml:"charset" default:"utf8mb4"`          // 字符集
	ParseTime bool   `mapstructure:"parse-time" json:"parse-time" yaml:"parse-time" default:"true"`    // 是否解析时间
	Loc       string `mapstructure:"loc" json:"loc" yaml:"loc" default:"Local"`                        // 时区
}

type PostgresInstanceConfig struct {
	Host     string `mapstructure:"host" json:"host" yaml:"host" validate:"required"`                 // 主机地址
	Port     int    `mapstructure:"port" json:"port" yaml:"port" validate:"required,min=1,max=65535"` // 端口
	Database string `mapstructure:"database" json:"database" yaml:"database" validate:"required"`     // 数据库名
	Username string `mapstructure:"username" json:"username" yaml:"username" validate:"required"`     // 用户名
	Password string `mapstructure:"password" json:"password" yaml:"password"`                         // 密码
	Config   string `mapstructure:"config" json:"config" yaml:"config"`                               // 额外配置参数
	SSLMode  string `mapstructure:"ssl-mode" json:"ssl-mode" yaml:"ssl-mode" default:"disable"`       // SSL模式
}

type SQLiteConfig struct {
	Path string `mapstructure:"path" json:"path" yaml:"path" validate:"required"` // 数据库文件路径
}

func (c DB) LogLevel() logger.LogLevel {
	switch c.LogMode {
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

func (c DB) GetSlowThreshold() time.Duration {
	if c.SlowThreshold <= 0 {
		return 200 * time.Millisecond
	}
	return time.Duration(c.SlowThreshold) * time.Millisecond
}

func (c DB) GetConnMaxLifetime() time.Duration {
	if c.ConnMaxLifetime == "" {
		return 3600 * time.Second
	}
	duration, _ := time.ParseDuration(c.ConnMaxLifetime)
	return duration
}

func (c DB) GetConnMaxIdleTime() time.Duration {
	if c.ConnMaxIdleTime == "" {
		return 600 * time.Second
	}
	duration, _ := time.ParseDuration(c.ConnMaxIdleTime)
	return duration
}
