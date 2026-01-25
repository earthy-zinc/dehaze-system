package database

import (
	"database/sql"
	"fmt"
	"regexp"
	"strings"

	"gorm.io/gorm"
	"gorm.io/gorm/schema"
)

// MaskDSN 隐藏DSN中的密码（用于日志输出）
// 将密码部分替换为 ***
func MaskDSN(dsn string) string {
	// MySQL: 使用更精确的 DSN 解析
	// 格式: username:password@protocol(address)/dbname?param=value

	// 简单的字符串替换可能不够准确
	// 考虑使用 URL 解析器
	if strings.Contains(dsn, "@tcp(") || strings.Contains(dsn, "@unix(") {
		parts := strings.SplitN(dsn, "@", 2)
		if len(parts) == 2 {
			auth := strings.SplitN(parts[0], ":", 2)
			if len(auth) == 2 && auth[1] != "" {
				return auth[0] + ":***@" + parts[1]
			}
		}
	}

	// PostgreSQL DSN
	if strings.Contains(dsn, "password=") {
		re := regexp.MustCompile(`password=[^\s&]+`)
		return re.ReplaceAllString(dsn, "password=***")
	}

	return dsn
}

// SetupConnectionPool 设置数据库连接池参数
func SetupConnectionPool(sqlDB *sql.DB, config *Config) {
	if sqlDB == nil {
		return
	}

	// 设置最大空闲连接数
	sqlDB.SetMaxIdleConns(config.MaxIdleConns)

	// 设置最大打开连接数
	sqlDB.SetMaxOpenConns(config.MaxOpenConns)

	// 设置连接最大生命周期
	if config.ConnMaxLifetime > 0 {
		sqlDB.SetConnMaxLifetime(config.ConnMaxLifetime)
	}

	// 设置连接最大空闲时间
	if config.ConnMaxIdleTime > 0 {
		sqlDB.SetConnMaxIdleTime(config.ConnMaxIdleTime)
	}
}

// GetGormConfig 获取通用Gorm配置
// 所有数据库实现都使用此配置
func GetGormConfig(config *Config) *gorm.Config {
	return &gorm.Config{
		// 日志配置
		Logger: NewGormLogger(config),

		// 命名策略
		NamingStrategy: schema.NamingStrategy{
			TablePrefix:   config.Prefix,   // 表名前缀
			SingularTable: config.Singular, // 是否使用单数表名
		},

		// 禁用外键约束（迁移时）
		DisableForeignKeyConstraintWhenMigrating: true,

		// 预编译语句缓存（提升性能）
		PrepareStmt: true,
	}
}

// BuildMySQLDSN 构建MySQL DSN
func BuildMySQLDSN(instance MySQLInstanceConfig, charset, loc string, parseTime bool) string {
	// 构建配置参数
	params := []string{}

	// 字符集
	if charset != "" {
		params = append(params, fmt.Sprintf("charset=%s", charset))
	}

	// 时区
	if loc != "" {
		params = append(params, fmt.Sprintf("loc=%s", loc))
	}

	// 是否解析时间
	if parseTime {
		params = append(params, "parseTime=True")
	}

	// 额外配置
	if instance.Config != "" {
		params = append(params, instance.Config)
	}

	// 构建DSN
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s",
		instance.Username,
		instance.Password,
		instance.Host,
		instance.Port,
		instance.Database,
	)

	if len(params) > 0 {
		dsn += "?" + strings.Join(params, "&")
	}

	return dsn
}

// BuildPostgresDSN 构建PostgreSQL DSN
func BuildPostgresDSN(instance PostgresInstanceConfig, sslMode string) string {
	// 构建配置参数
	params := []string{
		fmt.Sprintf("host=%s", instance.Host),
		fmt.Sprintf("port=%d", instance.Port),
		fmt.Sprintf("user=%s", instance.Username),
		fmt.Sprintf("dbname=%s", instance.Database),
	}

	// 密码（可选）
	if instance.Password != "" {
		params = append(params, fmt.Sprintf("password=%s", instance.Password))
	}

	// SSL模式
	if sslMode != "" {
		params = append(params, fmt.Sprintf("sslmode=%s", sslMode))
	}

	// 额外配置
	if instance.Config != "" {
		params = append(params, instance.Config)
	}

	return strings.Join(params, " ")
}

// ValidateConnection 验证数据库连接是否正常
func ValidateConnection(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("数据库实例为空")
	}

	sqlDB, err := db.DB()
	if err != nil {
		return fmt.Errorf("获取sql.DB失败: %w", err)
	}

	if err := sqlDB.Ping(); err != nil {
		return fmt.Errorf("连接数据库失败: %w", err)
	}

	return nil
}

// CloseDB 关闭数据库连接
func CloseDB(db *gorm.DB) error {
	if db == nil {
		return nil
	}

	sqlDB, err := db.DB()
	if err != nil {
		return fmt.Errorf("获取sql.DB失败: %w", err)
	}

	return sqlDB.Close()
}
