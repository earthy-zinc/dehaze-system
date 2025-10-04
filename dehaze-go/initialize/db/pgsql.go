package db

import (
	"path/filepath"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"

	"github.com/earthyzinc/dehaze-go/config"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize/internal"
)

// InitPostgreSQL 初始化PostgreSQL数据库的辅助函数
func InitPostgreSQL(m config.DB) *gorm.DB {
	if m.Name == "" {
		global.LOG.Info("初始化数据库失败，未配置数据库名称")
		return nil
	}

	dsn := filepath.Join(m.Path, m.Name+".db")

	postgreSQLConfig := postgres.Config{
		DSN:                  dsn, // DSN data source name
		PreferSimpleProtocol: false,
	}
	// 数据库配置
	if db, err := gorm.Open(postgres.New(postgreSQLConfig), internal.Gorm.Config(m)); err != nil {
		panic(err)
	} else {
		sqlDB, _ := db.DB()
		sqlDB.SetMaxIdleConns(m.MaxIdleConns)
		sqlDB.SetMaxOpenConns(m.MaxOpenConns)
		global.LOG.Info("成功连接PostgreSQL数据库")
		return db
	}
}
