package db

import (
	_ "github.com/go-sql-driver/mysql"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"

	"github.com/earthyzinc/dehaze-go/config"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize/internal"
)

// InitMysql 初始化Mysql数据库的辅助函数
func InitMysql(m config.DB) *gorm.DB {
	if m.Name == "" {
		global.LOG.Info("初始化数据库失败，未配置数据库名称")
		return nil
	}

	dsn := m.Username + ":" + m.Password + "@tcp(" + m.Path + ":" + m.Port + ")/" + m.Name + "?" + m.Config

	mysqlConfig := mysql.Config{
		DSN:                       dsn,   // DSN data source name
		DefaultStringSize:         191,   // string 类型字段的默认长度
		SkipInitializeWithVersion: false, // 根据版本自动配置
	}

	// 数据库配置
	if db, err := gorm.Open(mysql.New(mysqlConfig), internal.Gorm.Config(m)); err != nil {
		panic(err)
	} else {
		db.InstanceSet("gorm:table_options", "ENGINE="+m.Engine)
		sqlDB, _ := db.DB()
		sqlDB.SetMaxIdleConns(m.MaxIdleConns)
		sqlDB.SetMaxOpenConns(m.MaxOpenConns)
		return db
	}
}
