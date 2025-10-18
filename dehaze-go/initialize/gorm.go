package initialize

import (
	"os"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize/db"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func Gorm() {
	global.ACTIVE_DB_NAME = &global.CONFIG.Db.Name

	switch global.CONFIG.Db.Type {
	case "mysql":
		global.DB = db.InitMysql(global.CONFIG.Db)
	case "sqlite":
		global.DB = db.InitSQLite(global.CONFIG.Db)
	case "postgresql":
		global.DB = db.InitPostgreSQL(global.CONFIG.Db)
	default:
		global.DB = db.InitMysql(global.CONFIG.Db)
	}
}

func Migrate() {
	database := global.DB
	if database == nil {
		return
	}

	if gin.Mode() == gin.ReleaseMode {
		return
	}

	global.LOG.Info("当前处于开发/测试环境，迁移数据库中")

	err := database.AutoMigrate(
		model.SysUser{},
		model.SysRole{},

		model.SysDept{},
		model.SysMenu{},

		model.SysDict{},
		model.SysDictType{},

		model.SysAlgorithm{},

		model.SysDataset{},
		model.SysDatasetItem{},

		model.SysEvalLog{},
		model.SysPredLog{},

		model.SysFile{},
		model.SysWpxFile{},

		model.SysOperationRecord{},
	)

	if err != nil {
		global.LOG.Error("迁移数据库表失败", zap.Error(err))
		os.Exit(0)
	}

}
