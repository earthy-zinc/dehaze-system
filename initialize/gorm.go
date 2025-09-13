package initialize

import (
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize/db"
)

func Gorm() {
	global.ACTIVE_DB_NAME = &global.CONFIG.Db.Name

	switch global.CONFIG.Db.Type {
	case "mysql":
		global.DB = db.InitMysql(global.CONFIG.Db)
	default:
		global.DB = db.InitMysql(global.CONFIG.Db)
	}
}
