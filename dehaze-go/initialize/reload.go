package initialize

import (
	"github.com/earthyzinc/dehaze-go/global"
	"go.uber.org/zap"
)

// Reload 优雅地重新加载系统配置
func Reload() error {
	global.LOG.Info("正在重新加载系统配置...")

	// 重新加载配置文件
	if err := global.VIPER.ReadInConfig(); err != nil {
		global.LOG.Error("重新读取配置文件失败!", zap.Error(err))
		return err
	}

	// 重新初始化数据库连接
	if global.DB != nil {
		db, _ := global.DB.DB()
		err := db.Close()
		if err != nil {
			global.LOG.Error("关闭原数据库连接失败!", zap.Error(err))
			return err
		}
	}

	// 重新建立数据库连接
	Gorm()

	global.LOG.Info("系统配置重新加载完成")
	return nil
}
