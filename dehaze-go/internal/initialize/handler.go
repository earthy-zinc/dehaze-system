package initialize

import "github.com/earthyzinc/dehaze-go/pkg/utils"

func SetupHandlers() {
	// 注册系统重载处理函数
	utils.GlobalSystemEvents.RegisterReloadHandler(func() error {
		return Reload()
	})
}
