package main

import (
	"github.com/earthyzinc/dehaze-go/pkg/app"
)

//go:generate go env -w GO111MODULE=on
//go:generate go env -w GOPROXY=https://goproxy.cn,direct
//go:generate go mod tidy
//go:generate go mod download
func main() {
	// // 加载配置文件
	// initialize.Viper()
	// // 初始化日志库
	// initialize.Zap()
	// // 初始化本地缓存
	// initialize.LocalCache()
	// // 初始化数据库
	// initialize.Gorm()
	// // 迁移数据库表
	// initialize.Migrate()

	// // 注册全局函数
	// initialize.SetupHandlers()
	// // 初始化路由
	// initialize.Routers()
	// // 初始化Redis
	// initialize.Redis()

	// // 初始化Job管理器
	// initialize.StartJobs()

	// // 初始化web服务器
	// initialize.Server()

	if err := app.Run(); err != nil {
		panic(err)
	}

}
