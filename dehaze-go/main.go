package main

import (
	"github.com/earthyzinc/dehaze-go/initialize"
)

//go:generate go env -w GO111MODULE=on
//go:generate go env -w GOPROXY=https://goproxy.cn,direct
//go:generate go mod tidy
//go:generate go mod download

func main() {
	// 加载配置文件
	initialize.Viper()
	// 初始化日志库
	initialize.Zap()
	// 初始化本地缓存
	initialize.LocalCache()
	// 初始化数据库
	initialize.Gorm()
	// 迁移数据库表
	initialize.Migrate()

	// 注册全局函数
	initialize.SetupHandlers()
	// 初始化路由
	initialize.Routers()
	// 初始化Redis
	initialize.Redis()
	// 初始化MongoDB

	// 初始化JWT黑名单和权限系统

	// 初始化web服务器
	initialize.Server()
}
