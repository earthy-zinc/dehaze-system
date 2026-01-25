package app

import (
	"fmt"
	"net/http"
	"os"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin"
	"go.uber.org/zap"
)

// Application 应用核心上下文实例
type Application struct {
	*gin.Server
}

func New() *Application {
	return &Application{}
}

func Run() error {
	app := New()
	if err := app.Init(); err != nil {
		return err
	}
	if err := app.Run(); err != nil {
		return err
	}
	return nil
}

// Init 统一初始化所有核心组件
func (a *Application) Init() error {
	// 初始化配置和日志
	logger.InitDefaultLogger()
	if _, err := config.Init(); err != nil {
		return err
	}

	if err := logger.Init(); err != nil {
		return err
	}

	// 初始化服务器
	a.Server = gin.Init()

	// 初始化数据库

	// 初始化缓存

	// 注册全局函数

	// 初始化路由

	// 初始化定时任务

	return nil
}

// Run 启动所有服务
func (a *Application) Run() error {
	go func() {
		if err := a.Server.Run(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("listen: %s\n", err)
			logger.Error("WEB服务启动失败", zap.Error(err))
			os.Exit(1)
		}
	}()

	// 关闭定时任务

	return a.Server.Stop()
}
