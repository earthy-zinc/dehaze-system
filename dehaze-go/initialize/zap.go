package initialize

import (
	"fmt"
	"os"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize/internal"
	"github.com/earthyzinc/dehaze-go/utils"
)

// Zap 获取 zap.Logger
// Author [SliverHorn](https://github.com/SliverHorn)
func Zap() {
	if ok, _ := utils.PathExists(global.CONFIG.Zap.Directory); !ok { // 判断是否有Directory文件夹
		fmt.Printf("创建日志文件夹： %v\n", global.CONFIG.Zap.Directory)
		_ = os.Mkdir(global.CONFIG.Zap.Directory, os.ModePerm)
	}
	levels := global.CONFIG.Zap.Levels()
	length := len(levels)
	cores := make([]zapcore.Core, 0, length)
	for i := range length {
		core := internal.NewZapCore(levels[i])
		cores = append(cores, core)
	}
	logger := zap.New(zapcore.NewTee(cores...))
	if global.CONFIG.Zap.ShowLine {
		logger = logger.WithOptions(zap.AddCaller())
	}
	zap.ReplaceGlobals(logger)
	global.LOG = logger
}
