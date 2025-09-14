package initialize

import (
	"flag"
	"fmt"
	"os"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize/internal"
	"github.com/earthyzinc/dehaze-go/utils"
	"github.com/fsnotify/fsnotify"
	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

// getConfigPath 获取配置文件路径, 优先级: 命令行 > 环境变量 > 默认值
func getConfigPath() (config string) {
	// `-c` flag parse
	flag.StringVar(&config, "c", "", "配置文件的路径")
	flag.Parse()
	if config != "" {
		// 命令行参数不为空 将值赋值于config
		fmt.Printf("您正在使用命令行的 '-c' 参数传递的值, config 的路径为 %s\n", config)
		return
	}
	if env := os.Getenv(internal.ConfigEnv); env != "" {
		// 判断环境变量 DEHAZE_CONFIG
		config = env
		fmt.Printf("您正在使用 %s 环境变量, config 的路径为 %s\n", internal.ConfigEnv, config)
		return
	}

	switch gin.Mode() {
	// 根据 gin 模式文件名
	case gin.DebugMode:
		config = internal.ConfigDebugFile
	case gin.ReleaseMode:
		config = internal.ConfigReleaseFile
	case gin.TestMode:
		config = internal.ConfigTestFile
	}

	fmt.Printf("您正在使用 gin 的 %s 模式运行, config 的路径为 %s\n", gin.Mode(), config)

	_, err := os.Stat(config)
	if err != nil || os.IsNotExist(err) {
		config = internal.ConfigDefaultFile
		fmt.Printf("配置文件路径不存在, 使用默认配置文件路径: %s\n", config)
	}

	return
}

func Viper() {
	global.VALIDATE = validator.New()

	config := getConfigPath()
	v := viper.New()
	v.SetConfigFile(config)
	v.SetConfigType("yaml")
	err := v.ReadInConfig()
	if err != nil {
		panic(fmt.Errorf("读取配置文件发生了错误: %w", err))
	}

	v.WatchConfig()

	v.OnConfigChange(func(e fsnotify.Event) {
		fmt.Println("配置文件已变化:", e.Name)
		if err = v.Unmarshal(&global.CONFIG); err != nil {
			fmt.Println(err)
		}

		// 校验配置
		if err := global.VALIDATE.Struct(global.CONFIG); err != nil {
			// 校验失败时处理错误
			fmt.Println("配置校验失败:", err)
			global.LOG.Error("配置校验失败", zap.Error(err))
			panic(err)
		}

		err := utils.GlobalSystemEvents.TriggerReload()
		if err != nil {
			global.LOG.Error("重载系统失败!", zap.Error(err))
			return
		}
	})
	if err = v.Unmarshal(&global.CONFIG); err != nil {
		panic(fmt.Errorf("反序列化配置时发生了错误: %w", err))
	}

	// 校验配置
	if err := global.VALIDATE.Struct(global.CONFIG); err != nil {
		// 校验失败时处理错误
		fmt.Println("配置校验失败:", err)
		global.LOG.Error("配置校验失败", zap.Error(err))
		panic(err)
	}

	global.VIPER = v
}
