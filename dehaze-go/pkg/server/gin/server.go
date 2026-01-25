package gin

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type Server struct {
	engine *gin.Engine
	server *http.Server
}

func Init() *Server {
	cfg := config.GetConfig()

	if cfg.System.Env == "prod" {
		gin.SetMode(gin.ReleaseMode)
	}

	engine := gin.New()

	// 添加中间件
	engine.Use(
		middleware.DefaultLogger(),
		gin.Recovery(),
		cors.New(cors.Config{
			AllowOrigins:     []string{"*"},
			AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
			AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization", "X-Trace-ID"},
			ExposeHeaders:    []string{"X-Trace-ID"},
			AllowCredentials: true,
			MaxAge:           12 * time.Hour,
		}),
	)

	engine.GET("/health", func(c *gin.Context) {
		common.Ok(c)
	})

	return &Server{
		engine: engine,
	}
}

func (s *Server) GetEngine() *gin.Engine {
	return s.engine
}

// Run 启动Gin服务（封装端口、超时配置）
func (s *Server) Run() error {
	cfg := config.GetConfig()
	address := fmt.Sprintf("%s:%d", cfg.System.Host, cfg.System.Port)
	// 创建服务
	s.server = &http.Server{
		Addr:           address,
		Handler:        s.engine,
		ReadTimeout:    10 * time.Minute,
		WriteTimeout:   10 * time.Minute,
		MaxHeaderBytes: 1 << 20,
	}

	return s.server.ListenAndServe()
}

func (s *Server) Stop() error {
	// 等待中断信号以优雅地关闭服务器
	quit := make(chan os.Signal, 1)
	// kill (无参数) 默认发送 syscall.SIGTERM
	// kill -2 发送 syscall.SIGINT
	// kill -9 发送 syscall.SIGKILL，但是无法被捕获，所以不需要添加
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	logger.Info("关闭WEB服务...")

	// 设置5秒的超时时间
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	defer cancel()

	if err := s.server.Shutdown(ctx); err != nil {
		logger.Fatal("WEB服务关闭异常", zap.Error(err))
		return err
	}

	logger.Info("WEB服务已关闭")
	return nil
}
