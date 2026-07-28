package gin

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	_ "github.com/earthyzinc/dehaze-go/docs" // swagger docs
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
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
	stack := cfg.System.Env != "prod"

	engine.Use(
		middleware.Trace(),
		middleware.DefaultLogger(),
		middleware.Recovery(stack),
		middleware.ContextErrorHandler(),
		middleware.Prometheus(),
		middleware.CorsByRules(),
	)

	// 健康检查
	engine.GET("/health", func(c *gin.Context) {
		common.Ok(c)
	})

	// Swagger 文档 (仅非生产环境启用)
	if cfg.System.Env != "prod" {
		engine.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	}

	// Prometheus 指标（内网免鉴权，通过网络层隔离保障安全）
	engine.GET("/metrics", middleware.MetricsHandler())

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

// WaitForShutdown 阻塞等待系统中断信号，返回接收到的信号
func (s *Server) WaitForShutdown() os.Signal {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	return sig
}

// Stop 优雅关闭 HTTP 服务（纯 HTTP shutdown，不包含信号监听）
func (s *Server) Stop(ctx context.Context) error {
	if s.server == nil {
		return nil
	}
	logger.Info("关闭WEB服务...")
	if err := s.server.Shutdown(ctx); err != nil {
		logger.Error("WEB服务关闭异常", zap.Error(err))
		return err
	}
	logger.Info("WEB服务已关闭")
	return nil
}
