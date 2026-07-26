package app

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/api"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	afrepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm_favorite"
	apikeyrepo "github.com/earthyzinc/dehaze-go/internal/repository/api_key"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	ihrepo "github.com/earthyzinc/dehaze-go/internal/repository/input_history"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	taskrepo "github.com/earthyzinc/dehaze-go/internal/repository/task"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	"github.com/earthyzinc/dehaze-go/internal/router"
	algoservice "github.com/earthyzinc/dehaze-go/internal/service/algorithm"
	apikeyservice "github.com/earthyzinc/dehaze-go/internal/service/api_key"
	authservice "github.com/earthyzinc/dehaze-go/internal/service/auth"
	datasetservice "github.com/earthyzinc/dehaze-go/internal/service/dataset"
	deptservice "github.com/earthyzinc/dehaze-go/internal/service/dept"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	evalservice "github.com/earthyzinc/dehaze-go/internal/service/evaluation"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	ihservice "github.com/earthyzinc/dehaze-go/internal/service/input_history"
	menuservice "github.com/earthyzinc/dehaze-go/internal/service/menu"
	predservice "github.com/earthyzinc/dehaze-go/internal/service/prediction"
	roleservice "github.com/earthyzinc/dehaze-go/internal/service/role"
	taskservice "github.com/earthyzinc/dehaze-go/internal/service/task"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/postgres"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/sqlite"
	"github.com/earthyzinc/dehaze-go/pkg/job"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/mq"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
	dehazevalidator "github.com/earthyzinc/dehaze-go/pkg/validator"
	"github.com/earthyzinc/dehaze-go/pkg/websocket"
	gingin "github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// Application 应用核心上下文实例
// 目标：显式 wiring（构造函数注入）+ 清晰启动链路，避免运行时 DI 容器。
type Application struct {
	*gin.Server
	taskExecutor taskservice.AsyncTaskExecutor
	consumer     *mq.Consumer
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
	// 1) 配置与日志
	logger.InitDefaultLogger()
	if _, err := config.Init(); err != nil {
		return err
	}
	if err := logger.Init(); err != nil {
		return err
	}

	// 2) 数据库（注意：需导入 driver 触发 RegisterFactory）
	if err := database.Init(database.GetDatabaseConfig()); err != nil {
		return err
	}

	// 3) 缓存（内部会按配置决定是否初始化 Redis/本地缓存）
	if _, err := cache.Init(); err != nil {
		return err
	}

	// 3.1) WebSocket 管理器（依赖 Redis Pub/Sub）
	if redisClient := redis.GetClient(); redisClient != nil {
		if _, err := websocket.InitManager(redisClient); err != nil {
			logger.Error("WebSocket 管理器初始化失败", zap.Error(err))
		}
	}

	// 4) HTTP Server
	a.Server = gin.Init()

	// 5) 初始化 validator 中文翻译（需在 Gin 引擎创建后调用）
	dehazevalidator.Init()

	// 6) 显式 wiring：repo -> service -> api -> router
	gormDB := database.DB()
	cacheClient := cache.GetCache()

	// repositories
	userRepo := userrepo.NewUserRepository(gormDB)
	roleRepo := rolerepo.NewRoleRepository(gormDB)
	menuRepo := menurepo.NewMenuRepository(gormDB)
	deptRepo := deptrepo.NewDeptRepository(gormDB)
	dictTypeRepo := dictrepo.NewDictTypeRepository(gormDB)
	dictRepo := dictrepo.NewDictRepository(gormDB)
	algorithmRepo := algorepo.NewAlgorithmRepository(gormDB)
	algorithmFavRepo := afrepo.NewRepository(gormDB)
	datasetRepo := datasetrepo.NewDatasetRepository(gormDB)
	datasetItemRepo := datasetrepo.NewDatasetItemRepository(gormDB)
	datasetStatsRepo := datasetrepo.NewDatasetStatsRepository(gormDB)
	datasetItemFileRepo := datasetrepo.NewDatasetItemFileRepository(gormDB)
	itemFileRepo := filerepo.NewItemFileRepository(gormDB)
	fileRepo := filerepo.NewFileRepository(gormDB)
	taskRepo := taskrepo.NewTaskRepository(gormDB)
	inputHistoryRepo := ihrepo.NewInputHistoryRepository(gormDB)
	predLogRepo := predrepo.NewPredLogRepository(gormDB)
	apiKeyRepo := apikeyrepo.NewApiKeyRepository(gormDB)

	// services
	userService := userservice.NewUserService(userRepo, roleRepo, deptRepo, menuRepo)
	authService := authservice.NewAuthService(cacheClient, userService, gormDB)
	algorithmService := algoservice.NewAlgorithmService(algorithmRepo, predLogRepo)
	menuService := menuservice.NewMenuService(cacheClient, menuRepo, roleRepo)
	roleService := roleservice.NewRoleService(cacheClient, roleRepo, menuRepo)
	deptService := deptservice.NewDeptService(cacheClient, deptRepo)
	dictTypeService := dictservice.NewDictTypeService(gormDB, dictTypeRepo, dictRepo, cacheClient)
	dictService := dictservice.NewDictService(dictRepo, dictTypeRepo, cacheClient)
	// 存储服务（根据配置选择 MinIO 或本地存储）
	cfg := config.GetConfig()
	storageService, err := storage.NewStorage(cfg.File.Type, cfg.File.MinIO, cfg.File.Local)
	if err != nil {
		return fmt.Errorf("初始化存储服务失败: %w", err)
	}
	fileService := fileservice.NewFileService(fileRepo, storageService)
	a.taskExecutor = taskservice.NewAsyncTaskExecutor(cfg.RabbitMQ, zap.L())
	if err := a.taskExecutor.Initialize(); err != nil {
		return err
	}
	taskExecutor := a.taskExecutor
	taskService := taskservice.NewTaskService(taskRepo, datasetRepo, cacheClient, zap.L(), taskExecutor)
	itemFileService := fileservice.NewItemFileService(cacheClient, itemFileRepo, datasetItemRepo, fileService, taskExecutor)
	datasetService := datasetservice.NewDatasetService(cacheClient, datasetRepo, datasetItemRepo, datasetStatsRepo, itemFileRepo, fileRepo)
	datasetItemService := datasetservice.NewDatasetItemService(cacheClient, datasetItemRepo, datasetRepo, itemFileRepo, fileRepo, itemFileService)
	datasetOperationService := datasetservice.NewDatasetOperationService(
		cacheClient,
		datasetRepo,
		datasetItemRepo,
		datasetItemFileRepo,
		itemFileRepo,
		fileRepo,
		taskExecutor,
	)
	taskApi := api.NewSysTaskApi(taskService)
	inputHistoryService := ihservice.NewInputHistoryService(inputHistoryRepo)
	algoClient := algo.NewClient(cfg.Algorithm)
	predictionService := predservice.NewPredictionService(predLogRepo, algorithmRepo, algoClient, cacheClient)
	evalLogRepo := evalrepo.NewEvalLogRepository(gormDB)
	evaluationService := evalservice.NewEvaluationService(evalLogRepo, algorithmRepo, algoClient)
	apiKeyService := apikeyservice.NewApiKeyService(apiKeyRepo, userService)

	// 启动 MQ Consumer 消费死信队列
	// 注意：Go 后端不消费 export 主队列（由 Java/Python 执行任务），
	// 仅消费 DLQ 以更新任务状态为 FAILED
	if cfg.RabbitMQ.Enabled {
		a.consumer = mq.NewConsumer(cfg.RabbitMQ, zap.L())
		if err := a.consumer.Connect(); err != nil {
			logger.Error("MQ Consumer 连接失败，死信队列将无法消费", zap.Error(err))
		} else {
			if err := a.consumer.ConsumeDLQ("export", taskService.HandleDLQMessage); err != nil {
				logger.Error("注册死信队列 Consumer 失败", zap.Error(err))
			}
			logger.Info("MQ Consumer 已启动，消费 export 死信队列")
		}
	}

	// 启动定时清理任务（清理过期任务、失败记录、预测/评估僵尸任务等）
	job.InitJobs(storageService, predLogRepo, evalLogRepo)

	// apis
	authApi := api.NewAuthApi(authService)
	sysUserApi := api.NewSysUserApi(userService)
	sysRoleApi := api.NewSysRoleApi(roleService)
	sysDeptApi := api.NewSysDeptApi(deptService)
	sysDictApi := api.NewSysDictApi(dictService, dictTypeService)
	sysMenuApi := api.NewSysMenuApi(menuService)
	algorithmApi := api.NewAlgorithmApi(algorithmService, algorithmFavRepo)
	datasetApi := api.NewSysDatasetApi(datasetService, datasetOperationService)
	datasetItemApi := api.NewSysDatasetItemApi(datasetItemService, datasetOperationService, fileService)
	itemFileApi := api.NewSysItemFileApi(itemFileService, fileService)
	fileApi := api.NewSysFileApi(fileService)
	inputHistoryApi := api.NewSysInputHistoryApi(inputHistoryService)
	predictionApi := api.NewSysPredictionApi(predictionService)
	evaluationApi := api.NewSysEvaluationApi(evaluationService)
	apiKeyApi := api.NewApiKeyApi(apiKeyService)

	// routes
	engine := a.Server.GetEngine()

	// Readiness 探针 - 检查 DB/Redis/MQ 依赖
	engine.GET("/ready", a.readinessHandler())

	// WebSocket 端点（通过 query 参数 token 认证，不走 JWT 中间件）
	engine.GET("/ws", websocket.HandleWebSocket)

	v1 := engine.Group("/api/v1")

	// 公开路由（无需认证）
	router.RegisterNoAuthRoutes(v1, authApi)

	// 需要Session认证保护的路由
	protectedV1 := v1.Group("")
	protectedV1.Use(middleware.SessionAuth())
	protectedV1.Use(middleware.UserContextMiddleware())
	router.RegisterAuthRoutes(protectedV1, authApi)
	router.RegisterSysUserRoutes(protectedV1, sysUserApi)
	router.RegisterSysRoleRoutes(protectedV1, sysRoleApi)
	router.RegisterSysDeptRoutes(protectedV1, sysDeptApi)
	router.RegisterSysDictRoutes(protectedV1, sysDictApi)
	router.RegisterSysMenuRoutes(protectedV1, sysMenuApi)
	router.RegisterDatasetRoutes(protectedV1, datasetApi)
	router.RegisterFileRoutes(protectedV1, fileApi)
	router.RegisterDatasetItemRoutes(protectedV1, datasetItemApi)
	router.RegisterItemFileRoutes(protectedV1, itemFileApi)
	router.RegisterAlgorithmRoutes(protectedV1, algorithmApi)
	router.RegisterTaskRoutes(protectedV1, taskApi)
	router.RegisterImageInputRoutes(protectedV1, inputHistoryApi)
	router.RegisterPredictionRoutes(protectedV1, predictionApi)
	router.RegisterEvaluationRoutes(protectedV1, evaluationApi)
	router.RegisterApiKeyRoutes(protectedV1, apiKeyApi)

	middleware.ApiKeyAuth = func(ctx context.Context, rawKey string) (*security.CustomClaims, error) {
		authInfo, err := apiKeyService.AuthenticateByKey(ctx, rawKey)
		if err != nil {
			return nil, err
		}
		j := security.NewJWT()
		claims := j.CreateClaims(authInfo)
		return &claims, nil
	}

	return nil
}

// Run 启动所有服务并阻塞等待关闭信号
func (a *Application) Run() error {
	errCh := make(chan error, 1)
	go func() {
		if err := a.Server.Run(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	sigCh := make(chan os.Signal, 1)
	go func() {
		sigCh <- a.Server.WaitForShutdown()
	}()

	select {
	case err := <-errCh:
		logger.Error("WEB服务启动失败", zap.Error(err))
		_ = a.shutdown()
		return err
	case sig := <-sigCh:
		logger.Info("接收到关闭信号，开始优雅关闭...", zap.String("signal", sig.String()))
		return a.shutdown()
	}
}

// shutdown 按依赖反序关闭所有资源
func (a *Application) shutdown() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var errs []error

	// 0) 停止 WebSocket 管理器
	if wsManager := websocket.GetManager(); wsManager != nil {
		wsManager.Stop()
	}

	// 1) 停止定时任务
	job.StopJobs()

	// 2) HTTP Server
	if err := a.Server.Stop(ctx); err != nil {
		errs = append(errs, fmt.Errorf("HTTP Server: %w", err))
	}

	// 3) MQ Consumer
	if a.consumer != nil {
		if err := a.consumer.Close(); err != nil {
			logger.Error("关闭 MQ Consumer 失败", zap.Error(err))
			errs = append(errs, fmt.Errorf("MQConsumer: %w", err))
		} else {
			logger.Info("MQ Consumer 已关闭")
		}
	}

	// 4) 异步任务执行器（RabbitMQ Publisher）
	if a.taskExecutor != nil {
		if err := a.taskExecutor.Shutdown(); err != nil {
			logger.Error("关闭任务执行器失败", zap.Error(err))
			errs = append(errs, fmt.Errorf("TaskExecutor: %w", err))
		} else {
			logger.Info("任务执行器已关闭")
		}
	}

	// 5) 缓存
	if cm := cache.GetCacheManager(); cm != nil {
		if err := cm.Close(); err != nil {
			logger.Error("关闭缓存失败", zap.Error(err))
			errs = append(errs, fmt.Errorf("Cache: %w", err))
		}
	}

	// 6) 数据库
	if err := database.Close(); err != nil {
		logger.Error("关闭数据库连接失败", zap.Error(err))
		errs = append(errs, fmt.Errorf("Database: %w", err))
	} else {
		logger.Info("数据库连接已关闭")
	}

	// 7) 日志（最后刷新，保证上面的日志都写入）
	logger.Info("所有资源已关闭，刷新日志缓冲区")
	logger.Sync()

	if len(errs) > 0 {
		return fmt.Errorf("优雅关闭时发生 %d 个错误: %v", len(errs), errs)
	}
	return nil
}

// readinessHandler Readiness 探针处理函数
// 检查 DB/Redis/RabbitMQ 依赖，任一不可用返回 503
func (a *Application) readinessHandler() gingin.HandlerFunc {
	return func(c *gingin.Context) {
		ctx := c.Request.Context()
		components := make(map[string]string)
		allHealthy := true

		// DB check
		func() {
			db := database.DB()
			if db == nil {
				components["db"] = "DOWN"
				allHealthy = false
				return
			}
			sqlDB, err := db.DB()
			if err != nil {
				components["db"] = "DOWN"
				allHealthy = false
				return
			}
			if err := sqlDB.PingContext(ctx); err != nil {
				components["db"] = "DOWN"
				allHealthy = false
				return
			}
			components["db"] = "UP"
		}()

		// Redis check
		func() {
			redisClient := redis.GetClient()
			if redisClient == nil {
				components["redis"] = "DOWN"
				allHealthy = false
				return
			}
			if err := redisClient.Ping(ctx).Err(); err != nil {
				components["redis"] = "DOWN"
				allHealthy = false
				return
			}
			components["redis"] = "UP"
		}()

		// RabbitMQ check（仅当启用时检查 Consumer 与 Publisher）
		cfg := config.GetConfig()
		if cfg.RabbitMQ.Enabled {
			consumerOK := a.consumer != nil && a.consumer.IsConnected()
			publisherOK := a.taskExecutor != nil && a.taskExecutor.IsConnected()
			if !consumerOK || !publisherOK {
				components["rabbitmq"] = "DOWN"
				allHealthy = false
			} else {
				components["rabbitmq"] = "UP"
			}
		}

		status := "UP"
		code := http.StatusOK
		if !allHealthy {
			status = "DOWN"
			code = http.StatusServiceUnavailable
		}
		c.JSON(code, gingin.H{
			"status":     status,
			"components": components,
		})
	}
}
