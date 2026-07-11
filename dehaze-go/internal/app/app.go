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
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	ihrepo "github.com/earthyzinc/dehaze-go/internal/repository/input_history"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	taskrepo "github.com/earthyzinc/dehaze-go/internal/repository/task"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	"github.com/earthyzinc/dehaze-go/internal/router"
	algoservice "github.com/earthyzinc/dehaze-go/internal/service/algorithm"
	authservice "github.com/earthyzinc/dehaze-go/internal/service/auth"
	datasetservice "github.com/earthyzinc/dehaze-go/internal/service/dataset"
	deptservice "github.com/earthyzinc/dehaze-go/internal/service/dept"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	ihservice "github.com/earthyzinc/dehaze-go/internal/service/input_history"
	menuservice "github.com/earthyzinc/dehaze-go/internal/service/menu"
	predservice "github.com/earthyzinc/dehaze-go/internal/service/prediction"
	evalservice "github.com/earthyzinc/dehaze-go/internal/service/evaluation"
	roleservice "github.com/earthyzinc/dehaze-go/internal/service/role"
	taskservice "github.com/earthyzinc/dehaze-go/internal/service/task"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/postgres"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/sqlite"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
	dehazevalidator "github.com/earthyzinc/dehaze-go/pkg/validator"
	"go.uber.org/zap"
)

// Application 应用核心上下文实例
// 目标：显式 wiring（构造函数注入）+ 清晰启动链路，避免运行时 DI 容器。
type Application struct {
	*gin.Server
	taskExecutor taskservice.AsyncTaskExecutor
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
	cache.Init()

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

	// services
	userService := userservice.NewUserService(userRepo, roleRepo, deptRepo, menuRepo)
	authService := authservice.NewAuthService(cacheClient, userService)
	algorithmService := algoservice.NewAlgorithmService(algorithmRepo)
	menuService := menuservice.NewMenuService(cacheClient, menuRepo)
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
	taskApi := api.NewSysTaskApi(taskService, taskRepo)
	inputHistoryService := ihservice.NewInputHistoryService(inputHistoryRepo)
	algoClient := algo.NewClient(cfg.Algorithm)
	predLogRepo := predrepo.NewPredLogRepository(gormDB)
	predictionService := predservice.NewPredictionService(predLogRepo, algoClient, cacheClient)
	evalLogRepo := evalrepo.NewEvalLogRepository(gormDB)
	evaluationService := evalservice.NewEvaluationService(evalLogRepo, algoClient)

	// apis
	authApi := api.NewAuthApi(authService)
	sysUserApi := api.NewSysUserApi(userService)
	sysRoleApi := api.NewSysRoleApi(roleService)
	sysDeptApi := api.NewSysDeptApi(deptService)
	sysDictApi := api.NewSysDictApi(dictService, dictTypeService)
	sysMenuApi := api.NewSysMenuApi(menuService)
	algorithmApi := api.NewAlgorithmApi(algorithmService, algorithmFavRepo)
	datasetApi := api.NewSysDatasetApi(datasetService, datasetOperationService)
	datasetItemApi := api.NewSysDatasetItemApi(datasetItemService, datasetOperationService)
	itemFileApi := api.NewSysItemFileApi(itemFileService, fileService)
	fileApi := api.NewSysFileApi(fileService)
	inputHistoryApi := api.NewSysInputHistoryApi(inputHistoryService)
	predictionApi := api.NewSysPredictionApi(predictionService)
	evaluationApi := api.NewSysEvaluationApi(evaluationService)

	// routes
	engine := a.Server.GetEngine()
	v1 := engine.Group("/api/v1")

	// 公开路由（无需认证）
	router.RegisterNoAuthRoutes(v1, authApi)

	// 需要JWT认证保护的路由
	protectedV1 := v1.Group("")
	protectedV1.Use(middleware.JWTAuth())
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

	return nil
}

// Run 启动所有服务并阻塞等待关闭信号
func (a *Application) Run() error {
	go func() {
		if err := a.Server.Run(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("listen: %s\n", err)
			logger.Error("WEB服务启动失败", zap.Error(err))
			os.Exit(1)
		}
	}()

	// 阻塞等待系统中断信号
	sig := a.Server.WaitForShutdown()
	logger.Info("接收到关闭信号，开始优雅关闭...", zap.String("signal", sig.String()))

	return a.shutdown()
}

// shutdown 按依赖反序关闭所有资源
func (a *Application) shutdown() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var errs []error

	// 1) HTTP Server
	if err := a.Server.Stop(ctx); err != nil {
		errs = append(errs, fmt.Errorf("HTTP Server: %w", err))
	}

	// 2) 异步任务执行器（RabbitMQ）
	if a.taskExecutor != nil {
		if err := a.taskExecutor.Shutdown(); err != nil {
			logger.Error("关闭任务执行器失败", zap.Error(err))
			errs = append(errs, fmt.Errorf("TaskExecutor: %w", err))
		} else {
			logger.Info("任务执行器已关闭")
		}
	}

	// 3) 缓存
	if cm := cache.GetCacheManager(); cm != nil {
		if err := cm.Close(); err != nil {
			logger.Error("关闭缓存失败", zap.Error(err))
			errs = append(errs, fmt.Errorf("Cache: %w", err))
		}
	}

	// 4) 数据库
	if err := database.Close(); err != nil {
		logger.Error("关闭数据库连接失败", zap.Error(err))
		errs = append(errs, fmt.Errorf("Database: %w", err))
	} else {
		logger.Info("数据库连接已关闭")
	}

	// 5) 日志（最后刷新，保证上面的日志都写入）
	logger.Info("所有资源已关闭，刷新日志缓冲区")
	logger.Sync()

	if len(errs) > 0 {
		return fmt.Errorf("优雅关闭时发生 %d 个错误: %v", len(errs), errs)
	}
	return nil
}
