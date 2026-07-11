package app

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/api"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
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
	menuservice "github.com/earthyzinc/dehaze-go/internal/service/menu"
	roleservice "github.com/earthyzinc/dehaze-go/internal/service/role"
	taskservice "github.com/earthyzinc/dehaze-go/internal/service/task"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/postgres"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/sqlite"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
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
	datasetRepo := datasetrepo.NewDatasetRepository(gormDB)
	datasetItemRepo := datasetrepo.NewDatasetItemRepository(gormDB)
	datasetStatsRepo := datasetrepo.NewDatasetStatsRepository(gormDB)
	datasetItemFileRepo := datasetrepo.NewDatasetItemFileRepository(gormDB)
	itemFileRepo := filerepo.NewItemFileRepository(gormDB)
	fileRepo := filerepo.NewFileRepository(gormDB)
	taskRepo := taskrepo.NewTaskRepository(gormDB)

	// services
	userService := userservice.NewUserService(userRepo, roleRepo, deptRepo, menuRepo)
	authService := authservice.NewAuthService(cacheClient, userService)
	algorithmService := algoservice.NewAlgorithmService(algorithmRepo)
	menuService := menuservice.NewMenuService(cacheClient, menuRepo)
	roleService := roleservice.NewRoleService(cacheClient, roleRepo, menuRepo)
	deptService := deptservice.NewDeptService(cacheClient, deptRepo)
	dictTypeService := dictservice.NewDictTypeService(gormDB, dictTypeRepo, dictRepo, cacheClient)
	dictService := dictservice.NewDictService(dictRepo, dictTypeRepo, cacheClient)
	fileService := fileservice.NewFileService(fileRepo)
	a.taskExecutor = taskservice.NewAsyncTaskExecutor(config.GetConfig().RabbitMQ, zap.L())
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
	_ = taskService

	// apis
	authApi := api.NewAuthApi(authService)
	sysUserApi := api.NewSysUserApi(userService)
	sysRoleApi := api.NewSysRoleApi(roleService)
	sysDeptApi := api.NewSysDeptApi(deptService)
	sysDictApi := api.NewSysDictApi(dictService, dictTypeService)
	sysMenuApi := api.NewSysMenuApi(menuService)
	algorithmApi := api.NewAlgorithmApi(algorithmService)
	datasetApi := api.NewSysDatasetApi(datasetService, datasetOperationService)
	datasetItemApi := api.NewSysDatasetItemApi(datasetItemService, datasetOperationService)
	itemFileApi := api.NewSysItemFileApi(itemFileService)
	fileApi := api.NewSysFileApi(fileService)

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
