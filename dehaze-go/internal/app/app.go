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
	auditlogrepo "github.com/earthyzinc/dehaze-go/internal/repository/audit_log"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	fbrepo "github.com/earthyzinc/dehaze-go/internal/repository/feedback"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	ihrepo "github.com/earthyzinc/dehaze-go/internal/repository/input_history"
	loginlogrepo "github.com/earthyzinc/dehaze-go/internal/repository/login_log"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	msgrepo "github.com/earthyzinc/dehaze-go/internal/repository/message"
	orderrepo "github.com/earthyzinc/dehaze-go/internal/repository/order"
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	taskrepo "github.com/earthyzinc/dehaze-go/internal/repository/task"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	"github.com/earthyzinc/dehaze-go/internal/router"
	algoservice "github.com/earthyzinc/dehaze-go/internal/service/algorithm"
	apikeyservice "github.com/earthyzinc/dehaze-go/internal/service/api_key"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	authservice "github.com/earthyzinc/dehaze-go/internal/service/auth"
	datasetservice "github.com/earthyzinc/dehaze-go/internal/service/dataset"
	deptservice "github.com/earthyzinc/dehaze-go/internal/service/dept"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	evalservice "github.com/earthyzinc/dehaze-go/internal/service/evaluation"
	fbservice "github.com/earthyzinc/dehaze-go/internal/service/feedback"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	importexportservice "github.com/earthyzinc/dehaze-go/internal/service/import_export"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export/handlers"
	ihservice "github.com/earthyzinc/dehaze-go/internal/service/input_history"
	loginlogservice "github.com/earthyzinc/dehaze-go/internal/service/login_log"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	menuservice "github.com/earthyzinc/dehaze-go/internal/service/menu"
	msgservice "github.com/earthyzinc/dehaze-go/internal/service/message"
	orderservice "github.com/earthyzinc/dehaze-go/internal/service/order"
	paymentsvc "github.com/earthyzinc/dehaze-go/internal/service/payment"
	pkgsaleservice "github.com/earthyzinc/dehaze-go/internal/service/pkgsale"
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
	"github.com/earthyzinc/dehaze-go/pkg/mongo"
	"github.com/earthyzinc/dehaze-go/pkg/mq"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
	dehazevalidator "github.com/earthyzinc/dehaze-go/pkg/validator"
	"github.com/earthyzinc/dehaze-go/pkg/websocket"
	"github.com/earthyzinc/dehaze-go/pkg/xxljob"
	gingin "github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson"
	mongodriver "go.mongodb.org/mongo-driver/mongo"
	"go.uber.org/zap"
)

// Application 应用核心上下文实例
// 目标：显式 wiring（构造函数注入）+ 清晰启动链路，避免运行时 DI 容器。
type Application struct {
	*gin.Server
	taskExecutor    taskservice.AsyncTaskExecutor
	consumer        *mq.Consumer
	publisher       *mq.Publisher
	auditLogService *auditlogservice.AuditLogService
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

	// 3.2) MongoDB（审计日志）
	if err := mongo.InitMongo(); err != nil {
		logger.Error("MongoDB 初始化失败，审计日志功能不可用", zap.Error(err))
	} else {
		a.initMongoIndexes()
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

	// message module repositories
	msgRepo := msgrepo.NewMessageRepository(gormDB)
	msgTplRepo := msgrepo.NewMessageTemplateRepository(gormDB)
	annRepo := msgrepo.NewAnnouncementRepository(gormDB)
	notifySettingRepo := msgrepo.NewNotificationSettingRepository(gormDB)
	userLookupRepo := msgrepo.NewUserLookupRepository(gormDB)

	// member module repositories
	memberRepo := memberrepo.NewMemberRepository(gormDB)
	memberBenefitRepo := memberrepo.NewMemberBenefitRepository(gormDB)
	memberGrowthLogRepo := memberrepo.NewMemberGrowthLogRepository(gormDB)
	memberSignInRepo := memberrepo.NewMemberSignInRepository(gormDB)

	// package & order module repositories
	packageRepo := pkgsalerepo.NewPackageRepository(gormDB)
	couponRepo := pkgsalerepo.NewCouponRepository(gormDB)
	userCouponRepo := pkgsalerepo.NewUserCouponRepository(gormDB)
	orderRepo := orderrepo.NewOrderRepository(gormDB)
	paymentRepo := orderrepo.NewPaymentRecordRepository(gormDB)
	refundRepo := orderrepo.NewRefundRecordRepository(gormDB)
	autoRenewRepo := orderrepo.NewAutoRenewRepository(gormDB)

	// feedback module repositories
	ratingRepo := fbrepo.NewRatingRepository(gormDB)
	feedbackRepo := fbrepo.NewFeedbackRepository(gormDB)
	feedbackReplyRepo := fbrepo.NewFeedbackReplyRepository(gormDB)

	// services

	// audit log services (MongoDB)
	mongoDB := mongo.GetMongoDatabase("")
	var loginLogService *loginlogservice.LoginLogService
	if mongoDB != nil {
		loginLogRepo := loginlogrepo.NewLoginLogRepository(mongoDB)
		auditLogRepo := auditlogrepo.NewAuditLogRepository(mongoDB)
		loginLogService = loginlogservice.NewLoginLogService(loginLogRepo)
		a.auditLogService = auditlogservice.NewAuditLogService(auditLogRepo)
	}
	userService := userservice.NewUserService(userRepo, roleRepo, deptRepo, menuRepo, a.auditLogService)
	authService := authservice.NewAuthService(cacheClient, userService, loginLogService, gormDB)
	algorithmService := algoservice.NewAlgorithmService(algorithmRepo, predLogRepo)
	menuService := menuservice.NewMenuService(cacheClient, menuRepo, roleRepo)
	roleService := roleservice.NewRoleService(cacheClient, roleRepo, menuRepo, a.auditLogService)
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

	importExportFileGenerator := importexportservice.NewFileGenerator()
	importExportTemplateMgr := importexportservice.NewTemplateManager(importExportFileGenerator)
	exportHandlers := []importexportservice.ExportHandler{
		handlers.NewUserExportHandler(gormDB),
		handlers.NewRoleExportHandler(gormDB),
		handlers.NewDeptExportHandler(gormDB),
		handlers.NewMenuExportHandler(gormDB),
		handlers.NewDictExportHandler(gormDB),
		handlers.NewDatasetExportHandler(gormDB, storageService),
		handlers.NewAlgorithmExportHandler(gormDB),
	}
	importHandlers := []importexportservice.ImportHandler{
		handlers.NewUserImportHandler(gormDB, deptRepo),
		handlers.NewRoleImportHandler(gormDB),
		handlers.NewDeptImportHandler(gormDB),
		handlers.NewMenuImportHandler(gormDB),
		handlers.NewDictImportHandler(gormDB),
		handlers.NewAlgorithmImportHandler(gormDB),
	}
	exportRegistry := importexportservice.NewExportHandlerRegistry(exportHandlers)
	importRegistry := importexportservice.NewImportHandlerRegistry(importHandlers)
	importExportService := importexportservice.NewImportExportService(
		exportRegistry,
		importRegistry,
		importExportFileGenerator,
		importExportTemplateMgr,
		storageService,
		taskService,
		importexportservice.NoOpVirusScanner{},
		zap.L(),
	)
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
		a.auditLogService,
	)
	taskApi := api.NewSysTaskApi(taskService)
	importExportApi := api.NewImportExportApi(importExportService)
	inputHistoryService := ihservice.NewInputHistoryService(inputHistoryRepo)
	algoClient := algo.NewClient(cfg.Algorithm)
	evalLogRepo := evalrepo.NewEvalLogRepository(gormDB)
	apiKeyService := apikeyservice.NewApiKeyService(apiKeyRepo, userService)

	// message module services
	messageService := msgservice.NewMessageService(msgRepo, msgTplRepo, userLookupRepo, cacheClient)
	announcementService := msgservice.NewAnnouncementService(annRepo, userLookupRepo, messageService)
	messageTemplateService := msgservice.NewMessageTemplateService(msgTplRepo)
	notificationSettingService := msgservice.NewNotificationSettingService(notifySettingRepo)

	// member module services（需在 predictionService 之前构造，预测/评估需调用权益校验）
	memberService := memberservice.NewMemberService(gormDB, memberRepo, memberBenefitRepo, memberGrowthLogRepo, memberSignInRepo, cacheClient, a.auditLogService, messageService)

	predictionService := predservice.NewPredictionService(predLogRepo, algorithmRepo, algoClient, cacheClient, memberService)
	evaluationService := evalservice.NewEvaluationService(evalLogRepo, algorithmRepo, algoClient, memberService)

	// package & order module services
	packageService := pkgsaleservice.NewPackageService(gormDB, packageRepo, couponRepo, userCouponRepo, memberBenefitRepo, cacheClient)
	couponService := pkgsaleservice.NewCouponService(gormDB, couponRepo, userCouponRepo, cacheClient)
	paymentSvc := paymentsvc.NewPaymentChannelService(cfg.Payment)
	orderService := orderservice.NewOrderService(gormDB, orderRepo, paymentRepo, refundRepo, autoRenewRepo, packageRepo, couponRepo, userCouponRepo, memberRepo, memberBenefitRepo, paymentSvc, cacheClient, a.auditLogService)

	// feedback module services
	var alertPublisher *mq.Publisher
	if cfg.RabbitMQ.Enabled {
		alertPublisher = mq.NewPublisher(cfg.RabbitMQ, zap.L())
		if err := alertPublisher.Connect(); err != nil {
			logger.Error("MQ Publisher（低分告警）连接失败，低分告警事件将无法发布", zap.Error(err))
		}
	}
	a.publisher = alertPublisher
	lowRatingAlertService := fbservice.NewLowRatingAlertService(gormDB, ratingRepo, messageService, alertPublisher, zap.L())
	ratingService := fbservice.NewRatingService(gormDB, ratingRepo, predLogRepo, memberService, cacheClient, lowRatingAlertService, zap.L())
	feedbackService := fbservice.NewFeedbackService(gormDB, feedbackRepo, feedbackReplyRepo, cacheClient)

	// 启动 MQ Consumer 消费死信队列与低分告警队列
	// 注意：Go 后端不消费 export 主队列（由 Java/Python 执行任务），
	// 仅消费 DLQ 以更新任务状态为 FAILED
	if cfg.RabbitMQ.Enabled {
		a.consumer = mq.NewConsumer(cfg.RabbitMQ, zap.L())
		if err := a.consumer.Connect(); err != nil {
			logger.Error("MQ Consumer 连接失败，死信队列将无法消费", zap.Error(err))
		} else {
		if err := a.consumer.ConsumeDLQ("task.export", taskService.HandleDLQMessage); err != nil {
			logger.Error("注册死信队列 Consumer 失败", zap.Error(err))
		}
		if err := a.consumer.Consume("feedback.low_rating", lowRatingAlertService.HandleMessage); err != nil {
			logger.Error("注册低分告警队列 Consumer 失败", zap.Error(err))
		}
			logger.Info("MQ Consumer 已启动，消费 export 死信队列与 feedback.low_rating 队列")
		}
	}

	// 启动 XXL-Job 执行器并注册定时任务
	xxlExecutor := xxljob.Init(cfg)
	if xxlExecutor != nil {
		job.InitJobs(xxlExecutor, storageService, predLogRepo, evalLogRepo, orderService, announcementService, messageService, memberService)
		go func() {
			if err := xxlExecutor.Run(); err != nil {
				logger.Error("XXL-Job 执行器运行失败", zap.Error(err))
			}
		}()
	}

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

	// message module apis
	messageApi := api.NewMessageApi(messageService)
	announcementApi := api.NewAnnouncementApi(announcementService)
	messageTemplateApi := api.NewMessageTemplateApi(messageTemplateService)
	notificationSettingApi := api.NewNotificationSettingApi(notificationSettingService)

	// member module apis
	memberApi := api.NewMemberApi(memberService)

	// package & order module apis
	packageApi := api.NewPackageApi(packageService, couponService)
	orderApi := api.NewOrderApi(orderService)
	paymentApi := api.NewPaymentApi(paymentSvc, orderService)

	// feedback module apis
	feedbackApi := api.NewFeedbackApi(ratingService, feedbackService)

	// routes
	engine := a.Server.GetEngine()

	// Readiness 探针 - 检查 DB/Redis/MQ 依赖
	engine.GET("/ready", a.readinessHandler())

	// WebSocket 端点（通过 query 参数 token 认证，不走 JWT 中间件）
	engine.GET("/ws", websocket.HandleWebSocket)

	v1 := engine.Group("/api/v1")
	// 全局 IP 限流兜底（使用 config.yaml 的 ip-limit-count/ip-limit-time）
	v1.Use(middleware.IPRateLimiter())

	// 公开路由（无需认证）
	router.RegisterNoAuthRoutes(v1, authApi)
	router.RegisterPaymentRoutes(v1, paymentApi)

	// 需要Session认证保护的路由
	protectedV1 := v1.Group("")
	protectedV1.Use(middleware.ApiKeyAuthMiddleware())
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
	router.RegisterImportExportRoutes(protectedV1, importExportApi)
	router.RegisterImageInputRoutes(protectedV1, inputHistoryApi)
	router.RegisterPredictionRoutes(protectedV1, predictionApi)
	router.RegisterEvaluationRoutes(protectedV1, evaluationApi)
	router.RegisterApiKeyRoutes(protectedV1, apiKeyApi)
	router.RegisterMessageRoutes(protectedV1, messageApi)
	router.RegisterNotificationSettingRoutes(protectedV1, notificationSettingApi)
	router.RegisterAnnouncementRoutes(protectedV1, announcementApi)
	router.RegisterMessageTemplateRoutes(protectedV1, messageTemplateApi)
	router.RegisterMemberRoutes(protectedV1, memberApi)
	router.RegisterPackageRoutes(protectedV1, packageApi)
	router.RegisterOrderRoutes(protectedV1, orderApi)
	router.RegisterFeedbackRoutes(protectedV1, feedbackApi)

	middleware.ApiKeyAuth = func(ctx context.Context, rawKey string) (*security.CustomClaims, error) {
		authInfo, err := apiKeyService.AuthenticateByKey(ctx, rawKey)
		if err != nil {
			return nil, err
		}
		claims := security.CreateClaims(authInfo)
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
	xxljob.Stop()

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

	// 4.1) 低分告警 MQ Publisher
	if a.publisher != nil {
		if err := a.publisher.Close(); err != nil {
			logger.Error("关闭低分告警 MQ Publisher 失败", zap.Error(err))
			errs = append(errs, fmt.Errorf("AlertPublisher: %w", err))
		} else {
			logger.Info("低分告警 MQ Publisher 已关闭")
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

	// 6.1) MongoDB
	if err := mongo.Close(); err != nil {
		logger.Error("关闭MongoDB连接失败", zap.Error(err))
		errs = append(errs, fmt.Errorf("MongoDB: %w", err))
	}

	// 7) 日志（最后刷新，保证上面的日志都写入）
	logger.Info("所有资源已关闭，刷新日志缓冲区")
	logger.Sync()

	if len(errs) > 0 {
		return fmt.Errorf("优雅关闭时发生 %d 个错误: %v", len(errs), errs)
	}
	return nil
}

// initMongoIndexes 创建 MongoDB 索引
func (a *Application) initMongoIndexes() {
	db := mongo.GetMongoDatabase("")
	if db == nil {
		return
	}

	loginLogIndexes := []mongodriver.IndexModel{
		{Keys: bson.D{{Key: "userId", Value: 1}, {Key: "createTime", Value: -1}}},
		{Keys: bson.D{{Key: "createTime", Value: -1}}},
		{Keys: bson.D{{Key: "status", Value: 1}}},
	}
	if _, err := db.Collection("login_log").Indexes().CreateMany(context.Background(), loginLogIndexes); err != nil {
		logger.Error("创建login_log索引失败", zap.Error(err))
	}

	auditLogIndexes := []mongodriver.IndexModel{
		{Keys: bson.D{{Key: "operatorId", Value: 1}, {Key: "createTime", Value: -1}}},
		{Keys: bson.D{{Key: "targetType", Value: 1}, {Key: "targetId", Value: 1}, {Key: "createTime", Value: -1}}},
		{Keys: bson.D{{Key: "module", Value: 1}, {Key: "createTime", Value: -1}}},
	}
	if _, err := db.Collection("audit_log").Indexes().CreateMany(context.Background(), auditLogIndexes); err != nil {
		logger.Error("创建audit_log索引失败", zap.Error(err))
	}
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
