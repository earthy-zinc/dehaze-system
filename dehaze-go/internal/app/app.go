package app

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/api"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	aidomainrepo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	afrepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm_favorite"
	apikeyrepo "github.com/earthyzinc/dehaze-go/internal/repository/api_key"
	auditlogrepo "github.com/earthyzinc/dehaze-go/internal/repository/audit_log"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	favrepo "github.com/earthyzinc/dehaze-go/internal/repository/favorite"
	fbrepo "github.com/earthyzinc/dehaze-go/internal/repository/feedback"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	ihrepo "github.com/earthyzinc/dehaze-go/internal/repository/input_history"
	kborepo "github.com/earthyzinc/dehaze-go/internal/repository/kb"
	loginlogrepo "github.com/earthyzinc/dehaze-go/internal/repository/login_log"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	msgrepo "github.com/earthyzinc/dehaze-go/internal/repository/message"
	orderrepo "github.com/earthyzinc/dehaze-go/internal/repository/order"
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	presetrepo "github.com/earthyzinc/dehaze-go/internal/repository/preset"
	recrepo "github.com/earthyzinc/dehaze-go/internal/repository/recommendation"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	taskrepo "github.com/earthyzinc/dehaze-go/internal/repository/task"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	"github.com/earthyzinc/dehaze-go/internal/router"
	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	aidomainservice "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	algoservice "github.com/earthyzinc/dehaze-go/internal/service/algorithm"
	algoSelectService "github.com/earthyzinc/dehaze-go/internal/service/algorithm_select"
	apikeyservice "github.com/earthyzinc/dehaze-go/internal/service/api_key"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	authservice "github.com/earthyzinc/dehaze-go/internal/service/auth"
	compareservice "github.com/earthyzinc/dehaze-go/internal/service/compare"
	datasetservice "github.com/earthyzinc/dehaze-go/internal/service/dataset"
	deptservice "github.com/earthyzinc/dehaze-go/internal/service/dept"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	evalservice "github.com/earthyzinc/dehaze-go/internal/service/evaluation"
	favoriteservice "github.com/earthyzinc/dehaze-go/internal/service/favorite"
	fbservice "github.com/earthyzinc/dehaze-go/internal/service/feedback"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	importexportservice "github.com/earthyzinc/dehaze-go/internal/service/import_export"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export/handlers"
	ihservice "github.com/earthyzinc/dehaze-go/internal/service/input_history"
	kbservice "github.com/earthyzinc/dehaze-go/internal/service/kb"
	loginlogservice "github.com/earthyzinc/dehaze-go/internal/service/login_log"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	menuservice "github.com/earthyzinc/dehaze-go/internal/service/menu"
	msgservice "github.com/earthyzinc/dehaze-go/internal/service/message"
	orderservice "github.com/earthyzinc/dehaze-go/internal/service/order"
	paymentsvc "github.com/earthyzinc/dehaze-go/internal/service/payment"
	pkgsaleservice "github.com/earthyzinc/dehaze-go/internal/service/pkgsale"
	predservice "github.com/earthyzinc/dehaze-go/internal/service/prediction"
	presetservice "github.com/earthyzinc/dehaze-go/internal/service/preset"
	recservice "github.com/earthyzinc/dehaze-go/internal/service/recommendation"
	roleservice "github.com/earthyzinc/dehaze-go/internal/service/role"
	taskservice "github.com/earthyzinc/dehaze-go/internal/service/task"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	"github.com/earthyzinc/dehaze-go/pkg/aiclient"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/postgres"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/sqlite"
	"github.com/earthyzinc/dehaze-go/pkg/job"
	"github.com/earthyzinc/dehaze-go/pkg/lifecycle"
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
	lifecycleMgr    *lifecycle.Manager
	defaultStorage  storage.StorageService
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

	// 应用级生命周期管理器（异步 goroutine context + 优雅关闭等待）
	a.lifecycleMgr = lifecycle.NewManager()

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
	presetRepo := presetrepo.NewPresetRepository(gormDB)
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
	promotionRepo := pkgsalerepo.NewPromotionRepository(gormDB)
	orderRepo := orderrepo.NewOrderRepository(gormDB)
	paymentRepo := orderrepo.NewPaymentRecordRepository(gormDB)
	refundRepo := orderrepo.NewRefundRecordRepository(gormDB)
	autoRenewRepo := orderrepo.NewAutoRenewRepository(gormDB)

	// favorite module repositories
	favoriteRepo := favrepo.NewFavoriteRepository(gormDB)

	// feedback module repositories
	ratingRepo := fbrepo.NewRatingRepository(gormDB)
	feedbackRepo := fbrepo.NewFeedbackRepository(gormDB)
	feedbackReplyRepo := fbrepo.NewFeedbackReplyRepository(gormDB)

	// recommendation module repositories
	recommendationRepo := recrepo.NewRecommendationRepository(gormDB)
	recommendationRuleRepo := recrepo.NewRuleRepository(gormDB)

	// services

	// audit log services (MongoDB)
	mongoDB := mongo.GetMongoDatabase("")
	var loginLogService *loginlogservice.LoginLogService
	// 审计仓储提到 if 外：member 操作日志端点也需要它（Mongo 不可用时保持 nil，端点回空列表）
	var auditLogRepo *auditlogrepo.AuditLogRepository
	if mongoDB != nil {
		loginLogRepo := loginlogrepo.NewLoginLogRepository(mongoDB)
		auditLogRepo = auditlogrepo.NewAuditLogRepository(mongoDB)
		loginLogService = loginlogservice.NewLoginLogService(loginLogRepo)
		a.auditLogService = auditlogservice.NewAuditLogService(auditLogRepo)
	}
	userService := userservice.NewUserService(userRepo, roleRepo, deptRepo, menuRepo, memberRepo, a.auditLogService)
	menuService := menuservice.NewMenuService(cacheClient, menuRepo, roleRepo)
	roleService := roleservice.NewRoleService(cacheClient, roleRepo, menuRepo, a.auditLogService)
	deptService := deptservice.NewDeptService(cacheClient, deptRepo)
	dictTypeService := dictservice.NewDictTypeService(gormDB, dictTypeRepo, dictRepo, cacheClient)
	dictService := dictservice.NewDictService(dictRepo, dictTypeRepo, cacheClient)
	// favorite module services（需在 algorithmService/datasetOperationService 之前构造：对象删除时标记收藏失效）
	favoriteService := favoriteservice.NewFavoriteService(favoriteRepo, memberRepo, algorithmRepo, predLogRepo, datasetRepo, dictService)
	algorithmService := algoservice.NewAlgorithmService(algorithmRepo, predLogRepo, favoriteService)
	// 存储服务注册表（根据配置构建所有存储后端实例）
	cfg := config.GetConfig()
	storageRegistry, err := storage.NewRegistry(cfg.File)
	if err != nil {
		return fmt.Errorf("初始化存储服务失败: %w", err)
	}
	// 默认存储后端实例（供仅需单一后端的组件使用：导入导出、数据集导出、定时任务）
	defaultStorage, err := storageRegistry.Default()
	if err != nil {
		return fmt.Errorf("获取默认存储后端失败: %w", err)
	}
	a.defaultStorage = defaultStorage
	fileService := fileservice.NewFileService(fileRepo, storageRegistry)
	a.taskExecutor = taskservice.NewAsyncTaskExecutor(cfg.RabbitMQ, zap.L())
	if err := a.taskExecutor.Initialize(); err != nil {
		return err
	}
	taskExecutor := a.taskExecutor
	taskService := taskservice.NewTaskService(taskRepo, datasetRepo, cacheClient, zap.L(), taskExecutor, storageRegistry)
	itemFileService := fileservice.NewItemFileService(cacheClient, itemFileRepo, datasetItemRepo, fileService, taskExecutor, taskService)

	importExportFileGenerator := importexportservice.NewFileGenerator()
	importExportTemplateMgr := importexportservice.NewTemplateManager(importExportFileGenerator)
	exportHandlers := []importexportservice.ExportHandler{
		handlers.NewUserExportHandler(gormDB),
		handlers.NewRoleExportHandler(gormDB),
		handlers.NewDeptExportHandler(gormDB),
		handlers.NewMenuExportHandler(gormDB),
		handlers.NewDictExportHandler(gormDB),
		handlers.NewDatasetExportHandler(gormDB, defaultStorage),
		handlers.NewAlgorithmExportHandler(gormDB),
	}
	importHandlers := []importexportservice.ImportHandler{
		handlers.NewUserImportHandler(gormDB, deptRepo, cfg.System.DefaultPassword),
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
		defaultStorage,
		taskService,
		importexportservice.NoOpVirusScanner{},
		zap.L(),
	)
	datasetService := datasetservice.NewDatasetService(cacheClient, datasetRepo, datasetItemRepo, datasetStatsRepo, itemFileRepo, fileRepo)
	datasetItemService := datasetservice.NewDatasetItemService(cacheClient, datasetItemRepo, datasetRepo, itemFileRepo, fileRepo, fileService, itemFileService)
	datasetOperationService := datasetservice.NewDatasetOperationService(
		cacheClient,
		datasetRepo,
		datasetItemRepo,
		datasetItemFileRepo,
		itemFileRepo,
		fileRepo,
		fileService,
		taskExecutor,
		taskService,
		a.auditLogService,
		favoriteService,
	)
	taskApi := api.NewSysTaskApi(taskService)
	importExportApi := api.NewImportExportApi(importExportService)
	inputHistoryService := ihservice.NewInputHistoryService(inputHistoryRepo, memberRepo, memberBenefitRepo)
	algoClient, err := algo.NewClient(cfg.Algorithm)
	if err != nil {
		return fmt.Errorf("初始化算法客户端失败: %w", err)
	}
	aiClient, err := aiclient.New(cfg.AI)
	if err != nil {
		return fmt.Errorf("初始化 AI 转发客户端失败: %w", err)
	}
	evalLogRepo := evalrepo.NewEvalLogRepository(gormDB)
	// 模型白名单存在性校验需读 sys_ai_model（对齐 python ApiKeyService._validate_whitelist）
	apiKeyService := apikeyservice.NewApiKeyService(apiKeyRepo, userService, airepo.NewModelRepository(gormDB))

	// message module services
	messageService := msgservice.NewMessageService(msgRepo, msgTplRepo, userLookupRepo, cacheClient)
	announcementService := msgservice.NewAnnouncementService(annRepo, userLookupRepo, messageService)
	messageTemplateService := msgservice.NewMessageTemplateService(msgTplRepo)
	notificationSettingService := msgservice.NewNotificationSettingService(notifySettingRepo)

	// member module services（需在 predictionService/authService 之前构造，预测/评估/注册需调用权益校验）
	// aiBillingRepo 同时供 member 试用引导（source='trial' 积分汇总）与下方 AI 计费域使用
	aiBillingRepo := airepo.NewBillingRepository(gormDB)
	memberTrialDeps := trialStatusDepsAdapter{coupons: userCouponRepo, billing: aiBillingRepo, packages: packageRepo}
	var memberAuditLister memberservice.AuditLogLister
	if auditLogRepo != nil {
		memberAuditLister = auditLogListerAdapter{repo: auditLogRepo}
	}
	// aiBillingService 同时供 member 权益概览的 AI 类目（余额/今日已用）与下方计费域 A 类端点使用
	aiBillingService := aiservice.NewBillingService(gormDB, aiBillingRepo, redis.GetClient())
	memberService := memberservice.NewMemberService(gormDB, memberRepo, memberBenefitRepo, memberGrowthLogRepo, memberSignInRepo, cacheClient, a.auditLogService, messageService, a.lifecycleMgr, dictService, memberTrialDeps, memberAuditLister, aiCreditsAdapter{billing: aiBillingService}, packageOverridesAdapter{packages: packageRepo})

	// authService 依赖 userService + memberService（注册流程通过 UserService 创建用户、MemberService 初始化会员）
	authService := authservice.NewAuthService(cacheClient, userService, loginLogService, memberService)

	predictionService := predservice.NewPredictionService(predLogRepo, algorithmRepo, algoClient, cacheClient, memberService, a.lifecycleMgr)
	evaluationService := evalservice.NewEvaluationService(evalLogRepo, algorithmRepo, algoClient, memberService)

	// package & order module services
	packageService := pkgsaleservice.NewPackageService(gormDB, packageRepo, couponRepo, userCouponRepo, memberBenefitRepo, cacheClient)
	couponService := pkgsaleservice.NewCouponService(gormDB, couponRepo, userCouponRepo, memberRepo, userRepo, cacheClient)
	promotionService := pkgsaleservice.NewPromotionService(gormDB, promotionRepo, cacheClient)
	paymentSvc := paymentsvc.NewPaymentChannelService(cfg.Payment)
	orderService := orderservice.NewOrderService(gormDB, orderRepo, paymentRepo, refundRepo, autoRenewRepo, packageRepo, couponRepo, userCouponRepo, userRepo, memberRepo, memberBenefitRepo, paymentSvc, cacheClient, a.auditLogService, memberService)

	// algorithm select module services
	algorithmSelectService := algoSelectService.NewAlgorithmSelectService(algorithmRepo, predLogRepo, ratingRepo, predictionService)

	// feedback module services
	var alertPublisher *mq.Publisher
	if cfg.RabbitMQ.Enabled {
		alertPublisher = mq.NewPublisher(cfg.RabbitMQ, zap.L())
		if err := alertPublisher.Connect(); err != nil {
			logger.Error("MQ Publisher（低分告警）连接失败，低分告警事件将无法发布", zap.Error(err))
		}
	}
	a.publisher = alertPublisher
	lowRatingAlertService := fbservice.NewLowRatingAlertService(ratingRepo, userRepo, algorithmRepo, messageService, alertPublisher, zap.L())
	ratingService := fbservice.NewRatingService(gormDB, ratingRepo, predLogRepo, memberService, cache.GetCacheManager().GetL2Cache(), lowRatingAlertService, zap.L(), dictService)
	// 反馈/评价的日限计数器在 python 端直接存 Redis（测试与运维会直接操作该键），
	// 必须绕过多级缓存的 L1 本地层，否则外部清理 Redis 后计数器仍滞留 L1
	feedbackService := fbservice.NewFeedbackService(gormDB, feedbackRepo, feedbackReplyRepo, userRepo, cache.GetCacheManager().GetL2Cache())
	recommendationService := recservice.NewRecommendationService(recommendationRepo, recommendationRuleRepo, algorithmRepo)
	presetService := presetservice.NewPresetService(gormDB, presetRepo)
	presetservice.SeedSystemPresets(gormDB)
	compareService := compareservice.NewCompareService(evalLogRepo, predLogRepo, algorithmRepo)

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
			logger.Debug("MQ Consumer 已启动，消费 export 死信队列与 feedback.low_rating 队列")
		}
	}

	// 启动 XXL-Job 执行器并注册定时任务
	xxlExecutor := xxljob.Init(cfg)
	if xxlExecutor != nil {
		job.InitJobs(xxlExecutor, defaultStorage, predLogRepo, evalLogRepo, orderService, announcementService, messageService, memberService, cacheClient)
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
	algorithmApi := api.NewAlgorithmApi(algorithmService, algorithmFavRepo, importExportApi)
	datasetApi := api.NewSysDatasetApi(datasetService, datasetOperationService)
	datasetItemApi := api.NewSysDatasetItemApi(datasetItemService, datasetOperationService, fileService)
	itemFileApi := api.NewSysItemFileApi(itemFileService, fileService)
	fileApi := api.NewSysFileApi(fileService)
	inputHistoryApi := api.NewSysInputHistoryApi(inputHistoryService)
	predictionApi := api.NewSysPredictionApi(predictionService, fileService)
	evaluationApi := api.NewSysEvaluationApi(evaluationService)
	apiKeyApi := api.NewApiKeyApi(apiKeyService)

	// message module apis
	messageApi := api.NewMessageApi(messageService)
	announcementApi := api.NewAnnouncementApi(announcementService)
	messageTemplateApi := api.NewMessageTemplateApi(messageTemplateService)
	notificationSettingApi := api.NewNotificationSettingApi(notificationSettingService)

	// member module apis
	memberApi := api.NewMemberApi(memberService, orderService)

	// package & order module apis
	packageApi := api.NewPackageApi(packageService, couponService)
	promotionApi := api.NewPromotionApi(promotionService)
	orderApi := api.NewOrderApi(orderService)
	paymentApi := api.NewPaymentApi(paymentSvc, orderService)

	// favorite module apis
	favoriteApi := api.NewFavoriteApi(favoriteService)

	// algorithm select module apis
	algorithmSelectApi := api.NewAlgorithmSelectApi(algorithmSelectService)

	// feedback module apis
	feedbackApi := api.NewFeedbackApi(ratingService, feedbackService)

	// frontend log module api
	clientLogApi := api.NewClientLogApi()

	// recommendation module apis
	recommendationApi := api.NewRecommendationApi(recommendationService)

	// preset module apis
	presetApi := api.NewSysPresetApi(presetService)

	// compare module apis
	compareApi := api.NewCompareApi(compareService)

	// AI 域 B 类端点转发（行为在 dehaze-python）
	aiProxyApi := api.NewAIProxyApi(aiClient)

	// AI 域 A 类端点（原生实现，共享 MySQL/Redis，行为对齐 dehaze-python）
	aiModelRepo := airepo.NewModelRepository(gormDB)
	aiProviderRepo := airepo.NewProviderRepository(gormDB)
	aiProviderKeyRepo := airepo.NewProviderKeyRepository(gormDB)
	aiModelPriceRepo := airepo.NewModelPriceRepository(gormDB)
	aiMcpRepo := airepo.NewMcpRepository(gormDB)
	aiSkillRepo := airepo.NewSkillRepository(gormDB)
	aiHealthService := aiservice.NewHealthService(gormDB)
	aiModelService := aiservice.NewModelService(gormDB, aiModelRepo, aiModelPriceRepo, aiHealthService, messageService)
	aiProviderService := aiservice.NewProviderService(aiProviderRepo, aiProviderKeyRepo, aiHealthService)
	aiMcpService := aiservice.NewMcpServerService(gormDB, aiMcpRepo)
	aiSkillService := aiservice.NewSkillService(aiSkillRepo, storageRegistry)
	aiModelApi := api.NewAiModelApi(aiModelService)
	aiProviderApi := api.NewAiProviderApi(aiProviderService)
	aiMcpApi := api.NewAiMcpApi(aiMcpService)
	aiSkillApi := api.NewAiSkillApi(aiSkillService)
	aiKbApi := api.NewAiKbApi(kbservice.NewService(kborepo.NewRepository(gormDB)))
	aiA2AApi := api.NewAiA2AApi(aiservice.NewA2AService(airepo.NewA2ARepository(gormDB)))
	// 兼容调用审计走 MongoDB（ai_api_call_log）；采集不可用时端点内返回中间件服务错误，不摘路由
	var aiCompatAuditRepo *airepo.CompatAuditRepository
	if mongoDB != nil {
		aiCompatAuditRepo = airepo.NewCompatAuditRepository(mongoDB)
	}
	aiCompatApi := api.NewAiCompatAuditApi(aiservice.NewCompatAuditService(aiCompatAuditRepo))

	// AI 对话域 A 类端点（会话/消息/反馈/产物/记忆/Agent 与版本/A2A 端点/评测中心/定时任务）
	aiConvRepo := aidomainrepo.NewConversationRepository(gormDB)
	aiMsgRepo := aidomainrepo.NewMessageRepository(gormDB)
	aiAgentRepo := aidomainrepo.NewAgentRepository(gormDB)
	aiEvalRepo := aidomainrepo.NewEvalRepository(gormDB)
	aiMemoryRepo := aidomainrepo.NewMemoryRepository(gormDB)
	aiConversationApi := api.NewAiConversationApi(
		aidomainservice.NewConversationService(aiConvRepo, aiMsgRepo, aiAgentRepo),
		aidomainservice.NewMessageService(aiConvRepo, aiMsgRepo, aidomainrepo.NewThoughtRepository(gormDB)),
		aidomainservice.NewFeedbackService(aiMsgRepo, aidomainrepo.NewMessageFeedbackRepository(gormDB), aiMemoryRepo),
		aidomainservice.NewArtifactService(aidomainrepo.NewArtifactRepository(gormDB), aiConvRepo, aiMsgRepo, storageRegistry),
	)
	aiMemoryApi := api.NewAiMemoryApi(aidomainservice.NewMemoryService(aiMemoryRepo, a.auditLogService))
	aiEvalCenterService := aidomainservice.NewEvalCenterService(aiEvalRepo, aiAgentRepo)
	aiEvalApi := api.NewAiEvalApi(
		aidomainservice.NewEvalService(aiEvalRepo, aiAgentRepo, a.auditLogService),
		aiEvalCenterService,
	)
	aiAgentApi := api.NewAiAgentApi(
		aidomainservice.NewAgentService(aiAgentRepo, aiEvalRepo, a.auditLogService),
		aidomainservice.NewAgentVersionService(aiAgentRepo, a.auditLogService),
		aidomainservice.NewEndpointService(aiAgentRepo, a.auditLogService),
	)
	aiScheduleApi := api.NewAiScheduleApi(aidomainservice.NewScheduleService(aidomainrepo.NewScheduleRepository(gormDB)))
	aiUsageApi := api.NewAiUsageApi(aidomainservice.NewUsageService(aidomainrepo.NewUsageRepository(gormDB)))

	// AI 计费与可观测性域 A 类端点（原生实现，直接读写共享 MySQL/Redis）
	aiBillingApi := api.NewAiBillingApi(
		aiBillingService,
		aiservice.NewCostService(airepo.NewCostRepository(gormDB)),
	)
	aiObservabilityApi := api.NewAiObservabilityApi(aiservice.NewObservabilityService(
		airepo.NewObservabilityRepository(gormDB),
		aiBillingRepo,
		aiConvRepo,
		aiMsgRepo,
		aidomainrepo.NewThoughtRepository(gormDB),
	))

	// routes
	engine := a.Server.GetEngine()

	// Readiness 探针 - 检查 DB/Redis/MQ 依赖
	engine.GET("/ready", a.readinessHandler())

	// WebSocket 端点（通过 query 参数 token 认证，不走 JWT 中间件）
	engine.GET("/ws", websocket.HandleWebSocket)
	// 语音流式 ASR 的 WebSocket 转发（鉴权与帧协议事实源在 python，见 voice_proxy.go）
	router.RegisterVoiceWSProxyRoute(engine, aiProxyApi)

	// AI 兼容 API（OpenAI/Claude）与 A2A 标准入口：鉴权由 python 全局 ApiKeyAuthMiddleware 决定，
	// Go 侧不能先拦（x-api-key 协议入口过 AuthMiddleware 会一律 401）
	router.RegisterAICompatRoutes(engine, aiProxyApi)

	v1 := engine.Group("/api/v1")
	// 全局 IP 限流兜底（使用 config.yaml 的 ip-limit-count/ip-limit-time）
	v1.Use(middleware.IPRateLimiter())

	// 公开路由（无需认证）
	router.RegisterNoAuthRoutes(v1, authApi)
	router.RegisterPaymentRoutes(v1, paymentApi)
	// 前端日志接收 - 匿名允许上报（OptionalSessionAuth 仅注入已登录 user_id），对齐 Java permitAll
	router.RegisterClientLogRoutes(v1, clientLogApi)

	// 需要认证保护的路由
	protectedV1 := v1.Group("")
	protectedV1.Use(middleware.AuthMiddleware())
	protectedV1.Use(middleware.UserContextMiddleware())
	router.RegisterAuthRoutes(protectedV1, authApi)
	// 导入导出须先于各模块 CRUD 注册：其静态 `/{module}/template|_export|_import` 必须早于模块自己的
	// `/{module}/:id` 建节点，否则 `/dict/template` 会被 dict 的 `:id` 抢先匹配成 id="template"（404）
	router.RegisterImportExportRoutes(protectedV1, importExportApi)
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
	router.RegisterMessageRoutes(protectedV1, messageApi)
	router.RegisterNotificationSettingRoutes(protectedV1, notificationSettingApi)
	router.RegisterAnnouncementRoutes(protectedV1, announcementApi)
	router.RegisterMessageTemplateRoutes(protectedV1, messageTemplateApi)
	router.RegisterMemberRoutes(protectedV1, memberApi)
	router.RegisterPackageRoutes(protectedV1, packageApi)
	router.RegisterPromotionRoutes(protectedV1, promotionApi)
	router.RegisterOrderRoutes(protectedV1, orderApi)
	router.RegisterFeedbackRoutes(protectedV1, feedbackApi)
	router.RegisterRecommendationRoutes(protectedV1, recommendationApi)
	router.RegisterFavoriteRoutes(protectedV1, favoriteApi)
	router.RegisterPresetRoutes(protectedV1, presetApi)
	router.RegisterCompareRoutes(protectedV1, compareApi)
	router.RegisterAlgorithmSelectRoutes(protectedV1, algorithmSelectApi)
	// AI 域 A 类端点（原生 CRUD/查询，共享 MySQL/Redis，行为对齐 dehaze-python）
	router.RegisterAiModelRoutes(protectedV1, aiModelApi)
	router.RegisterAiProviderRoutes(protectedV1, aiProviderApi)
	router.RegisterAiMcpRoutes(protectedV1, aiMcpApi)
	router.RegisterAiSkillRoutes(protectedV1, aiSkillApi)
	router.RegisterAiKbRoutes(protectedV1, aiKbApi)
	router.RegisterAiA2ARoutes(protectedV1, aiA2AApi)
	router.RegisterAiCompatAuditRoutes(protectedV1, aiCompatApi)
	router.RegisterAiConversationRoutes(protectedV1, aiConversationApi)
	router.RegisterAiMemoryRoutes(protectedV1, aiMemoryApi)
	router.RegisterAiAgentRoutes(protectedV1, aiAgentApi)
	router.RegisterAiEvalRoutes(protectedV1, aiEvalApi)
	router.RegisterAiScheduleRoutes(protectedV1, aiScheduleApi, aiUsageApi)
	router.RegisterAiBillingRoutes(protectedV1, aiBillingApi)
	router.RegisterAiObservabilityRoutes(protectedV1, aiObservabilityApi)
	// AI 域 B 类端点（强依赖 deepagents/LLM/ES）：Go 只做转发，行为唯一实现在 dehaze-python
	router.RegisterAIProxyRoutes(protectedV1, aiProxyApi)
	// 语音域端点（本地 ASR/TTS 引擎仅在 python 进程内）：同样是转发
	router.RegisterVoiceProxyRoutes(protectedV1, aiProxyApi)

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

	// 0) 取消应用级 context 并等待关键异步 goroutine 完成（预测执行、配额落库）
	//    超时 8s，留 2s 给后续资源关闭
	if a.lifecycleMgr != nil {
		if err := a.lifecycleMgr.Shutdown(8 * time.Second); err != nil {
			logger.Warn("生命周期管理器关闭时部分任务未完成", zap.Error(err))
		} else {
			logger.Info("所有异步任务已完成")
		}
	}

	// 0.1) 停止 WebSocket 管理器
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
		{Keys: bson.D{{Key: "user_id", Value: 1}, {Key: "create_time", Value: -1}}},
		{Keys: bson.D{{Key: "create_time", Value: -1}}},
		{Keys: bson.D{{Key: "username", Value: 1}}},
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

		// MongoDB check（必选基础设施）
		func() {
			client := mongo.GetMongoClient()
			if client == nil {
				components["mongodb"] = "DOWN"
				allHealthy = false
				return
			}
			pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			if err := client.Ping(pingCtx, nil); err != nil {
				components["mongodb"] = "DOWN"
				allHealthy = false
				return
			}
			components["mongodb"] = "UP"
		}()

		// MinIO check（仅当默认存储后端为 minio 时检查）
		if cfg.File.Type == storage.StorageMinIO {
			if ms, ok := a.defaultStorage.(*storage.MinioStorageService); ok {
				pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
				err := ms.Ping(pingCtx)
				cancel()
				if err != nil {
					components["minio"] = "DOWN"
					allHealthy = false
				} else {
					components["minio"] = "UP"
				}
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

// trialStatusDepsAdapter 组装层适配器：把券仓储 / AI 积分流水仓储 / 套餐仓储接到
// member.TrialStatusDeps 窄接口，使 member 包不依赖这些模块的结构体（无导入环）。
type trialStatusDepsAdapter struct {
	coupons  *pkgsalerepo.UserCouponRepository
	billing  *airepo.BillingRepository
	packages *pkgsalerepo.PackageRepository
}

func (d trialStatusDepsAdapter) ActiveTrialCouponExpireTime(ctx context.Context, userID int64) (*time.Time, error) {
	return d.coupons.FindActiveTrialCouponExpireTime(ctx, userID)
}

// SumTrialCredits 取 source='trial' 的积分累计（python `sum_amount_by_user_and_source` 的 trial 桶）
func (d trialStatusDepsAdapter) SumTrialCredits(ctx context.Context, userID int64) (int64, error) {
	sums, err := d.billing.SumCreditLogBySource(ctx, userID, nil, nil)
	if err != nil {
		return 0, err
	}
	return sums["trial"], nil
}

// HasPaidOrder 是否存在已支付订单（python `has_paid_order`：status 2/3、未删除）
func (d trialStatusDepsAdapter) HasPaidOrder(ctx context.Context, userID int64) (bool, error) {
	count, err := d.packages.CountPaidOrdersByUser(ctx, userID)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

var _ memberservice.TrialStatusDeps = trialStatusDepsAdapter{}

// auditLogListerAdapter 组装层适配器：Mongo 审计仓储 → member.AuditLogLister，
// 负责把审计模型转成 member 自洽的取数结构（member 包不引审计模块的 bson 标签）。
type auditLogListerAdapter struct {
	repo *auditlogrepo.AuditLogRepository
}

func (a auditLogListerAdapter) ListByTarget(
	ctx context.Context, targetType string, targetID int64, page, pageSize int,
) ([]memberservice.MemberAuditLog, int64, error) {
	items, total, err := a.repo.ListByTarget(ctx, targetType, targetID, page, pageSize)
	if err != nil {
		return nil, 0, err
	}
	result := make([]memberservice.MemberAuditLog, 0, len(items))
	for _, item := range items {
		result = append(result, memberservice.MemberAuditLog{
			ID:          item.ID.Hex(),
			OperatorID:  item.OperatorID,
			Action:      item.Action,
			Module:      item.Module,
			BeforeValue: item.BeforeValue,
			AfterValue:  item.AfterValue,
			IP:          item.IP,
			CreateTime:  item.CreateTime,
		})
	}
	return result, total, nil
}

var _ memberservice.AuditLogLister = auditLogListerAdapter{}

// aiCreditsAdapter 组装层适配器：AI 计费服务 → member.AICreditsProvider
// （余额取 creditsBalance 字符串解析为整数，与 python `int(balance)` 同口径；今日已用取 dailyUsed）
type aiCreditsAdapter struct {
	billing *aiservice.BillingService
}

func (a aiCreditsAdapter) AICredits(ctx context.Context, userID int64) (int64, int64, error) {
	balance, err := a.billing.Balance(ctx, userID)
	if err != nil {
		return 0, 0, err
	}
	// creditsBalance 是**小数形态的字符串**（Decimal 序列化，如 "0.00"），不能用 ParseInt；
	// python 侧 `int(balance)` 是截断取整，故这里 ParseFloat 后向零截断（与 int() 同语义）。
	credits, err := strconv.ParseFloat(balance.CreditsBalance, 64)
	if err != nil {
		return 0, 0, err
	}
	return int64(credits), balance.DailyUsed, nil
}

var _ memberservice.AICreditsProvider = aiCreditsAdapter{}

// packageOverridesAdapter 组装层适配器：套餐仓储 → member.PackageOverridesProvider
// （python `resolve_card_overrides` 读 package.benefit_overrides，key 为 camelCase）
type packageOverridesAdapter struct {
	packages *pkgsalerepo.PackageRepository
}

func (a packageOverridesAdapter) PackageBenefitOverrides(ctx context.Context, levelCode string) (map[string]int, error) {
	pkg, err := a.packages.FindActiveVipByLevelCode(ctx, levelCode)
	if err != nil {
		return nil, err
	}
	if pkg == nil || !pkg.BenefitOverrides.Valid || pkg.BenefitOverrides.String == "" {
		return nil, nil
	}
	overrides := map[string]int{}
	if err := json.Unmarshal([]byte(pkg.BenefitOverrides.String), &overrides); err != nil {
		return nil, err
	}
	return overrides, nil
}

var _ memberservice.PackageOverridesProvider = packageOverridesAdapter{}
