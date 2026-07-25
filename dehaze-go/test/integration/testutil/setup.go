package testutil

import (
	"context"
	"os"

	"github.com/earthyzinc/dehaze-go/internal/api"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	"github.com/earthyzinc/dehaze-go/internal/router"
	authservice "github.com/earthyzinc/dehaze-go/internal/service/auth"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"

	"github.com/gin-gonic/gin"
)

// Engine 全局测试引擎，由 InitTestEnv 初始化后供各测试文件使用
var Engine *gin.Engine

// InitTestEnv 初始化集成测试环境（配置、数据库、缓存、日志）。
// 各模块的 TestMain 调用一次即可，避免重复初始化。
// projectRootRelPath 为从测试文件到项目根的相对路径，通常为 "../../../"。
func InitTestEnv(projectRootRelPath string) {
	if err := os.Chdir(projectRootRelPath); err != nil {
		panic("切换工作目录到项目根失败: " + err.Error())
	}

	gin.SetMode(gin.TestMode)

	logger.InitDefaultLogger()
	if _, err := config.Init(); err != nil {
		panic("加载测试配置失败: " + err.Error())
	}
	if err := logger.Init(); err != nil {
		panic("初始化日志失败: " + err.Error())
	}
	if err := database.Init(database.GetDatabaseConfig()); err != nil {
		panic("初始化数据库失败: " + err.Error())
	}
	if _, err := cache.Init(); err != nil {
		panic("初始化缓存失败: " + err.Error())
	}
}

// SetupAuthRouter 构建与生产一致的认证模块路由 + 中间件链路。
// 返回的 engine 可直接赋值给 Engine 供 HTTP 辅助函数使用。
func SetupAuthRouter() *gin.Engine {
	engine := gin.New()
	engine.Use(middleware.ContextErrorHandler())

	gormDB := database.DB()
	cacheClient := cache.GetCache()

	userRepo := userrepo.NewUserRepository(gormDB)
	roleRepo := rolerepo.NewRoleRepository(gormDB)
	deptRepo := deptrepo.NewDeptRepository(gormDB)
	menuRepo := menurepo.NewMenuRepository(gormDB)

	userService := userservice.NewUserService(userRepo, roleRepo, deptRepo, menuRepo)
	authService := authservice.NewAuthService(cacheClient, userService)
	authApi := api.NewAuthApi(authService)

	v1 := engine.Group("/api/v1")
	router.RegisterNoAuthRoutes(v1, authApi)

	protectedV1 := v1.Group("")
	protectedV1.Use(middleware.SessionAuth())
	protectedV1.Use(middleware.UserContextMiddleware())
	router.RegisterAuthRoutes(protectedV1, authApi)

	return engine
}

// CleanLoginFailCounts 清理指定用户列表的登录失败计数缓存，避免测试间污染
func CleanLoginFailCounts(usernames ...string) {
	ctx := context.Background()
	cacheClient := cache.GetCache()
	// httptest 默认 RemoteAddr 为 "192.0.2.1:1234"，对应 IP 为 192.0.2.1
	_ = cacheClient.Delete(ctx, "login:fail:ip:192.0.2.1")
	for _, u := range usernames {
		_ = cacheClient.Delete(ctx, "login:fail:user:"+u)
	}
}
