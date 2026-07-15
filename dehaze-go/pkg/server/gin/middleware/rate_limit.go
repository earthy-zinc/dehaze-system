package middleware

import (
	"fmt"
	"net/http"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"github.com/ulule/limiter/v3"
	mgin "github.com/ulule/limiter/v3/drivers/middleware/gin"
	sredis "github.com/ulule/limiter/v3/drivers/store/redis"
	"go.uber.org/zap"
)

// LimitType 限流类型
type LimitType string

const (
	LimitTypeIP     LimitType = "ip"     // IP限流
	LimitTypeUser   LimitType = "user"   // 用户限流
	LimitTypeGlobal LimitType = "global" // 全局限流
)

// RateLimiterConfig 限流配置
type RateLimiterConfig struct {
	Type         LimitType                   // 限流类型: ip/user/global
	Period       time.Duration               // 时间窗口
	Limit        int64                       // 最大请求数
	KeyGenerator func(c *gin.Context) string // 自定义key生成函数
	FallbackMode bool                        // 降级模式: true=Redis异常时放行, false=Redis异常时拒绝
	Message      string                      // 自定义限流提示消息
}

// createStore 创建限流存储后端（Redis）
func createStore() (limiter.Store, error) {
	client := redis.GetClient()
	if client == nil {
		return nil, fmt.Errorf("redis client not available")
	}

	return sredis.NewStoreWithOptions(client, limiter.StoreOptions{
		Prefix:   "rate:limit:",
		MaxRetry: 3,
	})
}

// createLimiterInstance 创建限流器实例
func createLimiterInstance(rate limiter.Rate, store limiter.Store) *limiter.Limiter {
	return limiter.New(store, rate)
}

// RateLimiterMiddleware 限流中间件
func RateLimiterMiddleware(cfg RateLimiterConfig) gin.HandlerFunc {
	// 设置默认值
	if cfg.Period == 0 {
		cfg.Period = time.Minute
	}
	if cfg.Limit <= 0 {
		cfg.Limit = 100
	}
	if cfg.Message == "" {
		cfg.Message = "请求太过频繁，请稍后再试"
	}

	// 创建 rate
	rate := limiter.Rate{
		Period: cfg.Period,
		Limit:  cfg.Limit,
	}

	// 创建 store
	store, err := createStore()
	if err != nil {
		logger.Warn("rate limiter: failed to create redis store", zap.Error(err))
		// Redis不可用，返回降级处理
		return func(c *gin.Context) {
			if cfg.FallbackMode {
				logger.Warn("rate limiter: redis unavailable, fallback to allow")
				c.Next()
				return
			}
			c.JSON(http.StatusOK, gin.H{
				"code": common.RATE_LIMIT.GetCode(),
				"msg":  "限流服务不可用",
			})
			c.Abort()
		}
	}

	// 创建 limiter 实例
	instance := createLimiterInstance(rate, store)

	// 创建自定义key提取函数
	keyGetter := func(c *gin.Context) string {
		if cfg.KeyGenerator != nil {
			return cfg.KeyGenerator(c)
		}
		return c.ClientIP()
	}

	// 创建中间件
	middleware := mgin.NewMiddleware(instance,
		mgin.WithKeyGetter(keyGetter),
		mgin.WithLimitReachedHandler(func(c *gin.Context) {
			// 获取限流上下文以计算剩余时间
			ctx, err := instance.Get(c.Request.Context(), keyGetter(c))
			if err == nil && ctx.Reset > 0 {
				remaining := time.Until(time.Unix(ctx.Reset, 0))
				seconds := int(remaining.Seconds()) + 1
				if seconds < 1 {
					seconds = 1
				}
				c.JSON(http.StatusOK, gin.H{
					"code": common.RATE_LIMIT.GetCode(),
					"msg":  fmt.Sprintf("%s，请 %d 秒后重试", cfg.Message, seconds),
				})
			} else {
				c.JSON(http.StatusOK, gin.H{
					"code": common.RATE_LIMIT.GetCode(),
					"msg":  cfg.Message,
				})
			}
			c.Abort()
		}),
		mgin.WithErrorHandler(func(c *gin.Context, err error) {
			logger.Error("rate limiter error", zap.Error(err))
			if cfg.FallbackMode {
				logger.Warn("rate limiter: redis error, fallback to allow", zap.Error(err))
				c.Next()
				return
			}
			c.JSON(http.StatusOK, gin.H{
				"code": common.RATE_LIMIT.GetCode(),
				"msg":  "限流服务异常",
			})
			c.Abort()
		}),
	)

	return middleware
}

// === Key 生成器 ===

// GenerateIPKey 生成IP限流key
func GenerateIPKey(prefix string) func(c *gin.Context) string {
	return func(c *gin.Context) string {
		return fmt.Sprintf("ip:%s:%s", prefix, c.ClientIP())
	}
}

// GenerateUserKey 生成用户限流key
// 通过 security.GetUserID 从 JWT claims 读取用户ID，缺失时退化为 IP 限流
func GenerateUserKey(prefix string) func(c *gin.Context) string {
	return func(c *gin.Context) string {
		userID := security.GetUserID(c)
		if userID == 0 {
			return fmt.Sprintf("user:%s:%s", prefix, c.ClientIP())
		}
		return fmt.Sprintf("user:%s:%d", prefix, userID)
	}
}

// GenerateGlobalKey 生成全局限流key
func GenerateGlobalKey(prefix string) func(c *gin.Context) string {
	return func(c *gin.Context) string {
		return fmt.Sprintf("global:%s", prefix)
	}
}

// === 预定义限流器 ===

// IPRateLimiter IP限流器（使用全局配置）
func IPRateLimiter() gin.HandlerFunc {
	cfg := config.GetConfig()
	return RateLimiterMiddleware(RateLimiterConfig{
		Type:         LimitTypeIP,
		Period:       time.Duration(cfg.System.LimitTimeIP) * time.Second,
		Limit:        int64(cfg.System.LimitCountIP),
		KeyGenerator: GenerateIPKey("default"),
		FallbackMode: true,
		Message:      "IP请求太过频繁，请稍后再试",
	})
}

// CustomIPRateLimiter 自定义IP限流器
func CustomIPRateLimiter(periodSeconds, maxRequests int, prefix string) gin.HandlerFunc {
	return RateLimiterMiddleware(RateLimiterConfig{
		Type:         LimitTypeIP,
		Period:       time.Duration(periodSeconds) * time.Second,
		Limit:        int64(maxRequests),
		KeyGenerator: GenerateIPKey(prefix),
		FallbackMode: true,
		Message:      "IP请求太过频繁，请稍后再试",
	})
}

// UserRateLimiter 用户限流器
func UserRateLimiter(periodSeconds, maxRequests int, prefix string) gin.HandlerFunc {
	return RateLimiterMiddleware(RateLimiterConfig{
		Type:         LimitTypeUser,
		Period:       time.Duration(periodSeconds) * time.Second,
		Limit:        int64(maxRequests),
		KeyGenerator: GenerateUserKey(prefix),
		FallbackMode: true,
		Message:      "用户请求太过频繁，请稍后再试",
	})
}

// GlobalRateLimiter 全局限流器
func GlobalRateLimiter(periodSeconds, maxRequests int, prefix string) gin.HandlerFunc {
	return RateLimiterMiddleware(RateLimiterConfig{
		Type:         LimitTypeGlobal,
		Period:       time.Duration(periodSeconds) * time.Second,
		Limit:        int64(maxRequests),
		KeyGenerator: GenerateGlobalKey(prefix),
		FallbackMode: true,
		Message:      "系统繁忙，请稍后再试",
	})
}

// PathRateLimiter 路径级别限流器（可针对特定API设置不同限流规则）
func PathRateLimiter(periodSeconds, maxRequests int) gin.HandlerFunc {
	return RateLimiterMiddleware(RateLimiterConfig{
		Type:   LimitTypeIP,
		Period: time.Duration(periodSeconds) * time.Second,
		Limit:  int64(maxRequests),
		KeyGenerator: func(c *gin.Context) string {
			return fmt.Sprintf("path:%s:%s", c.Request.URL.Path, c.ClientIP())
		},
		FallbackMode: true,
		Message:      "该接口请求太过频繁，请稍后再试",
	})
}
