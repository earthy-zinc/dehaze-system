package middleware

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
)

const (
	// 默认防重复提交 key 前缀
	defaultAntiRepeatPrefix = "anti_repeat:"
)

// AntiRepeatConfig 防重复提交配置
type AntiRepeatConfig struct {
	// Expire 过期时间，单位：秒，默认 5 秒
	Expire int
	// KeyGenerator 自定义 key 生成函数，如果为 nil 则使用默认策略
	KeyGenerator func(c *gin.Context) string
	// IncludeBody 是否包含请求体（POST/PUT/DELETE 等需要请求体的请求），默认 false
	IncludeBody bool
	// SkipPaths 需要跳过的路径列表（不进行防重复检查）
	SkipPaths []string
	// OnCustomError 自定义错误处理函数，如果为 nil 则使用默认错误处理
	OnCustomError func(c *gin.Context)
}

// AntiRepeat 创建防重复提交中间件
// 使用 Redis SETNX（SET if Not eXists）实现分布式锁
// 默认基于 JWT 的 jti + 请求方法 + URI 生成唯一 key
func AntiRepeat(config AntiRepeatConfig) gin.HandlerFunc {
	// 设置默认值
	if config.Expire == 0 {
		config.Expire = 5 // 默认 5 秒
	}

	// 如果没有提供自定义 key 生成器，使用默认策略
	if config.KeyGenerator == nil {
		config.KeyGenerator = defaultKeyGenerator
	}

	// 将 SkipPaths 转换为 map 提高查找效率
	skipPathsMap := make(map[string]bool)
	for _, path := range config.SkipPaths {
		skipPathsMap[path] = true
	}

	return func(c *gin.Context) {
		// 检查是否在跳过列表中
		if skipPathsMap[c.Request.URL.Path] {
			c.Next()
			return
		}

		// 获取缓存管理器
		cacheManager := cache.GetCacheManager()
		if cacheManager == nil {
			logger.Error("缓存管理器未初始化")
			c.Next()
			return
		}

		commonCache := cacheManager.GetCache()
		if commonCache == nil {
			logger.Warn("缓存不可用，防重复提交功能已降级放行",
				zap.String("path", c.Request.URL.Path),
				zap.String("method", c.Request.Method),
			)
			c.Next()
			return
		}

		// 生成唯一 key
		key := defaultAntiRepeatPrefix + config.KeyGenerator(c)

		// 尝试获取锁（使用缓存 SetNX）
		ctx := context.Background()
		expire := time.Duration(config.Expire) * time.Second

		// 尝试设置 key（如果 key 不存在则设置成功）
		setResult, err := commonCache.SetNX(ctx, key, "1", expire)
		if err != nil {
			logger.Error("防重复提交 Redis 操作失败",
				zap.String("key", key),
				zap.Error(err),
			)
			// Redis 操作失败时，记录日志并放行请求（容错处理）
			c.Next()
			return
		}

		// 如果 setResult 为 false，说明 key 已存在，是重复请求
		if !setResult {
			logger.Warn("检测到重复提交请求",
				zap.String("key", key),
				zap.String("path", c.Request.URL.Path),
				zap.String("method", c.Request.Method),
				zap.String("client_ip", c.ClientIP()),
			)

			// 使用自定义错误处理或默认错误处理
			if config.OnCustomError != nil {
				config.OnCustomError(c)
			} else {
				_ = c.Error(common.NewBizError(common.REPEAT_SUBMIT_ERROR, common.REPEAT_SUBMIT_ERROR.Msg))
			}

			c.Abort()
			return
		}

		// 正常请求，继续处理
		c.Next()

		// 注意：这里不会删除 Redis key，让它自然过期
		// 这样可以在过期时间内防止重复提交
	}
}

// defaultKeyGenerator 默认的 key 生成策略
// 基于 JWT 的 jti + 请求方法 + URI 生成唯一 key
// 如果 IncludeBody 为 true，还会包含请求体内容的 MD5
func defaultKeyGenerator(c *gin.Context) string {
	var keyParts []string

	// 1. 获取 JWT 的 jti（如果存在）
	if claims, exists := c.Get("claims"); exists {
		if customClaims, ok := claims.(*security.CustomClaims); ok && customClaims.ID != "" {
			keyParts = append(keyParts, customClaims.ID)
		}
	}

	// 2. 如果没有 JWT jti，使用客户端 IP
	if len(keyParts) == 0 {
		keyParts = append(keyParts, c.ClientIP())
	}

	// 3. 添加请求方法
	keyParts = append(keyParts, c.Request.Method)

	// 4. 添加请求路径
	keyParts = append(keyParts, c.Request.URL.Path)

	// 5. 如果需要包含请求体（POST/PUT/PATCH/DELETE）
	body := ""
	if c.Request.Method != http.MethodGet && c.Request.Method != http.MethodHead {
		// 读取请求体
		bodyBytes, err := io.ReadAll(c.Request.Body)
		if err == nil && len(bodyBytes) > 0 {
			body = string(bodyBytes)
			// 恢复请求体（因为 io.ReadAll 会消耗 Body）
			c.Request.Body = io.NopCloser(strings.NewReader(body))

			// 对请求体进行 MD5 哈希（避免 key 过长）
			if body != "" {
				hash := md5.Sum([]byte(body))
				bodyHash := hex.EncodeToString(hash[:])
				keyParts = append(keyParts, bodyHash)
			}
		}
	}

	// 使用冒号连接各部分
	return strings.Join(keyParts, ":")
}

// DefaultAntiRepeat 创建使用默认配置的防重复提交中间件
// 默认配置：
// - 过期时间：5 秒
// - 不包含请求体
// - 跳过路径：空
// - 错误处理：使用默认错误响应
func DefaultAntiRepeat() gin.HandlerFunc {
	return AntiRepeat(AntiRepeatConfig{
		Expire:      5,
		IncludeBody: false,
	})
}

// CustomKeyGenerator 自定义 key 生成器的示例
// 可以根据业务需求实现不同的 key 生成策略
func CustomKeyGenerator(includeBody bool) func(c *gin.Context) string {
	return func(c *gin.Context) string {
		var keyParts []string

		// 获取用户 ID（从 JWT claims）
		if claims, exists := c.Get("claims"); exists {
			if customClaims, ok := claims.(*security.CustomClaims); ok {
				keyParts = append(keyParts, fmt.Sprintf("user:%d", customClaims.UserID))
			}
		}

		// 添加请求方法
		keyParts = append(keyParts, c.Request.Method)

		// 添加请求路径
		keyParts = append(keyParts, c.Request.URL.Path)

		// 可选：包含请求体
		if includeBody && c.Request.Method != http.MethodGet && c.Request.Method != http.MethodHead {
			bodyBytes, _ := io.ReadAll(c.Request.Body)
			if len(bodyBytes) > 0 {
				// 恢复请求体
				c.Request.Body = io.NopCloser(strings.NewReader(string(bodyBytes)))

				// 对请求体进行 MD5 哈希
				hash := md5.Sum(bodyBytes)
				bodyHash := hex.EncodeToString(hash[:])
				keyParts = append(keyParts, bodyHash)
			}
		}

		return strings.Join(keyParts, ":")
	}
}

// CustomAntiRepeat 创建自定义配置的防重复提交中间件
//
// 示例 1：自定义过期时间（10 秒）
//
//	router.POST("/create", middleware.AntiRepeat(middleware.AntiRepeatConfig{
//	    Expire: 10,
//	}), handler.Create)
//
// 示例 2：包含请求体
//
//	router.POST("/update", middleware.AntiRepeat(middleware.AntiRepeatConfig{
//	    Expire:      5,
//	    IncludeBody: true,
//	}), handler.Update)
//
// 示例 3：跳过特定路径
//
//	router.Use(middleware.AntiRepeat(middleware.AntiRepeatConfig{
//	    SkipPaths: []string{"/api/public/health", "/api/public/ping"},
//	}))
//
// 示例 4：自定义 key 生成函数
//
//	router.POST("/submit", middleware.AntiRepeat(middleware.AntiRepeatConfig{
//	    KeyGenerator: middleware.CustomKeyGenerator(true),
//	}), handler.Submit)
//
// 示例 5：自定义错误处理
//
//	router.POST("/api/order/create", middleware.AntiRepeat(middleware.AntiRepeatConfig{
//	    Expire: 3,
//	    OnCustomError: func(c *gin.Context) {
//	        c.JSON(http.StatusOK, gin.H{
//	            "code": "A0002",
//	            "msg":  "订单正在处理中，请勿重复提交",
//	            "data": nil,
//	        })
//	    },
//	}), orderHandler.CreateOrder)
//
// 示例 6：组合配置
//
//	router.POST("/payment", middleware.AntiRepeat(middleware.AntiRepeatConfig{
//	    Expire:      10,
//	    IncludeBody: true,
//	    SkipPaths:   []string{"/api/payment/callback"},
//	    KeyGenerator: func(c *gin.Context) string {
//	        // 自定义基于订单 ID 的 key 生成逻辑
//	        orderID := c.PostForm("order_id")
//	        return fmt.Sprintf("payment:%s", orderID)
//	    },
//	    OnCustomError: func(c *gin.Context) {
//	        c.JSON(http.StatusOK, gin.H{
//	            "code": "A0002",
//	            "msg":  "支付请求处理中，请稍候...",
//	            "data": nil,
//	        })
//	    },
//	}), paymentHandler.ProcessPayment)
func CustomAntiRepeat(config AntiRepeatConfig) gin.HandlerFunc {
	return AntiRepeat(config)
}

// SetAntiRepeatLock 手动设置防重复提交锁（用于在业务逻辑中调用）
// 这可以用于某些需要在处理完成后立即解锁的场景
//
// 参数：
//   - key: 锁的 key
//   - expire: 过期时间（秒）
//
// 返回：
//   - bool: 是否成功设置锁（true 表示是第一次请求，false 表示锁已存在）
//   - error: 缓存操作错误
func SetAntiRepeatLock(key string, expire int) (bool, error) {
	cacheClient := cache.GetCache()
	if cacheClient == nil {
		return false, fmt.Errorf("缓存不可用")
	}

	ctx := context.Background()
	expireDuration := time.Duration(expire) * time.Second

	result, err := cacheClient.SetNX(ctx, defaultAntiRepeatPrefix+key, "1", expireDuration)
	if err != nil {
		return false, err
	}

	return result, nil
}

// DeleteAntiRepeatLock 手动删除防重复提交锁（谨慎使用）
// 注意：一般情况下不应该删除锁，让其自然过期即可
// 只有在特殊场景下（如业务处理失败，需要允许立即重试）才使用此函数
//
// 参数：
//   - key: 锁的 key
//
// 返回：
//   - error: 缓存操作错误
func DeleteAntiRepeatLock(key string) error {
	cache := cache.GetCache()
	if cache == nil {
		return fmt.Errorf("缓存不可用")
	}

	ctx := context.Background()
	return cache.Delete(ctx, defaultAntiRepeatPrefix+key)
}
