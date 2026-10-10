package middleware

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
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
	// defaultAntiRepeatPrefix 防重复提交 key 前缀。三端共享同一 Redis，**键格式即跨端契约**
	// （见 dehaze-doc/docs/05-改造计划/防重复提交三端统一改造计划.md）。
	defaultAntiRepeatPrefix = "anti_repeat:"
	// defaultAntiRepeatExpire 统一过期窗口（秒）。三端同一口径，禁止逐路由覆盖。
	defaultAntiRepeatExpire = 5
	// maxHashedBodySize 参与摘要的请求体上限；超过则不读，避免大请求体常驻内存。
	maxHashedBodySize = 256 * 1024
)

// AntiRepeatConfig 防重复提交配置
type AntiRepeatConfig struct {
	// Expire 过期时间，单位：秒，默认 5 秒（统一口径，业务路由不应覆盖）
	Expire int
	// KeyGenerator 自定义 key 生成函数，如果为 nil 则使用默认策略
	KeyGenerator func(c *gin.Context) string
	// SkipPaths 需要跳过的路径列表（不进行防重复检查）
	SkipPaths []string
	// OnCustomError 自定义错误处理函数，如果为 nil 则使用默认错误处理
	OnCustomError func(c *gin.Context)
}

// AntiRepeat 创建防重复提交中间件（Redis SETNX）。
//
// 键格式（三端必须完全一致）：
//
//	anti_repeat:{identity}:{METHOD}:{path}:{sortedQuery}:{bodyHash}
//
// 五段固定（query / bodyHash 为空时保留空段），保证同一请求在三端算出同一键。
// 参与防重的端点由路由显式挂载决定（白名单制），须与 python 中间件白名单、
// java @PreventDuplicateSubmit 完全一致（由跨端契约测试对拍）。
func AntiRepeat(config AntiRepeatConfig) gin.HandlerFunc {
	// 设置默认值
	if config.Expire == 0 {
		config.Expire = defaultAntiRepeatExpire
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
		ctx := c.Request.Context()
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

// defaultKeyGenerator 默认的 key 生成策略：
// 身份 + 请求方法 + 请求路径 + 排序归一后的 query + 请求体摘要（五段固定）。
func defaultKeyGenerator(c *gin.Context) string {
	return resolveIdentity(c) + ":" +
		c.Request.Method + ":" +
		c.Request.URL.Path + ":" +
		normalizeQuery(c.Request.URL.RawQuery) + ":" +
		resolveBodyHash(c)
}

// resolveIdentity 身份统一收敛到用户维度；无有效登录态时退化为客户端 IP。
// JWT 的 jti 随刷新变化，会让"重新登录后提交同一请求"各自算出新键而绕过防重。
func resolveIdentity(c *gin.Context) string {
	if claims, exists := c.Get("claims"); exists {
		if customClaims, ok := claims.(*security.CustomClaims); ok && customClaims.UserID != 0 {
			return fmt.Sprintf("user:%d", customClaims.UserID)
		}
	}
	return "ip:" + c.ClientIP()
}

// normalizeQuery 将 query 按 (key, value) 全排序后重新编码。
// 不排序等于留一条绕过路径：?a=1&b=2 与 ?b=2&a=1 会算出不同键。
func normalizeQuery(rawQuery string) string {
	if rawQuery == "" {
		return ""
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		// 非法 query：整体按空处理，避免解析歧义导致键不稳定
		return ""
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		items := append([]string(nil), values[key]...)
		sort.Strings(items)
		for _, item := range items {
			parts = append(parts, url.QueryEscape(key)+"="+url.QueryEscape(item))
		}
	}
	return strings.Join(parts, "&")
}

// resolveBodyHash 计算请求体摘要；multipart 与超大请求体按空摘要参与（与 python/java 口径一致）。
func resolveBodyHash(c *gin.Context) string {
	if strings.HasPrefix(strings.ToLower(c.GetHeader("Content-Type")), "multipart/") {
		return ""
	}
	if c.Request.Method == http.MethodGet || c.Request.Method == http.MethodHead || c.Request.Body == nil {
		return ""
	}
	if c.Request.ContentLength > maxHashedBodySize {
		return ""
	}
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil || len(bodyBytes) == 0 {
		return ""
	}
	// 恢复请求体（io.ReadAll 会消耗 Body），供后续 handler 参数绑定
	c.Request.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	sum := md5.Sum(bodyBytes)
	return hex.EncodeToString(sum[:])
}

// DefaultAntiRepeat 使用统一默认配置（过期 5 秒）创建防重复提交中间件。
// 业务路由一律使用本函数挂载，避免逐路由覆盖 Expire 造成三端口径漂移。
func DefaultAntiRepeat() gin.HandlerFunc {
	return AntiRepeat(AntiRepeatConfig{})
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
func SetAntiRepeatLock(ctx context.Context, key string, expire int) (bool, error) {
	cacheClient := cache.GetCache()
	if cacheClient == nil {
		return false, fmt.Errorf("缓存不可用")
	}

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
func DeleteAntiRepeatLock(ctx context.Context, key string) error {
	cacheClient := cache.GetCache()
	if cacheClient == nil {
		return fmt.Errorf("缓存不可用")
	}

	return cacheClient.Delete(ctx, defaultAntiRepeatPrefix+key)
}
