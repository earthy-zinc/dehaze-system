package middleware

import (
	"context"
	"strings"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	global_error "github.com/earthyzinc/dehaze-go/pkg/error"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// CaptchaConfig 验证码中间件配置
type CaptchaConfig struct {
	// EnabledPaths 需要验证码校验的路径列表（支持部分匹配）
	// 例如：["/login", "/register", "/forgot-password"]
	EnabledPaths []string

	// KeyPrefix Redis 键前缀，默认为 common.CaptchaCodePrefix
	KeyPrefix string

	// SkipOnRedisDown Redis 不可用时是否跳过验证码校验
	// 默认为 true，保证服务可用性
	SkipOnRedisDown bool

	// Message 验证码错误提示消息
	Message string
}

// Captcha 验证码校验中间件
// 修复说明 (P1 问题)：
// - 将硬编码路径匹配改为配置化 EnabledPaths
// - 支持多个路径配置，提高灵活性
// - 添加 SkipOnRedisDown 配置选项
// - 添加配置验证和默认值
//
// 使用示例:
//
//	config := CaptchaConfig{
//	    EnabledPaths: []string{"/login", "/register"},
//	    KeyPrefix: common.CaptchaCodePrefix,
//	    SkipOnRedisDown: true,
//	}
//	router.Use(Captcha(config))
func Captcha(config CaptchaConfig) gin.HandlerFunc {
	// 设置默认值
	if config.KeyPrefix == "" {
		config.KeyPrefix = common.CaptchaCodePrefix
	}
	if config.Message == "" {
		config.Message = "验证码校验失败"
	}

	return func(c *gin.Context) {
		// 检查当前路径是否在启用列表中
		path := c.Request.URL.Path
		shouldCheck := false
		for _, enabledPath := range config.EnabledPaths {
			if strings.Contains(path, enabledPath) {
				shouldCheck = true
				break
			}
		}

		// 如果路径不在启用列表中，跳过验证码校验
		if !shouldCheck {
			c.Next()
			return
		}

		// 从请求参数中获取验证码和验证码ID
		captchaCode := c.PostForm("captchaCode")
		captchaKey := c.PostForm("captchaKey")

		// 参数校验
		if captchaCode == "" || captchaKey == "" {
			logger.Warn("验证码参数缺失",
				zap.String("path", path),
				zap.String("ip", c.ClientIP()))
			common.FailWithCode(common.PARAM_IS_NULL, c)
			c.Abort()
			return
		}

		cacheClient := cache.GetCache()
		if cacheClient == nil {
			logger.Error("验证码缓存不可用")
			if config.SkipOnRedisDown {
				c.Next()
				return
			}
			common.FailWithCode(common.CACHE_SERVICE_ERROR, c)
			c.Abort()
			return
		}

		ctx := context.Background()
		captchaKey = config.KeyPrefix + captchaKey

		// 校验验证码
		storedCode, err := cacheClient.Get(ctx, captchaKey)

		if err != nil {
			// 验证码已过期或不存在``
			if err == global_error.ErrKeyNotFound {
				logger.Warn("验证码已过期或不存在",
					zap.String("captchaKey", captchaKey),
					zap.String("ip", c.ClientIP()))
				common.FailWithCode(common.VERIFY_CODE_TIMEOUT, c)
				c.Abort()
				return
			}
			// 其他错误
			logger.Error("验证码校验失败",
				zap.String("captchaKey", captchaKey),
				zap.Error(err))
			common.FailWithCode(common.CACHE_SERVICE_ERROR, c)
			c.Abort()
			return
		}

		// 验证码比对（不区分大小写）
		if !strings.EqualFold(storedCode, captchaCode) {
			logger.Warn("验证码错误",
				zap.String("captchaKey", captchaKey),
				zap.String("ip", c.ClientIP()))
			common.FailWithCode(common.VERIFY_CODE_ERROR, c)
			c.Abort()
			return
		}

		// 验证成功后删除已使用的验证码，防止重复使用
		if err := cacheClient.Delete(ctx, captchaKey); err != nil {
			logger.Error("删除验证码失败",
				zap.String("captchaKey", captchaKey),
				zap.Error(err))
			// 删除失败不中断流程，仅记录日志
		}

		logger.Info("验证码校验成功",
			zap.String("captchaKey", captchaKey),
			zap.String("ip", c.ClientIP()))

		c.Next()
	}
}
