package middleware

import (
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

var traceHeaders = []string{"X-Trace-ID", "traceparent", "sw8"}

// CorsByRules 按照配置处理跨域请求
//
// 生产环境强制使用白名单模式，禁止 "*" + AllowCredentials 组合。
// 白名单为空时拒绝所有跨域请求（不使用危险的默认放行）。
func CorsByRules() gin.HandlerFunc {
	cfg := config.GetConfig()

	// 构建允许的 origins 列表
	allowOrigins := make([]string, 0, len(cfg.Cors.Whitelist))
	for _, w := range cfg.Cors.Whitelist {
		allowOrigins = append(allowOrigins, w.AllowOrigin)
	}

	if len(allowOrigins) == 0 {
		// 白名单为空：拒绝所有跨域请求
		return func(c *gin.Context) {
			c.AbortWithStatus(403)
		}
	}

	// 获取第一个白名单配置作为默认 headers/methods 配置
	first := cfg.Cors.Whitelist[0]
	corsConfig := cors.Config{
		AllowOrigins:     allowOrigins,
		AllowMethods:     splitAndTrim(first.AllowMethods),
		AllowHeaders:     ensureHeaders(splitAndTrim(first.AllowHeaders), traceHeaders),
		ExposeHeaders:    ensureHeaders(splitAndTrim(first.ExposeHeaders), traceHeaders),
		AllowCredentials: first.AllowCredentials,
		MaxAge:           12 * time.Hour,
	}

	// strict-whitelist 模式：严格白名单校验
	if cfg.Cors.Mode == "strict-whitelist" {
		corsConfig.AllowOriginFunc = func(origin string) bool {
			for _, allowed := range allowOrigins {
				if origin == allowed {
					return true
				}
			}
			return false
		}
		// 使用 AllowOriginFunc 时需清空 AllowOrigins
		corsConfig.AllowOrigins = nil
	}

	return cors.New(corsConfig)
}

// splitAndTrim 分割并去除空白
func splitAndTrim(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	result := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

func ensureHeaders(headers []string, extras []string) []string {
	if len(extras) == 0 {
		return headers
	}
	result := make([]string, 0, len(headers)+len(extras))
	seen := make(map[string]struct{}, len(headers)+len(extras))
	for _, h := range headers {
		if h == "" {
			continue
		}
		if _, ok := seen[h]; ok {
			continue
		}
		seen[h] = struct{}{}
		result = append(result, h)
	}
	for _, h := range extras {
		if h == "" {
			continue
		}
		if _, ok := seen[h]; ok {
			continue
		}
		seen[h] = struct{}{}
		result = append(result, h)
	}
	return result
}
