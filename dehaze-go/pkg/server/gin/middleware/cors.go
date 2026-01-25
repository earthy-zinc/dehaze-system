package middleware

import (
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Cors 放行所有跨域请求
func Cors() gin.HandlerFunc {
	return cors.New(cors.Config{
		AllowAllOrigins:  true,
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Content-Type", "AccessToken", "X-CSRF-Token", "Authorization", "Token", "X-Token", "X-User-Id"},
		ExposeHeaders:    []string{"Content-Length", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Content-Type", "New-Token", "New-Expires-At"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	})
}

// CorsByRules 按照配置处理跨域请求
func CorsByRules() gin.HandlerFunc {
	cfg := config.GetConfig()

	// allow-all 模式：放行所有跨域请求
	if cfg.Cors.Mode == "allow-all" {
		return Cors()
	}

	// 构建允许的 origins 列表
	allowOrigins := make([]string, 0, len(cfg.Cors.Whitelist))
	for _, w := range cfg.Cors.Whitelist {
		allowOrigins = append(allowOrigins, w.AllowOrigin)
	}

	// 获取第一个白名单配置作为默认 headers/methods 配置
	var corsConfig cors.Config
	if len(cfg.Cors.Whitelist) > 0 {
		first := cfg.Cors.Whitelist[0]
		corsConfig = cors.Config{
			AllowOrigins:     allowOrigins,
			AllowMethods:     splitAndTrim(first.AllowMethods),
			AllowHeaders:     splitAndTrim(first.AllowHeaders),
			ExposeHeaders:    splitAndTrim(first.ExposeHeaders),
			AllowCredentials: first.AllowCredentials,
			MaxAge:           12 * time.Hour,
		}
	} else {
		// 没有配置白名单时的默认配置
		corsConfig = cors.Config{
			AllowOrigins:     []string{},
			AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
			AllowHeaders:     []string{"Content-Type", "Authorization"},
			AllowCredentials: true,
			MaxAge:           12 * time.Hour,
		}
	}

	// strict-whitelist 模式：严格白名单校验
	if cfg.Cors.Mode == "strict-whitelist" {
		corsConfig.AllowOriginFunc = func(origin string) bool {
			// 健康检查接口始终放行
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
