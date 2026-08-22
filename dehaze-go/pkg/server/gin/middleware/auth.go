package middleware

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	"go.uber.org/zap"
)

const (
	SessionPrefix         = "session:"
	SessionCookieName     = "X-Session-Id"
	SessionTTL            = 7 * 24 * time.Hour
	SessionRenewThreshold = 24 * time.Hour
)

type SessionData struct {
	UserID      int64    `json:"userId"`
	Username    string   `json:"username"`
	DeptID      int64    `json:"deptId"`
	DataScope   int8     `json:"dataScope"`
	Authorities []string `json:"authorities"`
	Nickname    string   `json:"nickname"`
}

// ApiKeyAuthenticator 是 API Key 认证的校验函数签名，由 app 层注入具体实现。
type ApiKeyAuthenticator func(ctx context.Context, rawKey string) (*security.CustomClaims, error)

// ApiKeyAuth 是外部注入的 API Key 校验实现（默认 nil）。
var ApiKeyAuth ApiKeyAuthenticator

// AuthMiddleware 统一认证中间件：根据请求凭证类型二选一。
// Bearer dhak_* -> API Key 认证；Session Cookie/Header -> Session 认证。
func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.Request.Header.Get("Authorization")
		if authHeader != "" && strings.HasPrefix(authHeader, "Bearer ") {
			token := strings.TrimSpace(authHeader[7:])
			if strings.HasPrefix(token, "dhak_") {
				authWithApiKey(c, token)
				return
			}
		}
		authWithSession(c)
	}
}

func authWithApiKey(c *gin.Context, token string) {
	if ApiKeyAuth == nil {
		unauthorized(c)
		return
	}
	claims, err := ApiKeyAuth(c.Request.Context(), token)
	if err != nil {
		unauthorized(c)
		return
	}
	c.Set("claims", claims)
	c.Request = c.Request.WithContext(trace.WithUserID(c.Request.Context(), claims.UserID))
	c.Next()
}

func authWithSession(c *gin.Context) {
	sessionID := extractSessionID(c)
	if sessionID == "" {
		unauthorized(c)
		return
	}

	cacheClient := cache.GetCache()
	if cacheClient == nil {
		unauthorized(c)
		return
	}

	sessionJSON, err := cacheClient.Get(c.Request.Context(), SessionPrefix+sessionID)
	if err != nil || sessionJSON == "" {
		unauthorized(c)
		return
	}

	var session SessionData
	if err := json.Unmarshal([]byte(sessionJSON), &session); err != nil {
		logger.Error("解析Session数据失败", zap.String("sessionId", sessionID), zap.Error(err))
		unauthorized(c)
		return
	}

	claims := &security.CustomClaims{
		UserID:      session.UserID,
		DeptID:      session.DeptID,
		DataScope:   session.DataScope,
		Authorities: session.Authorities,
	}
	claims.Subject = session.Username
	claims.ID = sessionID
	c.Set("claims", claims)
	c.Request = c.Request.WithContext(trace.WithUserID(c.Request.Context(), session.UserID))

	ttl, err := cacheClient.TTL(c.Request.Context(), SessionPrefix+sessionID)
	if err == nil && ttl > 0 && ttl < SessionRenewThreshold {
		if _, err := cacheClient.Expire(c.Request.Context(), SessionPrefix+sessionID, SessionTTL); err != nil {
			logger.Warn("Session续期失败", zap.String("sessionId", sessionID), zap.Error(err))
		}
	}

	c.Next()
}

// OptionalSessionAuth 可选会话认证中间件。
// session 缺失或无效时放行（匿名），仅当存在合法 session 时解析并注入 user_id。
// 用于允许匿名访问但需要"已登录则注入操作者"的接口（如前端日志接收 POST /api/v1/logs/client）。
func OptionalSessionAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		sessionID := extractSessionID(c)
		if sessionID == "" {
			c.Next()
			return
		}

		cacheClient := cache.GetCache()
		if cacheClient == nil {
			c.Next()
			return
		}

		sessionJSON, err := cacheClient.Get(c.Request.Context(), SessionPrefix+sessionID)
		if err != nil || sessionJSON == "" {
			c.Next()
			return
		}

		var session SessionData
		if err := json.Unmarshal([]byte(sessionJSON), &session); err != nil {
			logger.Error("解析Session数据失败", zap.String("sessionId", sessionID), zap.Error(err))
			c.Next()
			return
		}

		claims := &security.CustomClaims{
			UserID:      session.UserID,
			DeptID:      session.DeptID,
			DataScope:   session.DataScope,
			Authorities: session.Authorities,
		}
		claims.Subject = session.Username
		claims.ID = sessionID
		c.Set("claims", claims)
		c.Request = c.Request.WithContext(trace.WithUserID(c.Request.Context(), session.UserID))

		c.Next()
	}
}

func ExtractSessionID(c *gin.Context) string {
	if cookie, err := c.Cookie(SessionCookieName); err == nil && cookie != "" {
		return cookie
	}
	return c.Request.Header.Get(SessionCookieName)
}

func extractSessionID(c *gin.Context) string {
	return ExtractSessionID(c)
}

func getCookieConfig() (secure bool, path string) {
	secure = true
	path = "/api"
	if config.Config != nil {
		if config.Config.Session.Cookie.Path != "" {
			path = config.Config.Session.Cookie.Path
		}
		secure = config.Config.Session.Cookie.Secure
	}
	return secure, path
}

func SetSessionCookie(c *gin.Context, sessionID string, rememberMe bool) {
	maxAge := -1
	if rememberMe {
		maxAge = int(SessionTTL.Seconds())
	}
	secure, path := getCookieConfig()
	http.SetCookie(c.Writer, &http.Cookie{
		Name:     SessionCookieName,
		Value:    sessionID,
		MaxAge:   maxAge,
		Path:     path,
		HttpOnly: true,
		Secure:   secure,
		SameSite: http.SameSiteLaxMode,
	})
}

func ClearSessionCookie(c *gin.Context) {
	secure, path := getCookieConfig()
	http.SetCookie(c.Writer, &http.Cookie{
		Name:     SessionCookieName,
		Value:    "",
		MaxAge:   0,
		Path:     path,
		HttpOnly: true,
		Secure:   secure,
		SameSite: http.SameSiteLaxMode,
	})
}

func unauthorized(c *gin.Context) {
	c.JSON(http.StatusUnauthorized, common.Response{
		Code:    common.TOKEN_INVALID.Code,
		Data:    map[string]any{},
		Msg:     common.TOKEN_INVALID.Msg,
		TraceId: trace.FromContext(c.Request.Context()),
	})
	c.Abort()
}
