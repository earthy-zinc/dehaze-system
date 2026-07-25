package middleware

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
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

var ApiKeyAuth func(ctx context.Context, rawKey string) (*security.CustomClaims, error)

func SessionAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := c.Request.Header.Get("Authorization")
		if token != "" && strings.HasPrefix(token, "Bearer ") {
			token = token[7:]
		}
		if token != "" && strings.HasPrefix(token, "dhak_") {
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
			c.Next()
			return
		}

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

		ttl, err := cacheClient.TTL(c.Request.Context(), SessionPrefix+sessionID)
		if err == nil && ttl > 0 && ttl < SessionRenewThreshold {
			if _, err := cacheClient.Expire(c.Request.Context(), SessionPrefix+sessionID, SessionTTL); err != nil {
				logger.Warn("Session续期失败", zap.String("sessionId", sessionID), zap.Error(err))
			}
		}

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
