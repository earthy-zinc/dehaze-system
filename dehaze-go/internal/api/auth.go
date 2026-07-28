package api

import (
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	authservice "github.com/earthyzinc/dehaze-go/internal/service/auth"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
)

type AuthApi struct {
	authService authservice.IAuthService
}

func NewAuthApi(authService authservice.IAuthService) *AuthApi {
	return &AuthApi{
		authService: authService,
	}
}

func (a *AuthApi) Captcha(c *gin.Context) {
	clientIP := c.ClientIP()
	result, err := a.authService.GetCaptcha(c.Request.Context(), clientIP)
	if err != nil {
		logger.Error("验证码获取失败", zap.Error(err))
		_ = c.Error(err)
		return
	}

	common.OkWithData(result, c)
}

func (a *AuthApi) Login(c *gin.Context) {
	var req bo.LoginRequest
	if err := c.ShouldBind(&req); err != nil {
		_ = c.Error(err)
		return
	}

	clientIP := c.ClientIP()
	userAgent := c.GetHeader("User-Agent")
	result, err := a.authService.Login(c.Request.Context(), &req, clientIP, userAgent)
	if err != nil {
		_ = c.Error(err)
		return
	}

	if result != nil {
		rememberMe := req.RememberMe != nil && *req.RememberMe
		middleware.SetSessionCookie(c, result.SessionID, rememberMe)
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}

func (a *AuthApi) Register(c *gin.Context) {
	var req bo.RegisterRequest
	if err := c.ShouldBind(&req); err != nil {
		_ = c.Error(err)
		return
	}

	clientIP := c.ClientIP()
	result, err := a.authService.Register(c.Request.Context(), &req, clientIP)
	if err != nil {
		_ = c.Error(err)
		return
	}

	if result != nil {
		middleware.SetSessionCookie(c, result.SessionID, false)
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}

func (a *AuthApi) Logout(c *gin.Context) {
	if err := a.authService.Logout(c); err != nil {
		_ = c.Error(err)
		return
	}

	middleware.ClearSessionCookie(c)
	common.OkWithMessage(common.SUCCESS.Msg, c)
}

func (a *AuthApi) GetAuthInfo(c *gin.Context) {
	userID := security.GetUserID(c)
	if userID == 0 {
		_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "未登录或登录已过期"))
		return
	}

	result, err := a.authService.GetAuthInfo(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}
