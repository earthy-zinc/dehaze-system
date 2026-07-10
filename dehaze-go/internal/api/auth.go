package api

import (
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	authservice "github.com/earthyzinc/dehaze-go/internal/service/auth"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// AuthApi 认证API处理器
// 职责：负责HTTP请求解析、参数验证和响应组装，不包含业务逻辑
type AuthApi struct {
	authService authservice.IAuthService
}

// NewAuthApi 创建认证API实例
func NewAuthApi(authService authservice.IAuthService) *AuthApi {
	return &AuthApi{
		authService: authService,
	}
}

// Captcha 获取验证码
// @Summary 获取验证码
// @Description 生成图形验证码，返回验证码ID和Base64编码的图片
// @Tags 认证管理
// @Accept json
// @Produce json
// @Success 200 {object} common.Response{data=dto.CaptchaResult}
// @Router /api/v1/auth/captcha [get]
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

// Login 用户登录
// @Summary 用户登录
// @Description 使用用户名密码登录，返回JWT访问令牌
// @Tags 认证管理
// @Accept json
// @Produce json
// @Param request body bo.LoginRequest true "登录请求"
// @Success 200 {object} common.Response{data=dto.LoginResult}
// @Router /api/v1/auth/login [post]
func (a *AuthApi) Login(c *gin.Context) {
	var req bo.LoginRequest
	if err := c.ShouldBind(&req); err != nil {
		_ = c.Error(err)
		return
	}

	clientIP := c.ClientIP()
	result, err := a.authService.Login(c.Request.Context(), &req, clientIP)
	if err != nil {
		_ = c.Error(err)
		return
	}

	// 设置Token到Cookie
	if result != nil {
		security.SetToken(c, result.AccessToken, int(result.Expires/1000)) // 毫秒转秒
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}

// Logout 用户注销
// @Summary 用户注销
// @Description 注销当前用户，将Token加入黑名单
// @Tags 认证管理
// @Accept json
// @Produce json
// @Success 200 {object} common.Response
// @Router /api/v1/auth/logout [post]
func (a *AuthApi) Logout(c *gin.Context) {
	if err := a.authService.Logout(c); err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage(common.SUCCESS.Msg, c)
}

// GetAuthInfo 获取当前用户认证信息
// @Summary 获取当前用户认证信息
// @Description 获取当前登录用户的信息、角色和权限
// @Tags 认证管理
// @Accept json
// @Produce json
// @Success 200 {object} common.Response{data=vo.UserInfoVO}
// @Router /api/v1/auth/me [get]
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

// RefreshToken 刷新令牌
// @Summary 刷新令牌
// @Description 使用当前有效的 Token 获取新的访问令牌，原 Token 的 jti 会被加入黑名单
// @Tags 认证管理
// @Accept json
// @Produce json
// @Success 200 {object} common.Response{data=dto.LoginResult}
// @Router /api/v1/auth/refresh [post]
func (a *AuthApi) RefreshToken(c *gin.Context) {
	token := security.GetToken(c)
	if token == "" {
		_ = c.Error(common.NewBizError(common.TOKEN_INVALID, common.TOKEN_INVALID.Msg))
		return
	}

	result, err := a.authService.RefreshToken(c.Request.Context(), token)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}
