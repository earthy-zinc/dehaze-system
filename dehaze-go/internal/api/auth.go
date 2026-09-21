package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
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

// ListLoginLogs 登录日志分页查询（管理员全量，普通用户仅本人）
// @Summary 登录日志查询
// @Tags 认证接口
// @Produce application/json
// @Param pageNum query int false "页码"
// @Param pageSize query int false "每页条数(≤100)"
// @Param username query string false "按用户名筛选"
// @Param ip query string false "按IP筛选"
// @Param status query int false "登录状态(1:成功;0:失败)"
// @Param deviceType query string false "设备类型"
// @Param startTime query string false "开始时间"
// @Param endTime query string false "结束时间"
// @Success 200 {object} common.Response{data=common.PageResult}
// @Router /api/v1/auth/login-logs [get]
func (a *AuthApi) ListLoginLogs(c *gin.Context) {
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}

	q := query.LoginLogQuery{
		Username:   c.Query("username"),
		IP:         c.Query("ip"),
		DeviceType: c.Query("deviceType"),
		StartTime:  c.Query("startTime"),
		EndTime:    c.Query("endTime"),
		PageNum:    pageNum,
		PageSize:   pageSize,
	}
	if statusStr := c.Query("status"); statusStr != "" {
		status, parseErr := strconv.Atoi(statusStr)
		if parseErr != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "状态参数格式不正确"))
			return
		}
		q.Status = &status
	}

	// 普通用户（非 ROOT/ADMIN 角色）仅可查看本人日志
	if !isAdmin(c) {
		q.UserIDs = []int64{security.GetUserID(c)}
	}

	result, err := a.authService.ListLoginLogs(c.Request.Context(), &q)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}

// ListSessions 在线会话列表（管理员）
// @Summary 在线会话列表
// @Tags 认证接口
// @Produce application/json
// @Param username query string true "用户名（精确匹配）"
// @Success 200 {object} common.Response{data=[]session.SessionInfo}
// @Router /api/v1/auth/sessions [get]
func (a *AuthApi) ListSessions(c *gin.Context) {
	username := c.Query("username")
	if username == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户名不能为空"))
		return
	}

	result, err := a.authService.ListSessions(c.Request.Context(), username)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}

// KickSession 踢出指定在线会话（管理员）
// @Summary 踢出在线会话
// @Tags 认证接口
// @Produce application/json
// @Param sessionId path string true "会话ID"
// @Success 200 {object} common.Response
// @Router /api/v1/auth/sessions/{sessionId} [delete]
func (a *AuthApi) KickSession(c *gin.Context) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "会话ID不能为空"))
		return
	}

	if err := a.authService.KickSession(c.Request.Context(), sessionID); err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("会话已踢出", c)
}

// isAdmin 判断当前登录用户是否具备管理员角色（ROOT/ADMIN），与 Python 端 UserContext.is_admin 对齐
func isAdmin(c *gin.Context) bool {
	claims := security.GetUserInfo(c)
	if claims == nil {
		return false
	}
	for _, authority := range claims.Authorities {
		if authority == "ROLE_ROOT" || authority == "ROLE_ADMIN" {
			return true
		}
	}
	return false
}
