package auth

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/session"
	"github.com/gin-gonic/gin"
)

type IAuthService interface {
	Login(ctx context.Context, req *bo.LoginRequest, clientIP, userAgent string) (*dto.LoginResult, error)
	Register(ctx context.Context, req *bo.RegisterRequest, clientIP string) (*dto.LoginResult, error)
	Logout(c *gin.Context) error
	GetCaptcha(ctx context.Context, clientIP string) (*dto.CaptchaResult, error)
	// VerifyCaptchaStatus 校验验证码并消费（GETDEL 语义）。
	// 返回 (是否通过, Key 是否不存在/已过期)，用于区分 A0214（验证码错误）与 A0213（验证码已过期）。
	VerifyCaptchaStatus(ctx context.Context, captchaKey, captchaCode string) (ok bool, expired bool)
	GetAuthInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error)
	// ListLoginLogs 登录日志分页查询（q.UserIDs 非空时限定可见用户范围：普通用户仅本人）
	ListLoginLogs(ctx context.Context, q *query.LoginLogQuery) (*vo.PageResult[vo.LoginLogVO], error)
	// ListSessions 在线会话列表（按用户名精确过滤）
	ListSessions(ctx context.Context, username string) ([]session.SessionInfo, error)
	// KickSession 踢出指定在线会话
	KickSession(ctx context.Context, sessionID string) error
}
