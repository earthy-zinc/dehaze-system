package auth

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/gin-gonic/gin"
)

// IAuthService 认证服务接口
type IAuthService interface {
	Login(ctx context.Context, req *bo.LoginRequest, clientIP string) (*dto.LoginResult, error)
	Logout(c *gin.Context) error
	RefreshToken(ctx context.Context, refreshToken string) (*dto.LoginResult, error)
	GetCaptcha(ctx context.Context, clientIP string) (*dto.CaptchaResult, error)
	VerifyCaptcha(ctx context.Context, captchaKey, captchaCode string) bool
	GetAuthInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error)
	AddTokenToBlacklist(ctx context.Context, token string) error
	IsTokenBlacklisted(ctx context.Context, token string) bool
}
