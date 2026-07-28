package auth

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/gin-gonic/gin"
)

type IAuthService interface {
	Login(ctx context.Context, req *bo.LoginRequest, clientIP, userAgent string) (*dto.LoginResult, error)
	Register(ctx context.Context, req *bo.RegisterRequest, clientIP string) (*dto.LoginResult, error)
	Logout(c *gin.Context) error
	GetCaptcha(ctx context.Context, clientIP string) (*dto.CaptchaResult, error)
	VerifyCaptcha(ctx context.Context, captchaKey, captchaCode string) bool
	GetAuthInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error)
}
