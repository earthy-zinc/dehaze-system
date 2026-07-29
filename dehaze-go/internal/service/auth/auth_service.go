package auth

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	loginlogservice "github.com/earthyzinc/dehaze-go/internal/service/login_log"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"github.com/mojocn/base64Captcha"
	"github.com/google/uuid"
	"go.uber.org/zap"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"

	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
)

type AuthService struct {
	cacheClient     types.ICache
	userService     userservice.IUserService
	loginLogService *loginlogservice.LoginLogService
	db              *gorm.DB
}

func NewAuthService(cacheClient types.ICache, userService userservice.IUserService, loginLogService *loginlogservice.LoginLogService, db *gorm.DB) IAuthService {
	return &AuthService{
		cacheClient:     cacheClient,
		userService:     userService,
		loginLogService: loginLogService,
		db:              db,
	}
}

func (s *AuthService) recordLogin(ctx context.Context, userID *int64, username, ip, userAgent string, status int, message string) {
	if s.loginLogService == nil {
		return
	}
	browser, osName := parseUserAgent(userAgent)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				logger.Error("登录日志写入panic", zap.Any("panic", r))
			}
		}()
		_ = s.loginLogService.RecordLogin(ctx, userID, username, ip, status, message, browser, osName, "")
	}()
}

func parseUserAgent(ua string) (browser, os string) {
	switch {
	case strings.Contains(ua, "Windows"):
		os = "Windows"
	case strings.Contains(ua, "Mac OS"):
		os = "macOS"
	case strings.Contains(ua, "Android"):
		os = "Android"
	case strings.Contains(ua, "iPhone") || strings.Contains(ua, "iPad"):
		os = "iOS"
	case strings.Contains(ua, "Linux"):
		os = "Linux"
	}
	switch {
	case strings.Contains(ua, "Edg/"):
		browser = "Edge"
	case strings.Contains(ua, "Chrome/"):
		browser = "Chrome"
	case strings.Contains(ua, "Firefox/"):
		browser = "Firefox"
	case strings.Contains(ua, "Safari/"):
		browser = "Safari"
	}
	return
}

func (s *AuthService) Login(ctx context.Context, req *bo.LoginRequest, clientIP, userAgent string) (*dto.LoginResult, error) {
	if req == nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "登录请求不能为空")
	}

	username := strings.ToLower(strings.TrimSpace(req.Username))
	password := req.Password

	if err := s.checkLoginFailCount(ctx, clientIP, username); err != nil {
		s.recordLogin(ctx, nil, username, clientIP, userAgent, 0, err.Error())
		return nil, err
	}

	if !s.VerifyCaptcha(ctx, req.CaptchaKey, req.CaptchaCode) {
		s.incrementLoginFailCount(ctx, clientIP, username)
		s.recordLogin(ctx, nil, username, clientIP, userAgent, 0, "验证码错误")
		return nil, common.NewBizError(common.VERIFY_CODE_ERROR, "验证码错误")
	}

	u := &model.SysUser{Username: username, Password: password}
	user, err := s.userService.Login(ctx, u)
	if err != nil {
		s.incrementLoginFailCount(ctx, clientIP, username)
		logger.Warn("登录失败: 用户名不存在或密码错误",
			zap.String("username", username),
			zap.String("clientIP", clientIP),
			zap.Error(err))
		s.recordLogin(ctx, nil, username, clientIP, userAgent, 0, err.Error())
		return nil, err
	}

	if user.Status != 1 {
		s.recordLogin(ctx, &user.UserId, username, clientIP, userAgent, 0, "用户已被禁用")
		return nil, common.NewBizError(common.USER_ACCOUNT_LOCKED, "用户已被禁用")
	}

	sessionID := uuid.New().String()

	var authorities []string
	for _, role := range user.Roles {
		authorities = append(authorities, "ROLE_"+role)
	}
	authorities = append(authorities, user.Perms...)

	sessionData := middleware.SessionData{
		UserID:      user.UserId,
		Username:    user.Username,
		DeptID:      user.DeptId,
		DataScope:   user.DataScope,
		Authorities: authorities,
		Nickname:    user.Nickname,
	}

	sessionJSON, err := json.Marshal(sessionData)
	if err != nil {
		logger.Error("序列化Session数据失败", zap.Error(err))
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建Session失败", err)
	}

	if err := s.cacheClient.Set(ctx, common.SessionPrefix+sessionID, string(sessionJSON), middleware.SessionTTL); err != nil {
		logger.Error("存储Session失败", zap.Error(err))
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建Session失败", err)
	}

	cfg := config.GetConfig()
	if cfg.System.UseMultiPoint {
		if err := s.handleMultiPointSession(ctx, sessionID, user.Username); err != nil {
			return nil, err
		}
	}

	s.resetLoginFailCount(ctx, clientIP, username)

	logger.Info("用户登录成功",
		zap.String("username", username),
		zap.String("clientIP", clientIP))

	s.recordLogin(ctx, &user.UserId, username, clientIP, userAgent, 1, "登录成功")

	return &dto.LoginResult{
		SessionID: sessionID,
		User: &dto.LoginUser{
			ID:       user.UserId,
			Username: user.Username,
			Nickname: user.Nickname,
		},
	}, nil
}

func (s *AuthService) Register(ctx context.Context, req *bo.RegisterRequest, clientIP string) (*dto.LoginResult, error) {
	if req == nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "注册请求不能为空")
	}

	username := strings.ToLower(strings.TrimSpace(req.Username))
	nickname := strings.TrimSpace(req.Nickname)

	if !s.VerifyCaptcha(ctx, req.CaptchaKey, req.CaptchaCode) {
		return nil, common.NewBizError(common.VERIFY_CODE_ERROR, "验证码错误")
	}

	var existingCount int64
	if err := s.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("username = ?", username).Count(&existingCount).Error; err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "检查用户名失败", err)
	}
	if existingCount > 0 {
		return nil, common.NewBizError(common.DATA_EXISTS, "用户名已被注册")
	}

	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "密码加密失败", err)
	}

	user := &model.SysUser{
		Username: username,
		Nickname: nickname,
		Password: string(hashedPassword),
		Gender:   1,
		Status:   1,
		Deleted:  0,
	}
	if err := s.db.WithContext(ctx).Create(user).Error; err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建用户失败", err)
	}

	var guestRole model.SysRole
	if err := s.db.WithContext(ctx).
		Where("code = ? AND status = 1", "GUEST").
		First(&guestRole).Error; err == nil {
		userRole := &model.SysUserRole{UserID: user.ID, RoleID: guestRole.ID}
		s.db.WithContext(ctx).Create(userRole)
	}

	s.resetLoginFailCount(ctx, clientIP, username)

	sessionID := uuid.New().String()
	authorities := []string{"ROLE_GUEST"}

	sessionData := middleware.SessionData{
		UserID:      user.ID,
		Username:    user.Username,
		Nickname:    user.Nickname,
		DeptID:      0,
		DataScope:   guestRole.DataScope,
		Authorities: authorities,
	}

	sessionJSON, err := json.Marshal(sessionData)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建Session失败", err)
	}

	if err := s.cacheClient.Set(ctx, common.SessionPrefix+sessionID, string(sessionJSON), middleware.SessionTTL); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建Session失败", err)
	}

	logger.Info("用户注册成功", zap.String("username", username))

	return &dto.LoginResult{
		SessionID: sessionID,
		User: &dto.LoginUser{
			ID:       user.ID,
			Username: user.Username,
			Nickname: user.Nickname,
		},
	}, nil
}

func (s *AuthService) Logout(c *gin.Context) error {
	sessionID := middleware.ExtractSessionID(c)
	if sessionID != "" {
		if err := s.cacheClient.Delete(c.Request.Context(), common.SessionPrefix+sessionID); err != nil {
			logger.Error("注销失败：删除Session失败", zap.Error(err))
		}
	}

	if claims, err := security.GetClaims(c); err == nil && claims != nil {
		username := claims.Subject
		if username != "" {
			if err := s.cacheClient.Delete(c.Request.Context(), username); err != nil {
				logger.Warn("清理用户登录状态缓存失败", zap.String("username", username), zap.Error(err))
			}
		}
	}

	logger.Info("用户注销成功")
	return nil
}

func (s *AuthService) GetCaptcha(ctx context.Context, clientIP string) (*dto.CaptchaResult, error) {
	cfg := config.GetConfig()

	if cfg.Captcha.RetryCount > 0 {
		key := "captcha:limit:" + clientIP
		count, err := s.cacheClient.Get(ctx, key)
		if err == nil {
			currentCount, _ := strconv.Atoi(string(count))
			if currentCount >= cfg.Captcha.RetryCount {
				return nil, common.NewBizError(common.PARAM_ERROR, "验证码获取次数已达上限，请稍后重试")
			}
		}
		s.cacheClient.Incr(ctx, key)
		s.cacheClient.Expire(ctx, key, time.Duration(cfg.Captcha.TimeOut)*time.Second)
	}

	driver := base64Captcha.NewDriverDigit(
		cfg.Captcha.Height,
		cfg.Captcha.Width,
		cfg.Captcha.Length,
		0.7, 80)

	store := security.GetCaptchaStore()
	cp := base64Captcha.NewCaptcha(driver, store)

	id, b64s, _, err := cp.Generate()
	if err != nil {
		logger.Error("验证码生成失败", zap.Error(err))
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "验证码生成失败", err)
	}

	return &dto.CaptchaResult{
		CaptchaKey:    id,
		CaptchaBase64: b64s,
	}, nil
}

func (s *AuthService) VerifyCaptcha(ctx context.Context, captchaKey, captchaCode string) bool {
	if captchaKey == "" || captchaCode == "" {
		return false
	}

	store := security.GetCaptchaStore()
	return store.Verify(captchaKey, captchaCode, true)
}

func (s *AuthService) GetAuthInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error) {
	return s.userService.GetCurrentUserInfo(ctx, userID)
}

func (s *AuthService) handleMultiPointSession(ctx context.Context, newSessionID, username string) error {
	oldSessionID, err := s.cacheClient.Get(ctx, username)
	if err == nil && oldSessionID != "" {
		if err := s.cacheClient.Delete(ctx, common.SessionPrefix+oldSessionID); err != nil {
			logger.Warn("多端登录：删除旧Session失败", zap.String("username", username), zap.Error(err))
		} else {
			logger.Info("多端登录：已删除旧Session", zap.String("username", username))
		}
	}

	if err := s.cacheClient.Set(ctx, username, newSessionID, middleware.SessionTTL); err != nil {
		logger.Error("存储用户登录状态失败", zap.Error(err))
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置登录状态失败", err)
	}

	return nil
}

func getLoginSecurityConfig() (failLimit int, lockTime time.Duration) {
	cfg := config.GetConfig()
	failLimit = cfg.System.LoginFailLimit
	if failLimit <= 0 {
		failLimit = 5
	}
	lockSeconds := cfg.System.LoginFailLockTime
	if lockSeconds <= 0 {
		lockSeconds = 300
	}
	lockTime = time.Duration(lockSeconds) * time.Second
	return
}

func (s *AuthService) incrementLoginFailCount(ctx context.Context, clientIP, username string) {
	failLimit, lockTime := getLoginSecurityConfig()

	ipKey := "login:fail:ip:" + clientIP
	ipCount, _ := s.cacheClient.Incr(ctx, ipKey)
	s.cacheClient.Expire(ctx, ipKey, lockTime)
	if ipCount >= int64(failLimit) {
		logger.Warn("IP登录失败次数过多",
			zap.String("clientIP", clientIP),
			zap.Int64("failCount", ipCount))
	}

	if username != "" {
		userKey := "login:fail:" + username
		userCount, _ := s.cacheClient.Incr(ctx, userKey)
		s.cacheClient.Expire(ctx, userKey, lockTime)
		if userCount >= int64(failLimit) {
			logger.Warn("用户名登录失败次数过多",
				zap.String("username", username),
				zap.Int64("failCount", userCount))
		}
	}
}

func (s *AuthService) checkLoginFailCount(ctx context.Context, clientIP, username string) error {
	failLimit, _ := getLoginSecurityConfig()

	ipKey := "login:fail:ip:" + clientIP
	ipCount, err := s.cacheClient.Get(ctx, ipKey)
	if err == nil {
		count, _ := strconv.Atoi(string(ipCount))
		if count >= failLimit {
			return common.NewBizError(common.PASSWORD_ENTER_EXCEED_LIMIT, "登录失败次数过多，IP已临时锁定，请稍后重试")
		}
	}

	if username != "" {
		userKey := "login:fail:" + username
		userCount, err := s.cacheClient.Get(ctx, userKey)
		if err == nil {
			count, _ := strconv.Atoi(string(userCount))
			if count >= failLimit {
				return common.NewBizError(common.PASSWORD_ENTER_EXCEED_LIMIT, "登录失败次数过多，账户已临时锁定，请稍后重试")
			}
		}
	}

	return nil
}

func (s *AuthService) resetLoginFailCount(ctx context.Context, clientIP, username string) {
	ipKey := "login:fail:ip:" + clientIP
	if err := s.cacheClient.Delete(ctx, ipKey); err != nil {
		logger.Warn("重置IP登录失败次数失败", zap.String("clientIP", clientIP), zap.Error(err))
	}

	if username != "" {
		userKey := "login:fail:" + username
		if err := s.cacheClient.Delete(ctx, userKey); err != nil {
			logger.Warn("重置用户名登录失败次数失败", zap.String("username", username), zap.Error(err))
		}
	}
}
