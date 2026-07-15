package auth

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"github.com/mojocn/base64Captcha"
	"go.uber.org/zap"
)

// AuthService 认证服务实现
type AuthService struct {
	cacheClient types.ICache
	userService userservice.IUserService
}

// NewAuthService 创建认证服务实例
func NewAuthService(cacheClient types.ICache, userService userservice.IUserService) IAuthService {
	return &AuthService{
		cacheClient: cacheClient,
		userService: userService,
	}
}

// Login 用户登录
func (s *AuthService) Login(ctx context.Context, req *bo.LoginRequest, clientIP string) (*dto.LoginResult, error) {
	if req == nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "登录请求不能为空")
	}

	// 1. 用户名预处理：小写转换和空格清理
	username := strings.ToLower(strings.TrimSpace(req.Username))
	password := req.Password

	// 2. 检查登录失败次数是否超限（防暴力破解，双重维度：IP + 用户名）
	if err := s.checkLoginFailCount(ctx, clientIP, username); err != nil {
		return nil, err
	}

	// 3. 验证码校验
	if !s.VerifyCaptcha(ctx, req.CaptchaKey, req.CaptchaCode) {
		// 记录登录失败次数（双重维度：IP + 用户名）
		s.incrementLoginFailCount(ctx, clientIP, username)
		return nil, common.NewBizError(common.VERIFY_CODE_ERROR, "验证码错误")
	}

	// 4. 用户认证
	u := &model.SysUser{Username: username, Password: password}
	user, err := s.userService.Login(ctx, u)
	if err != nil {
		// 记录登录失败次数（双重维度：IP + 用户名）
		s.incrementLoginFailCount(ctx, clientIP, username)
		logger.Warn("登录失败: 用户名不存在或密码错误",
			zap.String("username", username),
			zap.String("clientIP", clientIP),
			zap.Error(err))
		return nil, err
	}

	// 5. 检查用户状态
	if user.Status != 1 {
		return nil, common.NewBizError(common.USER_ACCOUNT_LOCKED, "用户已被禁用")
	}

	// 6. 生成JWT Token（双Token机制）
	accessToken, refreshToken, accessClaims, _, err := security.LoginTokenWithRefresh(user)
	if err != nil {
		logger.Error("生成Token失败", zap.Error(err))
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "生成Token失败", err)
	}

	// 7. 处理多端登录互斥
	cfg := config.GetConfig()
	if cfg.System.UseMultiPoint {
		if err := s.handleMultiPointLogin(ctx, accessToken, user.Username); err != nil {
			return nil, err
		}
	}

	// 8. 登录成功，重置失败次数
	s.resetLoginFailCount(ctx, clientIP, username)

	// 9. 计算过期时间（毫秒）
	expires := int64(0)
	if accessClaims.ExpiresAt != nil {
		expires = accessClaims.ExpiresAt.Unix() * 1000 // 转换为毫秒
	}

	logger.Info("用户登录成功",
		zap.String("username", username),
		zap.String("clientIP", clientIP))

	return &dto.LoginResult{
		AccessToken:  accessToken,
		TokenType:    "Bearer",
		RefreshToken: refreshToken,
		Expires:      expires,
		User: &dto.LoginUser{
			ID:       user.UserId,
			Username: user.Username,
			Nickname: user.Nickname,
		},
	}, nil
}

// Logout 用户注销
func (s *AuthService) Logout(c *gin.Context) error {
	token := security.GetToken(c)
	if token == "" {
		return nil // 无Token视为已注销
	}

	// 将Token加入黑名单
	if err := s.AddTokenToBlacklist(c.Request.Context(), token); err != nil {
		logger.Error("注销失败：加入黑名单失败", zap.Error(err))
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "注销失败", err)
	}

	// 清理Cookie中的Token
	security.ClearToken(c)

	// 尝试从Claims获取用户名，清理多端登录缓存
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

// RefreshToken 刷新令牌
func (s *AuthService) RefreshToken(ctx context.Context, refreshToken string) (*dto.LoginResult, error) {
	if refreshToken == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "刷新令牌不能为空")
	}

	// 1. 解析并验证refreshToken
	j := security.NewJWT()
	claims, err := j.ParseToken(refreshToken)
	if err != nil {
		if errors.Is(err, security.ErrTokenExpired) {
			return nil, common.NewBizError(common.TOKEN_INVALID, "刷新令牌已过期，请重新登录")
		}
		return nil, common.NewBizError(common.TOKEN_INVALID, "无效的刷新令牌")
	}

	// 2. 检查refreshToken是否在黑名单
	if s.IsTokenBlacklisted(ctx, refreshToken) {
		return nil, common.NewBizError(common.TOKEN_INVALID, "刷新令牌已失效，请重新登录")
	}

	// 3. 获取用户最新认证信息（确保权限实时更新）
	userAuthInfo, err := s.userService.GetUserAuthInfo(ctx, claims.Subject)
	if err != nil {
		return nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")
	}

	// 4. 检查用户状态
	if userAuthInfo.Status != 1 {
		return nil, common.NewBizError(common.USER_ACCOUNT_LOCKED, "用户已被禁用")
	}

	// 5. 生成新的Token对
	newAccessToken, newRefreshToken, newAccessClaims, _, err := security.LoginTokenWithRefresh(userAuthInfo)
	if err != nil {
		logger.Error("生成新Token失败", zap.Error(err))
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "生成Token失败", err)
	}

	// 6. 将旧的refreshToken加入黑名单（通过jti）
	cfg := config.GetConfig()
	refreshTTL := cfg.JWT.RefreshTokenTTL
	if refreshTTL <= 0 {
		refreshTTL = 7 * 24 * 3600
	}
	// 解析旧refreshToken获取jti
	if oldClaims, parseErr := j.ParseToken(refreshToken); parseErr == nil && oldClaims.ID != "" {
		if err := s.cacheClient.Set(ctx, common.BlacklistPrefix+oldClaims.ID, "1", time.Duration(refreshTTL)*time.Second); err != nil {
			logger.Warn("将旧refreshToken加入黑名单失败", zap.Error(err))
		}
	}

	// 7. 处理多端登录互斥
	if cfg.System.UseMultiPoint {
		if err := s.handleMultiPointLogin(ctx, newAccessToken, userAuthInfo.Username); err != nil {
			return nil, err
		}
	}

	// 8. 计算过期时间（毫秒）
	expires := int64(0)
	if newAccessClaims.ExpiresAt != nil {
		expires = newAccessClaims.ExpiresAt.Unix() * 1000
	}

	logger.Info("令牌刷新成功", zap.String("username", userAuthInfo.Username))

	return &dto.LoginResult{
		AccessToken:  newAccessToken,
		TokenType:    "Bearer",
		RefreshToken: newRefreshToken,
		Expires:      expires,
		User: &dto.LoginUser{
			ID:       userAuthInfo.UserId,
			Username: userAuthInfo.Username,
			Nickname: userAuthInfo.Nickname,
		},
	}, nil
}

// GetCaptcha 获取验证码
func (s *AuthService) GetCaptcha(ctx context.Context, clientIP string) (*dto.CaptchaResult, error) {
	cfg := config.GetConfig()

	// 检查验证码获取次数限制（防刷）
	if cfg.Captcha.RetryCount > 0 {
		key := "captcha:limit:" + clientIP
		count, err := s.cacheClient.Get(ctx, key)
		if err == nil {
			currentCount, _ := strconv.Atoi(string(count))
			if currentCount >= cfg.Captcha.RetryCount {
				return nil, common.NewBizError(common.PARAM_ERROR, "验证码获取次数已达上限，请稍后重试")
			}
		}
		// 增加获取次数
		s.cacheClient.Incr(ctx, key)
		s.cacheClient.Expire(ctx, key, time.Duration(cfg.Captcha.TimeOut)*time.Second)
	}

	// 生成验证码
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

// VerifyCaptcha 校验验证码
func (s *AuthService) VerifyCaptcha(ctx context.Context, captchaKey, captchaCode string) bool {
	if captchaKey == "" || captchaCode == "" {
		return false
	}

	store := security.GetCaptchaStore()
	return store.Verify(captchaKey, captchaCode, true)
}

// GetAuthInfo 获取当前用户认证信息
func (s *AuthService) GetAuthInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error) {
	return s.userService.GetCurrentUserInfo(ctx, userID)
}

// AddTokenToBlacklist 将Token加入黑名单（存储jti而非完整Token，节省内存）
func (s *AuthService) AddTokenToBlacklist(ctx context.Context, token string) error {
	if token == "" {
		return nil
	}

	// 解析Token获取jti
	j := security.NewJWT()
	claims, err := j.ParseToken(token)
	if err != nil {
		if errors.Is(err, security.ErrTokenExpired) {
			// Token已过期，无法通过验证，无需加入黑名单
			return nil
		}
		return common.WrapBizError(common.TOKEN_INVALID, "解析Token失败", err)
	}

	jti := claims.ID
	if jti == "" {
		return common.NewBizError(common.SYSTEM_EXECUTION_ERROR, "Token缺少jti")
	}

	// 黑名单过期时间使用 Token 实际剩余有效期，避免过度占用 Redis 内存
	remaining := time.Until(claims.ExpiresAt.Time)
	if remaining <= 0 {
		logger.Info("Token已过期，无需加入黑名单", zap.String("jti", jti))
		return nil
	}
	ttl := remaining

	if err := s.cacheClient.Set(ctx, common.BlacklistPrefix+jti, "1", ttl); err != nil {
		return err
	}

	logger.Info("Token已加入黑名单", zap.String("jti", jti))
	return nil
}

// IsTokenBlacklisted 检查Token是否在黑名单中（通过jti检查）
func (s *AuthService) IsTokenBlacklisted(ctx context.Context, token string) bool {
	if token == "" {
		return false
	}

	// 解析Token获取jti
	j := security.NewJWT()
	claims, err := j.ParseToken(token)
	if err != nil {
		// Token解析失败，视为无效Token
		return false
	}

	jti := claims.ID
	if jti == "" {
		return false
	}

	exists, err := s.cacheClient.Exists(ctx, common.BlacklistPrefix+jti)
	if err != nil {
		// 缓存服务异常时，记录日志但不阻止请求（保证服务可用性）
		logger.Error("检查Token黑名单失败", zap.Error(err))
		return false
	}
	return exists
}

// handleMultiPointLogin 处理多端登录互斥
func (s *AuthService) handleMultiPointLogin(ctx context.Context, newToken, username string) error {
	// 获取当前用户的旧Token
	oldToken, err := s.cacheClient.Get(ctx, username)
	if err == nil && oldToken != "" {
		// 将旧Token加入黑名单（通过jti）
		j := security.NewJWT()
		if oldClaims, parseErr := j.ParseToken(oldToken); parseErr == nil && oldClaims.ID != "" {
			cfg := config.GetConfig()
			ttl := time.Duration(cfg.JWT.TTL) * time.Second
			if setErr := s.cacheClient.Set(ctx, common.BlacklistPrefix+oldClaims.ID, "1", ttl); setErr != nil {
				logger.Error("多端登录：将旧Token加入黑名单失败", zap.String("username", username), zap.String("jti", oldClaims.ID), zap.Error(setErr))
			} else {
				logger.Info("多端登录：已将旧Token加入黑名单", zap.String("username", username), zap.String("jti", oldClaims.ID))
			}
		}
	}

	// 存储新Token
	if err := security.SetJWT(ctx, newToken, username); err != nil {
		logger.Error("存储用户登录状态失败", zap.Error(err))
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置登录状态失败", err)
	}

	return nil
}

// getLoginSecurityConfig 获取登录安全配置，提供默认值
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

// incrementLoginFailCount 增加登录失败次数（双重维度：IP + 用户名）
func (s *AuthService) incrementLoginFailCount(ctx context.Context, clientIP, username string) {
	failLimit, lockTime := getLoginSecurityConfig()

	// IP维度计数
	ipKey := "login:fail:ip:" + clientIP
	ipCount, _ := s.cacheClient.Incr(ctx, ipKey)
	s.cacheClient.Expire(ctx, ipKey, lockTime)
	if ipCount >= int64(failLimit) {
		logger.Warn("IP登录失败次数过多，可能存在暴力破解风险",
			zap.String("clientIP", clientIP),
			zap.Int64("failCount", ipCount))
	}

	// 用户名维度计数
	if username != "" {
		userKey := "login:fail:user:" + username
		userCount, _ := s.cacheClient.Incr(ctx, userKey)
		s.cacheClient.Expire(ctx, userKey, lockTime)
		if userCount >= int64(failLimit) {
			logger.Warn("用户名登录失败次数过多，可能存在暴力破解风险",
				zap.String("username", username),
				zap.Int64("failCount", userCount))
		}
	}
}

// checkLoginFailCount 检查登录失败次数是否超限（双重维度：IP + 用户名）
func (s *AuthService) checkLoginFailCount(ctx context.Context, clientIP, username string) error {
	failLimit, _ := getLoginSecurityConfig()

	// 检查IP维度
	ipKey := "login:fail:ip:" + clientIP
	ipCount, err := s.cacheClient.Get(ctx, ipKey)
	if err == nil {
		count, _ := strconv.Atoi(string(ipCount))
		if count >= failLimit {
			logger.Warn("IP登录失败次数超限，已临时锁定",
				zap.String("clientIP", clientIP),
				zap.Int("failCount", count))
			return common.NewBizError(common.PASSWORD_ENTER_EXCEED_LIMIT, "登录失败次数过多，IP已临时锁定，请稍后重试")
		}
	}

	// 检查用户名维度
	if username != "" {
		userKey := "login:fail:user:" + username
		userCount, err := s.cacheClient.Get(ctx, userKey)
		if err == nil {
			count, _ := strconv.Atoi(string(userCount))
			if count >= failLimit {
				logger.Warn("用户名登录失败次数超限，已临时锁定",
					zap.String("username", username),
					zap.Int("failCount", count))
				return common.NewBizError(common.PASSWORD_ENTER_EXCEED_LIMIT, "登录失败次数过多，账户已临时锁定，请稍后重试")
			}
		}
	}

	return nil
}

// resetLoginFailCount 重置登录失败次数（双重维度：IP + 用户名）
func (s *AuthService) resetLoginFailCount(ctx context.Context, clientIP, username string) {
	// 重置IP维度
	ipKey := "login:fail:ip:" + clientIP
	if err := s.cacheClient.Delete(ctx, ipKey); err != nil {
		logger.Warn("重置IP登录失败次数失败", zap.String("clientIP", clientIP), zap.Error(err))
	}

	// 重置用户名维度
	if username != "" {
		userKey := "login:fail:user:" + username
		if err := s.cacheClient.Delete(ctx, userKey); err != nil {
			logger.Warn("重置用户名登录失败次数失败", zap.String("username", username), zap.Error(err))
		}
	}
}
