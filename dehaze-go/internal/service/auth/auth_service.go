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
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	loginlogservice "github.com/earthyzinc/dehaze-go/internal/service/login_log"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	"github.com/earthyzinc/dehaze-go/internal/service/session"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/mojocn/base64Captcha"
	"go.uber.org/zap"

	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
)

type AuthService struct {
	cacheClient     types.ICache
	userService     userservice.IUserService
	loginLogService *loginlogservice.LoginLogService
	memberService   memberservice.IMemberService
}

func NewAuthService(cacheClient types.ICache, userService userservice.IUserService, loginLogService *loginlogservice.LoginLogService, memberService memberservice.IMemberService) IAuthService {
	return &AuthService{
		cacheClient:     cacheClient,
		userService:     userService,
		loginLogService: loginLogService,
		memberService:   memberService,
	}
}

func (s *AuthService) recordLogin(ctx context.Context, userID *int64, username, ip, userAgent string, status int, message, deviceType string) {
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
		writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_ = s.loginLogService.RecordLogin(writeCtx, userID, username, ip, status, message, browser, osName, deviceType)
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
	deviceType := normalizeDeviceType(req.DeviceType)

	if err := s.checkLoginFailCount(ctx, clientIP, username); err != nil {
		s.recordLogin(ctx, nil, username, clientIP, userAgent, 0, err.Error(), deviceType)
		return nil, err
	}

	captchaOK, captchaExpired := s.VerifyCaptchaStatus(ctx, req.CaptchaKey, req.CaptchaCode)
	if !captchaOK {
		s.incrementLoginFailCount(ctx, clientIP, username)
		// 与 Python 端一致：Key 不存在/已消费/超时 → A0213，比对失败 → A0214
		code, msg := common.VERIFY_CODE_ERROR, "验证码错误"
		if captchaExpired {
			code, msg = common.VERIFY_CODE_TIMEOUT, "验证码已过期"
		}
		s.recordLogin(ctx, nil, username, clientIP, userAgent, 0, msg, deviceType)
		return nil, common.NewBizError(code, msg)
	}

	u := &model.SysUser{Username: username, Password: password}
	user, err := s.userService.Login(ctx, u)
	if err != nil {
		s.incrementLoginFailCount(ctx, clientIP, username)
		logger.Warn("登录失败: 用户名不存在或密码错误",
			zap.String("username", username),
			zap.String("clientIP", clientIP),
			zap.Error(err))
		s.recordLogin(ctx, nil, username, clientIP, userAgent, 0, err.Error(), deviceType)
		return nil, err
	}

	if user.Status != 1 {
		s.recordLogin(ctx, &user.UserId, username, clientIP, userAgent, 0, "用户已被禁用", deviceType)
		return nil, common.NewBizError(common.USER_ACCOUNT_LOCKED, "用户已被禁用")
	}

	// 会员档案兜底：种子账号与后台创建的用户不走注册流程，登录时确保
	// sys_member 行存在（否则计费配额校验 fail-closed 误报"配额不足"）
	if err := s.memberService.EnsureMemberProfile(ctx, user.UserId); err != nil {
		return nil, err
	}

	sessionID := uuid.New().String()

	var authorities []string
	for _, role := range user.Roles {
		authorities = append(authorities, "ROLE_"+role)
	}
	authorities = append(authorities, user.Perms...)

	sessionData := middleware.SessionData{
		UserID:         user.UserId,
		Username:       user.Username,
		DeptID:         user.DeptId,
		DataScope:      user.DataScope,
		Authorities:    authorities,
		Nickname:       user.Nickname,
		DeviceType:     deviceType,
		LoginIP:        clientIP,
		LoginTime:      time.Now().Format("2006-01-02 15:04:05"),
		LastAccessTime: time.Now().Format("2006-01-02 15:04:05"),
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
		// 管理员（ROOT/ADMIN）不受等级权益约束，固定 10 台；普通用户取等级权益 max_devices
		maxDevices := session.AdminMaxDevices
		adminSession := false
		for _, role := range user.Roles {
			if role == "ROOT" || role == "ADMIN" {
				adminSession = true
				break
			}
		}
		if !adminSession {
			md, mdErr := s.memberService.GetMaxDevices(ctx, user.UserId)
			if mdErr != nil {
				return nil, mdErr
			}
			maxDevices = md
		}
		if err := session.RegisterSession(ctx, user.UserId, sessionID, maxDevices); err != nil {
			return nil, err
		}
	}

	s.resetLoginFailCount(ctx, clientIP, username)

	logger.Info("用户登录成功",
		zap.String("username", username),
		zap.String("clientIP", clientIP))

	s.recordLogin(ctx, &user.UserId, username, clientIP, userAgent, 1, "登录成功", deviceType)

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

	captchaOK, captchaExpired := s.VerifyCaptchaStatus(ctx, req.CaptchaKey, req.CaptchaCode)
	if !captchaOK {
		code, msg := common.VERIFY_CODE_ERROR, "验证码错误"
		if captchaExpired {
			code, msg = common.VERIFY_CODE_TIMEOUT, "验证码已过期"
		}
		return nil, common.NewBizError(code, msg)
	}

	// 通过 UserService 完成用户创建 + GUEST 角色分配，不在 auth 层直接操作用户表
	user, dataScope, err := s.userService.Register(ctx, username, nickname, req.Password)
	if err != nil {
		return nil, err
	}

	// 通过 MemberService 初始化默认会员记录
	if err := s.memberService.InitDefaultMember(ctx, user.ID); err != nil {
		return nil, err
	}

	s.resetLoginFailCount(ctx, clientIP, username)

	sessionID := uuid.New().String()
	authorities := []string{"ROLE_GUEST"}

	sessionData := middleware.SessionData{
		UserID:      user.ID,
		Username:    user.Username,
		Nickname:    user.Nickname,
		DeptID:      0,
		DataScope:   dataScope,
		Authorities: authorities,
	}

	sessionJSON, err := json.Marshal(sessionData)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建Session失败", err)
	}

	if err := s.cacheClient.Set(ctx, common.SessionPrefix+sessionID, string(sessionJSON), middleware.SessionTTL); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建Session失败", err)
	}

	// 注册签发的会话同样登记进设备数索引，否则该会话不占额度（与登录路径一致）
	if config.GetConfig().System.UseMultiPoint {
		maxDevices, mdErr := s.memberService.GetMaxDevices(ctx, user.ID)
		if mdErr != nil {
			return nil, mdErr
		}
		if err := session.RegisterSession(ctx, user.ID, sessionID, maxDevices); err != nil {
			return nil, err
		}
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
		// 会话索引（session:user:{userId} ZSet）剔除本会话元素，其他端在线会话不受影响
		if userID, err := security.RequireUserID(c); err == nil {
			if rmErr := session.RemoveFromIndex(c.Request.Context(), userID, sessionID); rmErr != nil {
				logger.Warn("注销清理会话索引失败", zap.Error(rmErr))
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

// VerifyCaptchaStatus 校验验证码并消费（GETDEL 语义，杜绝并发重放）。
// 返回 (是否通过, Key 是否不存在/已过期)：与 Python 端 verify_captcha_status 一致，
// 用于区分 A0214（验证码错误）与 A0213（验证码已过期）。
func (s *AuthService) VerifyCaptchaStatus(ctx context.Context, captchaKey, captchaCode string) (ok bool, expired bool) {
	if captchaKey == "" {
		return false, true
	}

	stored := security.GetCaptchaStore().Get(captchaKey, true)
	if stored == "" {
		return false, true
	}
	return strings.EqualFold(stored, captchaCode), false
}

func (s *AuthService) GetAuthInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error) {
	return s.userService.GetCurrentUserInfo(ctx, userID)
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

// normalizeDeviceType 设备类型归一化，与 Python 端 DEVICE_TYPES 对齐
func normalizeDeviceType(deviceType string) string {
	switch deviceType {
	case "web", "android", "flutter", "miniprogram":
		return deviceType
	default:
		return "web"
	}
}

// ListLoginLogs 登录日志分页查询（q.UserIDs 非空时限定可见用户范围：普通用户仅本人）
func (s *AuthService) ListLoginLogs(ctx context.Context, q *query.LoginLogQuery) (*vo.PageResult[vo.LoginLogVO], error) {
	if s.loginLogService == nil {
		return nil, common.NewBizError(common.SYSTEM_EXECUTION_ERROR, "登录日志服务不可用")
	}

	logs, total, err := s.loginLogService.PageLogs(ctx, q)
	if err != nil {
		return nil, err
	}

	list := make([]vo.LoginLogVO, 0, len(logs))
	for _, item := range logs {
		list = append(list, vo.LoginLogVO{
			ID:         item.ID.Hex(),
			UserID:     item.UserID,
			Username:   item.Username,
			IP:         item.IP,
			Location:   item.Location,
			Browser:    item.Browser,
			OS:         item.OS,
			DeviceType: item.DeviceType,
			Status:     item.Status,
			Message:    item.Message,
			LoginTime:  item.CreateTime.Format("2006-01-02 15:04:05"),
		})
	}
	return &vo.PageResult[vo.LoginLogVO]{List: list, Total: total}, nil
}

// ListSessions 在线会话列表（按用户名精确过滤）
func (s *AuthService) ListSessions(ctx context.Context, username string) ([]session.SessionInfo, error) {
	return session.ListByUsername(ctx, username)
}

// KickSession 踢出指定在线会话
func (s *AuthService) KickSession(ctx context.Context, sessionID string) error {
	return session.KickByID(ctx, sessionID)
}
