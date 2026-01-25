package api

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"github.com/gin-gonic/gin"
	"github.com/mojocn/base64Captcha"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

type AuthApi struct {
	cacheClient types.ICache
}

func (a *AuthApi) Captcha(c *gin.Context) {
	// 判断验证码是否开启
	cfg := config.GetConfig()
	openCaptcha := cfg.Captcha.RetryCount     // 是否开启防爆次数
	openCaptchaTimeOut := cfg.Captcha.TimeOut // 缓存超时时间
	key := c.ClientIP()

	v, err := a.cacheClient.Get(c, key)
	if err != nil {
		a.cacheClient.Set(c, key, 1, time.Second*time.Duration(openCaptchaTimeOut))
	}

	if openCaptcha != 0 && utils.InterfaceToInt(v) >= openCaptcha {
		common.FailWithMessage("验证码获取失败，已经达到最大获取次数，请稍后重试", c)
		return
	}
	// 字符,公式,验证码配置
	// 生成默认数字的driver
	driver := base64Captcha.NewDriverDigit(
		cfg.Captcha.Height,
		cfg.Captcha.Width,
		cfg.Captcha.Length,
		0.7, 80)
	var cp *base64Captcha.Captcha
	var store = security.GetCaptchaStore()
	cp = base64Captcha.NewCaptcha(driver, store)

	id, b64s, _, err := cp.Generate()

	if err != nil {
		logger.Error("验证码获取失败!", zap.Error(err))
		common.FailWithMessage("验证码获取失败", c)
		return
	}

	common.OkWithDetailed(
		gin.H{
			"captchaKey":    id,
			"captchaBase64": b64s,
		},
		"验证码获取成功",
		c,
	)
}

// Login User login structure
type Login struct {
	Username    string `form:"username" json:"username" validate:"required,min=3"` // 用户名，至少3个字符
	Password    string `form:"password" json:"password" validate:"required,min=6"` // 密码，至少6个字符
	CaptchaCode string `form:"captchaCode" json:"captchaCode" validate:"required"` // 验证码，必须存在
	CaptchaKey  string `form:"captchaKey" json:"captchaKey" validate:"required"`   // 验证码ID，必须存在
}

func (a *AuthApi) Login(c *gin.Context) {
	var loginReq Login
	if err := c.ShouldBind(&loginReq); err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	userIp := c.ClientIP()
	// 判断验证码是否开启
	cfg := config.GetConfig()
	retryCount := cfg.Captcha.RetryCount  // 是否开启防爆次数
	captchaTimeOut := cfg.Captcha.TimeOut // 缓存超时时间
	v, err := a.cacheClient.Get(c, userIp)
	if err != nil {
		a.cacheClient.Set(c, userIp, 1, time.Second*time.Duration(captchaTimeOut))
	}

	var oc = retryCount == 0 || retryCount < utils.InterfaceToInt(v)
	var store = security.GetCaptchaStore()
	if !oc && (loginReq.CaptchaCode == "" || loginReq.CaptchaKey == "" || !store.Verify(loginReq.CaptchaKey, loginReq.CaptchaCode, true)) {
		// 验证码次数+1
		a.cacheClient.Incr(c, userIp)
		common.FailWithMessage("验证码错误", c)
		return
	}

	u := &model.SysUser{Username: loginReq.Username, Password: loginReq.Password}
	user, err := getUserService().Login(c.Request.Context(), u)
	if err != nil {
		logger.Error("登陆失败! 用户名不存在或者密码错误!", zap.Error(err))
		// 验证码次数+1
		a.cacheClient.Incr(c, userIp)
		common.FailWithMessage("用户名不存在或者密码错误", c)
		return
	}

	if user.Status != 1 {
		common.FailWithMessage("用户已被禁用", c)
		return
	}

	token, claims, err := security.LoginToken(user)
	if err != nil {
		logger.Error("获取token失败!", zap.Error(err))
		common.FailWithMessage("获取token失败", c)
		return
	}
	if !cfg.System.UseMultiPoint {
		security.SetToken(c, token, int(claims.RegisteredClaims.ExpiresAt.Unix()-time.Now().Unix()))
		common.OkWithDetailed(gin.H{
			"accessToken": token,
			"tokenType":   "Bearer",
		}, "登录成功", c)
		return
	}

	if jwt, err := a.cacheClient.Get(context.Background(), user.Username); err == redis.Nil {
		if err := security.SetJWT(token, user.Username); err != nil {
			logger.Error("设置登录状态失败!", zap.Error(err))
			common.FailWithMessage("设置登录状态失败", c)
			return
		}
		security.SetToken(c, token, int(claims.RegisteredClaims.ExpiresAt.Unix()-time.Now().Unix()))
		common.OkWithDetailed(gin.H{
			"accessToken": token,
			"tokenType":   "Bearer",
		}, "登录成功", c)
		return
	} else if err != nil {
		logger.Error("设置登录状态失败!", zap.Error(err))
		common.FailWithMessage("设置登录状态失败", c)
	} else {
		// 设置JWT黑名单
		a.cacheClient.Set(context.Background(), common.BlacklistPrefix+jwt, nil, time.Duration(cfg.JWT.TTL))

		if err := security.SetJWT(token, user.Username); err != nil {
			common.FailWithMessage("设置登录状态失败", c)
			return
		}
		security.SetToken(c, token, int(claims.RegisteredClaims.ExpiresAt.Unix()-time.Now().Unix()))
		common.OkWithDetailed(gin.H{
			"accessToken": token,
			"tokenType":   "Bearer",
		}, "登录成功", c)
	}
}
