package api

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/utils"
	"github.com/gin-gonic/gin"
	"github.com/mojocn/base64Captcha"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

type AuthApi struct{}

// 类型转换
func interfaceToInt(v any) (i int) {
	switch v := v.(type) {
	case int:
		i = v
	case int8:
		i = int(v)
	case int16:
		i = int(v)
	case int32:
		i = int(v)
	case int64:
		i = int(v)
	case uint:
		i = int(v)
	case uint8:
		i = int(v)
	case uint16:
		i = int(v)
	case uint32:
		i = int(v)
	case uint64:
		i = int(v)
	default:
		i = 0
	}
	return
}

func (a *AuthApi) Captcha(c *gin.Context) {
	// 判断验证码是否开启
	openCaptcha := global.CONFIG.Captcha.RetryCount     // 是否开启防爆次数
	openCaptchaTimeOut := global.CONFIG.Captcha.TimeOut // 缓存超时时间
	key := c.ClientIP()
	v, ok := global.LOCAL_CACHE.Get(key)
	if !ok {
		global.LOCAL_CACHE.Set(key, 1, time.Second*time.Duration(openCaptchaTimeOut))
	}

	if openCaptcha != 0 && interfaceToInt(v) >= openCaptcha {
		common.FailWithMessage("验证码获取失败，已经达到最大获取次数，请稍后重试", c)
		return
	}
	// 字符,公式,验证码配置
	// 生成默认数字的driver
	driver := base64Captcha.NewDriverDigit(
		global.CONFIG.Captcha.Height,
		global.CONFIG.Captcha.Width,
		global.CONFIG.Captcha.Length,
		0.7, 80)
	var cp *base64Captcha.Captcha
	var store = utils.GetCaptchaStore()
	if global.REDIS != nil {
		cp = base64Captcha.NewCaptcha(driver, store.(*utils.RedisStore).UseWithCtx(c))
	} else {
		cp = base64Captcha.NewCaptcha(driver, store)
	}

	id, b64s, _, err := cp.Generate()

	if err != nil {
		global.LOG.Error("验证码获取失败!", zap.Error(err))
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
	if err := c.ShouldBindQuery(&loginReq); err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	key := c.ClientIP()
	// 判断验证码是否开启
	openCaptcha := global.CONFIG.Captcha.RetryCount     // 是否开启防爆次数
	openCaptchaTimeOut := global.CONFIG.Captcha.TimeOut // 缓存超时时间
	v, ok := global.LOCAL_CACHE.Get(key)
	if !ok {
		global.LOCAL_CACHE.Set(key, 1, time.Second*time.Duration(openCaptchaTimeOut))
	}

	var oc bool = openCaptcha == 0 || openCaptcha < interfaceToInt(v)
	var store = utils.GetCaptchaStore()
	if !oc && (loginReq.CaptchaCode == "" || loginReq.CaptchaKey == "" || !store.Verify(loginReq.CaptchaKey, loginReq.CaptchaCode, true)) {
		// 验证码次数+1
		global.LOCAL_CACHE.Increment(key, 1)
		common.FailWithMessage("验证码错误", c)
		return
	}

	u := &model.SysUser{Username: loginReq.Username, Password: loginReq.Password}
	user, err := userService.Login(u)
	if err != nil {
		global.LOG.Error("登陆失败! 用户名不存在或者密码错误!", zap.Error(err))
		// 验证码次数+1
		global.LOCAL_CACHE.Increment(key, 1)
		common.FailWithMessage("用户名不存在或者密码错误", c)
		return
	}

	if user.Status != 1 {
		common.FailWithMessage("用户已被禁用", c)
		return
	}

	token, claims, err := utils.LoginToken(user)
	if err != nil {
		global.LOG.Error("获取token失败!", zap.Error(err))
		common.FailWithMessage("获取token失败", c)
		return
	}
	if !global.CONFIG.System.UseMultiPoint {
		utils.SetToken(c, token, int(claims.RegisteredClaims.ExpiresAt.Unix()-time.Now().Unix()))
		common.OkWithDetailed(gin.H{
			"accessToken": token,
			"tokenType":   "Bearer",
		}, "登录成功", c)
		return
	}

	if global.REDIS != nil {
		if jwt, err := global.REDIS.Get(context.Background(), user.Username).Result(); err == redis.Nil {
			if err := utils.SetRedisJWT(token, user.Username); err != nil {
				global.LOG.Error("设置登录状态失败!", zap.Error(err))
				common.FailWithMessage("设置登录状态失败", c)
				return
			}
			utils.SetToken(c, token, int(claims.RegisteredClaims.ExpiresAt.Unix()-time.Now().Unix()))
			common.OkWithDetailed(gin.H{
				"accessToken": token,
				"tokenType":   "Bearer",
			}, "登录成功", c)
			return
		} else if err != nil {
			global.LOG.Error("设置登录状态失败!", zap.Error(err))
			common.FailWithMessage("设置登录状态失败", c)
		} else {
			// 设置JWT黑名单
			global.REDIS.Set(context.Background(), common.BLACKLIST_PREFIX+jwt, nil, time.Duration(global.CONFIG.JWT.TTL))

			if err := utils.SetRedisJWT(token, user.Username); err != nil {
				common.FailWithMessage("设置登录状态失败", c)
				return
			}
			utils.SetToken(c, token, int(claims.RegisteredClaims.ExpiresAt.Unix()-time.Now().Unix()))
			common.OkWithDetailed(gin.H{
				"accessToken": token,
				"tokenType":   "Bearer",
			}, "登录成功", c)
		}
	} else {
		// global.LOCAL_CACHE.SetDefault(common.BLACKLIST_PREFIX+jwt, struct{}{})
		common.FailWithMessage("服务端未设置redis，暂时无法签发jwt", c)
	}
}
