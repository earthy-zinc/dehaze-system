package api

import (
	"net/http"
	"time"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/utils"
	"github.com/gin-gonic/gin"
	"github.com/mojocn/base64Captcha"
	"go.uber.org/zap"
)

type AuthApi struct{}

var store = utils.NewCaptchaStore()

// 类型转换
func interfaceToInt(v interface{}) (i int) {
	switch v := v.(type) {
	case int:
		i = v
	default:
		i = 0
	}
	return
}

func (a *AuthApi) Captcha(c *gin.Context) {
	// 判断验证码是否开启
	openCaptcha := global.CONFIG.Captcha.OpenCaptcha               // 是否开启防爆次数
	openCaptchaTimeOut := global.CONFIG.Captcha.OpenCaptchaTimeOut // 缓存超时时间
	key := c.ClientIP()
	v, ok := global.BlackCache.Get(key)
	if !ok {
		global.BlackCache.Set(key, 1, time.Second*time.Duration(openCaptchaTimeOut))
	}

	if openCaptcha != 0 && interfaceToInt(v) >= openCaptcha {
		common.FailWithMessage("验证码获取失败，已经达到最大获取次数，请稍后重试", c)
		return
	}
	// 字符,公式,验证码配置
	// 生成默认数字的driver
	driver := base64Captcha.NewDriverDigit(
		global.CONFIG.Captcha.ImgHeight,
		global.CONFIG.Captcha.ImgWidth,
		global.CONFIG.Captcha.KeyLong,
		0.7, 80)
	var cp *base64Captcha.Captcha
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

	common.OkWithDetailed(gin.H{
		"captchaKey":    id,
		"captchaBase64": b64s,
	}, "验证码获取成功", c)
}

// Login User login structure
type Login struct {
	Username    string `json:"username"`    // 用户名
	Password    string `json:"password"`    // 密码
	CaptchaCode string `json:"captchaCode"` // 验证码
	CaptchaKey  string `json:"captchaKey"`  // 验证码ID
}

func (a *AuthApi) Login(c *gin.Context) {
	var loginReq Login
	if err := c.ShouldBindJSON(&loginReq); err != nil {
		common.FailWithMessage(err.Error(), c)
	}

	c.JSON(http.StatusOK, gin.H{"code": 0, "message": "ok", "data": gin.H{
		"username":    loginReq.Username,
		"password":    loginReq.Password,
		"captchaCode": loginReq.CaptchaCode,
	}})

	u := &model.SysUser{Username: loginReq.Username, Password: loginReq.Password}
	user, err := userService.Login(u)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}
	global.LOG.Info(user.Username + "登录成功！")
}
