package security

import (
	"net"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

// CustomClaims 自定义声明结构体
// 实现 UserClaims 接口，支持GORM自动填充回调
type CustomClaims struct {
	UserID      int64    `json:"userId"`
	DeptID      int64    `json:"deptId"` // 修复类型为int64以匹配UserAuthInfo
	DataScope   int8     `json:"dataScope"`
	Authorities []string `json:"authorities"`
	jwt.RegisteredClaims
}

// GetUserID 实现 UserClaims 接口
// 用于GORM自动填充回调获取当前用户ID
func (c *CustomClaims) GetUserID() int64 {
	return c.UserID
}

func ClearToken(c *gin.Context) {
	// 增加cookie Authorization 向来源的web添加
	host, _, err := net.SplitHostPort(c.Request.Host)
	if err != nil {
		host = c.Request.Host
	}

	if net.ParseIP(host) != nil {
		c.SetCookie("Authorization", "", -1, "/", "", false, false)
	} else {
		c.SetCookie("Authorization", "", -1, "/", host, false, false)
	}
}

func SetToken(c *gin.Context, token string, maxAge int) {
	// 增加cookie Authorization 向来源的web添加
	host, _, err := net.SplitHostPort(c.Request.Host)
	if err != nil {
		host = c.Request.Host
	}

	if net.ParseIP(host) != nil {
		c.SetCookie("Authorization", token, maxAge, "/", "", false, false)
	} else {
		c.SetCookie("Authorization", token, maxAge, "/", host, false, false)
	}
}

func GetToken(c *gin.Context) string {
	token := c.Request.Header.Get("Authorization")
	if token != "" {
		// 去掉 Bearer 前缀
		if len(token) > 7 && token[:7] == "Bearer " {
			token = token[7:]
		}
		return token
	}
	// 从 Cookie 获取
	j := NewJWT()
	token, _ = c.Cookie("Authorization")
	claims, err := j.ParseToken(token)
	if err != nil {
		logger.Error("重新写入cookie token失败,未能成功解析token,请检查请求头是否存在Authorization且claims是否为规定结构")
		return token
	}
	SetToken(c, token, int((claims.ExpiresAt.Unix()-time.Now().Unix())/60))
	return token
}

func GetClaims(c *gin.Context) (*CustomClaims, error) {
	token := GetToken(c)
	j := NewJWT()
	claims, err := j.ParseToken(token)
	if err != nil {
		logger.Error("从Gin的Context中获取从jwt解析信息失败, 请检查请求头是否存在Authorization且claims是否为规定结构")
	}
	return claims, err
}

// GetUserID 从Gin的Context中获取从jwt解析出来的用户ID
func GetUserID(c *gin.Context) int64 {
	if claims, exists := c.Get("claims"); !exists {
		if cl, err := GetClaims(c); err != nil {
			return 0
		} else {
			return cl.UserID
		}
	} else {
		waitUse := claims.(*CustomClaims)
		return waitUse.UserID
	}
}

func GetUserName(c *gin.Context) string {
	if claims, exists := c.Get("claims"); !exists {
		if cl, err := GetClaims(c); err != nil {
			return ""
		} else {
			return cl.Subject
		}
	} else {
		waitUse := claims.(*CustomClaims)
		return waitUse.Subject
	}
}

// GetUserInfo 从Gin的Context中获取从jwt解析出来的用户角色id
func GetUserInfo(c *gin.Context) *CustomClaims {
	if claims, exists := c.Get("claims"); !exists {
		if cl, err := GetClaims(c); err != nil {
			return nil
		} else {
			return cl
		}
	} else {
		waitUse := claims.(*CustomClaims)
		return waitUse
	}
}

// IsRoot 判断当前登录用户是否为超级管理员（authorities 中包含 ROLE_ROOT）
func IsRoot(c *gin.Context) bool {
	claims := GetUserInfo(c)
	if claims == nil {
		return false
	}
	for _, authority := range claims.Authorities {
		if authority == "ROLE_ROOT" {
			return true
		}
	}
	return false
}

// LoginToken 便捷包级函数（生产代码使用）
func LoginToken(user *model.UserAuthInfo) (token string, claims CustomClaims, err error) {
	j := NewJWT()
	claims = j.CreateClaims(user)
	token, err = j.CreateToken(claims)
	return
}

// LoginTokenWithRefresh 便捷包级函数（生产代码使用），从全局配置读取 RefreshTokenTTL
func LoginTokenWithRefresh(user *model.UserAuthInfo) (accessToken, refreshToken string, accessClaims, refreshClaims CustomClaims, err error) {
	j := NewJWT()
	cfg := config.GetConfig()
	refreshTTL := cfg.JWT.RefreshTokenTTL
	if refreshTTL <= 0 {
		refreshTTL = 7 * 24 * 3600
	}
	return j.LoginTokenWithRefresh(user, time.Duration(refreshTTL)*time.Second)
}

// LoginTokenWithRefresh 实例方法，refreshTTL 由调用方传入（测试友好）
func (j *JWT) LoginTokenWithRefresh(user *model.UserAuthInfo, refreshTTL time.Duration) (accessToken, refreshToken string, accessClaims, refreshClaims CustomClaims, err error) {
	accessClaims = j.CreateClaims(user)
	accessToken, err = j.CreateToken(accessClaims)
	if err != nil {
		return
	}

	refreshClaims = CustomClaims{
		UserID:      user.UserId,
		DeptID:      user.DeptId,
		DataScope:   user.DataScope,
		Authorities: []string{},
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(refreshTTL)),
			Subject:   user.Username,
			ID:        uuid.New().String(),
		},
	}
	refreshToken, err = j.CreateToken(refreshClaims)
	return
}
