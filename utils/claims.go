package utils

import (
	"net"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

// 自定义声明结构体
type CustomClaims struct {
	UserID      int64    `json:"userId"`
	DeptID      int64    `json:"deptId"`
	DataScope   int      `json:"dataScope"`
	Authorities []string `json:"authorities"`
	jwt.RegisteredClaims
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
	if token == "" {
		j := NewJWT()
		token, _ = c.Cookie("Authorization")
		claims, err := j.ParseToken(token)
		if err != nil {
			global.LOG.Error("重新写入cookie token失败,未能成功解析token,请检查请求头是否存在Authorization且claims是否为规定结构")
			return token
		}
		SetToken(c, token, int((claims.ExpiresAt.Unix()-time.Now().Unix())/60))
	}
	return token
}

func GetClaims(c *gin.Context) (*CustomClaims, error) {
	token := GetToken(c)
	j := NewJWT()
	claims, err := j.ParseToken(token)
	if err != nil {
		global.LOG.Error("从Gin的Context中获取从jwt解析信息失败, 请检查请求头是否存在Authorization且claims是否为规定结构")
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

func LoginToken(user *model.SysUser) (token string, claims CustomClaims, err error) {
	userAuthInfo := model.UserAuthInfo{}

	j := NewJWT()
	claims = j.CreateClaims(userAuthInfo)
	token, err = j.CreateToken(claims)
	return
}
