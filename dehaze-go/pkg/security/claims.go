package security

import (
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

type CustomClaims struct {
	UserID      int64    `json:"userId"`
	DeptID      int64    `json:"deptId"`
	DataScope   int8     `json:"dataScope"`
	Authorities []string `json:"authorities"`
	jwt.RegisteredClaims
}

func (c *CustomClaims) GetUserID() int64 {
	return c.UserID
}

func (c *CustomClaims) GetDeptID() int64 {
	return c.DeptID
}

func (c *CustomClaims) GetDataScope() int8 {
	return c.DataScope
}

func CreateClaims(authInfo *model.UserAuthInfo) CustomClaims {
	var authorities []string
	for _, role := range authInfo.Roles {
		authorities = append(authorities, "ROLE_"+role)
	}
	authorities = append(authorities, authInfo.Perms...)
	claims := CustomClaims{
		UserID:      authInfo.UserId,
		DeptID:      authInfo.DeptId,
		DataScope:   authInfo.DataScope,
		Authorities: authorities,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(7 * 24 * time.Hour)),
			Subject:   authInfo.Username,
			ID:        uuid.New().String(),
		},
	}
	return claims
}

func GetClaims(c *gin.Context) (*CustomClaims, error) {
	if claims, exists := c.Get("claims"); exists {
		if cl, ok := claims.(*CustomClaims); ok {
			return cl, nil
		}
	}
	return nil, errors.New("未找到认证信息")
}

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

func RequireUserID(c *gin.Context) (int64, error) {
	userID := GetUserID(c)
	if userID == 0 {
		return 0, common.NewBizError(common.ACCESS_UNAUTHORIZED, "访问未授权")
	}
	return userID, nil
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

func IsAdmin(c *gin.Context) bool {
	claims := GetUserInfo(c)
	if claims == nil {
		return false
	}
	for _, authority := range claims.Authorities {
		if authority == "ROLE_ROOT" || authority == "ROLE_ADMIN" {
			return true
		}
	}
	return false
}
