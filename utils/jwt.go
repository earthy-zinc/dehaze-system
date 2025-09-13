package utils

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	jwt "github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

type JWT struct {
	Key []byte
}

var (
	ErrTokenValid            = errors.New("未知错误")
	ErrTokenExpired          = errors.New("token已过期")
	ErrTokenNotValidYet      = errors.New("token尚未激活")
	ErrTokenMalformed        = errors.New("这不是一个token")
	ErrTokenSignatureInvalid = errors.New("无效签名")
	ErrTokenInvalid          = errors.New("无法处理此token")
)

func NewJWT() *JWT {
	return &JWT{
		[]byte(global.CONFIG.JWT.Key),
	}
}

func (j *JWT) CreateClaims(authInfo model.UserAuthInfo) CustomClaims {
	// 获取角色信息
	roles := authInfo.Roles

	// 处理角色信息
	var authorities []string
	if len(roles) > 0 {
		authorities = make([]string, len(roles))
		for i, role := range roles {
			authorities[i] = "ROLE_" + role
		}
	} else {
		authorities = []string{} // 空切片表示无角色
	}

	claims := CustomClaims{
		UserID:      authInfo.UserId,
		DeptID:      authInfo.DeptId,
		DataScope:   authInfo.DataScope,
		Authorities: authorities,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Duration(global.CONFIG.JWT.TTL))), // 过期时间 7天  配置文件
			Subject:   authInfo.Username,
			ID:        uuid.New().String(),
		},
	}
	return claims
}

// CreateToken 创建一个token
func (j *JWT) CreateToken(claims CustomClaims) (string, error) {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(j.Key)
}

// ParseToken 解析 token
func (j *JWT) ParseToken(tokenString string) (*CustomClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &CustomClaims{}, func(token *jwt.Token) (i interface{}, e error) {
		return j.Key, nil
	})

	if err != nil {
		switch {
		case errors.Is(err, jwt.ErrTokenExpired):
			return nil, ErrTokenExpired
		case errors.Is(err, jwt.ErrTokenMalformed):
			return nil, ErrTokenMalformed
		case errors.Is(err, jwt.ErrTokenSignatureInvalid):
			return nil, ErrTokenSignatureInvalid
		case errors.Is(err, jwt.ErrTokenNotValidYet):
			return nil, ErrTokenNotValidYet
		default:
			return nil, ErrTokenInvalid
		}
	}
	if token != nil {
		if claims, ok := token.Claims.(*CustomClaims); ok && token.Valid {
			return claims, nil
		}
	}
	return nil, ErrTokenValid
}

//@author: [piexlmax](https://github.com/piexlmax)
//@function: SetRedisJWT
//@description: jwt存入redis并设置过期时间
//@param: jwt string, userName string
//@return: err error

func SetRedisJWT(jwt string, userName string) (err error) {
	// 此处过期时间等于jwt过期时间
	timer := time.Duration(global.CONFIG.JWT.TTL)
	err = global.REDIS.Set(context.Background(), userName, jwt, timer).Err()
	return err
}
