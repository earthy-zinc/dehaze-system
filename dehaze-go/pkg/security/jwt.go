package security

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

type JWT struct {
	Key []byte
	TTL time.Duration
}

var (
	ErrTokenValid            = errors.New("未知错误")
	ErrTokenExpired          = errors.New("token已过期")
	ErrTokenNotValidYet      = errors.New("token尚未激活")
	ErrTokenMalformed        = errors.New("这不是一个token")
	ErrTokenSignatureInvalid = errors.New("无效签名")
	ErrTokenInvalid          = errors.New("无法处理此token")
)

// NewJWT 便捷构造函数，从全局配置读取（生产代码使用）
func NewJWT() *JWT {
	cfg := config.GetConfig()
	return NewJWTWithConfig([]byte(cfg.JWT.Key), time.Duration(cfg.JWT.TTL)*time.Second)
}

// NewJWTWithConfig 可注入配置的构造函数（测试友好）
func NewJWTWithConfig(key []byte, ttl time.Duration) *JWT {
	return &JWT{Key: key, TTL: ttl}
}

func (j *JWT) CreateClaims(authInfo *model.UserAuthInfo) CustomClaims {
	// 合并角色（带 ROLE_ 前缀）和权限到 authorities
	var authorities []string

	// 添加角色（带 ROLE_ 前缀，用于 IsRoot 等角色判断）
	for _, role := range authInfo.Roles {
		authorities = append(authorities, "ROLE_"+role)
	}

	// 添加实际权限（用于权限中间件校验）
	authorities = append(authorities, authInfo.Perms...)

	claims := CustomClaims{
		UserID:      authInfo.UserId,
		DeptID:      authInfo.DeptId,
		DataScope:   authInfo.DataScope,
		Authorities: authorities,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(j.TTL)),
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

// SetJWT 便捷包级函数，从全局配置读取（生产代码使用）
func SetJWT(ctx context.Context, token string, userName string) error {
	j := NewJWT()
	return j.SetToken(ctx, token, userName)
}

// SetToken 将Token存入缓存，TTL来自实例字段（测试友好）
func (j *JWT) SetToken(ctx context.Context, token string, userName string) error {
	cacheClient := cache.GetCache()
	return cacheClient.Set(ctx, userName, token, j.TTL)
}
