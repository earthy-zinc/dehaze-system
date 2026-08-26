package auth

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// 以下用例注入 MockICache，聚焦边界/异常/状态转换分支；验证码成功依赖全局 miniredis 验证码 store。

func TestLogin_EmptyUsername_Boundary(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "10.0.0.31"
	captchaKey := "ck-empty"
	captchaCode := "1357"
	setCaptcha(t, captchaKey, captchaCode)

	// 空用户名归一化后仍为 ""，checkLoginFailCount 不查 userKey，仅查 IP 键
	f.cache.EXPECT().Get(mock.Anything, "login:fail:ip:"+clientIP).Return("", errors.New("nf")).Once()
	f.userService.EXPECT().Login(ctx, mock.MatchedBy(func(u *model.SysUser) bool {
		return u.Username == "" && u.Password == "Secret@123"
	})).Return(nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")).Once()
	f.cache.EXPECT().Incr(mock.Anything, "login:fail:ip:"+clientIP).Return(int64(1), nil).Once()
	f.cache.EXPECT().Expire(mock.Anything, "login:fail:ip:"+clientIP, mock.Anything).Return(true, nil).Once()

	req := &bo.LoginRequest{
		Username:    "   ", // 仅空白，归一化为空串
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := f.authService.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_NOT_EXIST)
	// 空用户名不应写入 userKey 失败计数
	f.cache.AssertNotCalled(t, "Incr", mock.Anything, "login:fail:", mock.Anything)
}

func TestLogin_SuperLongUsername_Boundary(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "10.0.0.33"
	captchaKey := "ck-long"
	captchaCode := "2468"
	setCaptcha(t, captchaKey, captchaCode)

	longName := strings.Repeat("a", 200)
	f.cache.EXPECT().Get(mock.Anything, "login:fail:ip:"+clientIP).Return("", errors.New("nf")).Once()
	f.cache.EXPECT().Get(mock.Anything, "login:fail:"+longName).Return("", errors.New("nf")).Once()
	f.userService.EXPECT().Login(ctx, mock.MatchedBy(func(u *model.SysUser) bool {
		return u.Username == longName
	})).Return(nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")).Once()
	f.cache.EXPECT().Incr(mock.Anything, "login:fail:ip:"+clientIP).Return(int64(1), nil).Once()
	f.cache.EXPECT().Expire(mock.Anything, "login:fail:ip:"+clientIP, mock.Anything).Return(true, nil).Once()
	f.cache.EXPECT().Incr(mock.Anything, "login:fail:"+longName).Return(int64(1), nil).Once()
	f.cache.EXPECT().Expire(mock.Anything, "login:fail:"+longName, mock.Anything).Return(true, nil).Once()

	req := &bo.LoginRequest{
		Username:    longName,
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := f.authService.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_NOT_EXIST)
}

func TestLogin_CaseInsensitiveNormalization(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "10.0.0.35"
	captchaKey := "ck-case"
	captchaCode := "3691"
	setCaptcha(t, captchaKey, captchaCode)

	f.cache.EXPECT().Get(mock.Anything, "login:fail:ip:"+clientIP).Return("", errors.New("nf")).Once()
	f.cache.EXPECT().Get(mock.Anything, "login:fail:zhangsan").Return("", errors.New("nf")).Once()
	var gotUsername string
	f.userService.EXPECT().Login(ctx, mock.MatchedBy(func(u *model.SysUser) bool {
		gotUsername = u.Username
		return true
	})).Return(newTestUser(), nil).Once()
	f.cache.EXPECT().Set(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
	f.cache.EXPECT().Delete(mock.Anything, "login:fail:ip:"+clientIP).Return(nil).Once()
	f.cache.EXPECT().Delete(mock.Anything, "login:fail:zhangsan").Return(nil).Once()

	req := &bo.LoginRequest{
		Username:    "ZhangSan", // 大小写混合
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := f.authService.Login(ctx, req, clientIP, "")

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "zhangsan", gotUsername, "用户名应被归一化为小写")
}

func TestLogin_DependencyError_Propagates(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "10.0.0.37"
	captchaKey := "ck-dep"
	captchaCode := "4815"
	setCaptcha(t, captchaKey, captchaCode)

	f.cache.EXPECT().Get(mock.Anything, "login:fail:ip:"+clientIP).Return("", errors.New("nf")).Once()
	f.cache.EXPECT().Get(mock.Anything, "login:fail:zhangsan").Return("", errors.New("nf")).Once()
	// 依赖返回系统级错误（非用户不存在），应原样透传并递增失败计数
	depErr := common.NewBizError(common.SYSTEM_EXECUTION_ERROR, "数据库异常")
	f.userService.EXPECT().Login(ctx, mock.Anything).Return(nil, depErr).Once()
	f.cache.EXPECT().Incr(mock.Anything, mock.Anything).Return(int64(1), nil).Twice()
	f.cache.EXPECT().Expire(mock.Anything, mock.Anything, mock.Anything).Return(true, nil).Times(2)

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := f.authService.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.SYSTEM_EXECUTION_ERROR)
}

func TestLogin_ConsecutiveFailures_ReachesLockThreshold(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "10.0.0.39"
	captchaKey := "ck-thresh"
	captchaCode := "5102"

	// 前 5 次：IP/用户名失败计数均未达阈值（Get 返回未找到），每次写入失败计数
	f.cache.EXPECT().Get(mock.Anything, "login:fail:ip:"+clientIP).Return("", errors.New("nf")).Times(5)
	f.cache.EXPECT().Get(mock.Anything, "login:fail:zhangsan").Return("", errors.New("nf")).Times(5)
	f.userService.EXPECT().Login(ctx, mock.Anything).Return(nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")).Times(5)
	f.cache.EXPECT().Incr(mock.Anything, "login:fail:ip:"+clientIP).Return(int64(1), nil).Times(5)
	f.cache.EXPECT().Expire(mock.Anything, "login:fail:ip:"+clientIP, mock.Anything).Return(true, nil).Times(5)
	f.cache.EXPECT().Incr(mock.Anything, "login:fail:zhangsan").Return(int64(1), nil).Times(5)
	f.cache.EXPECT().Expire(mock.Anything, "login:fail:zhangsan", mock.Anything).Return(true, nil).Times(5)
	// 第 6 次：IP 失败计数已达阈值（Get 返回 5），直接锁定
	f.cache.EXPECT().Get(mock.Anything, "login:fail:ip:"+clientIP).Return("5", nil).Once()

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}

	for i := 0; i < 5; i++ {
		// 验证码验证后会被清除，每次失败重试前重新写入有效验证码
		setCaptcha(t, captchaKey, captchaCode)
		r, e := f.authService.Login(ctx, req, clientIP, "")
		assert.Nil(t, r)
		assertBizError(t, e, common.USER_NOT_EXIST)
	}

	// 第 6 次触发锁定（状态转换：计数达阈值）
	r, e := f.authService.Login(ctx, req, clientIP, "")
	assert.Nil(t, r)
	assertBizError(t, e, common.PASSWORD_ENTER_EXCEED_LIMIT)
	assert.Contains(t, e.Error(), "IP已临时锁定")
}

func TestLogout_WithSession(t *testing.T) {
	f := setupTest(t)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/api/v1/auth/logout", nil)
	sessionID := "sess-logout-001"
	c.Request.AddCookie(&http.Cookie{Name: middleware.SessionCookieName, Value: sessionID})

	// 断言真实删除 session 与用户状态缓存
	f.cache.EXPECT().Delete(mock.Anything, common.SessionPrefix+sessionID).Return(nil).Once()

	err := f.authService.Logout(c)

	assert.NoError(t, err)
}
