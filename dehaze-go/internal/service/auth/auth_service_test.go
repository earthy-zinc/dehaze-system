package auth

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestMain(m *testing.M) {
	config.Config = &config.AppConfig{
		System: options.System{
			UseMultiPoint:     false,
			LoginFailLimit:    5,
			LoginFailLockTime: 300,
		},
	}
	os.Exit(m.Run())
}

type testFixture struct {
	cache       *mocks.MockICache
	userService *mocks.MockIUserService
	authService *AuthService
}

func setupTest(t *testing.T) *testFixture {
	t.Helper()
	mockCache := mocks.NewMockICache(t)
	mockUserService := mocks.NewMockIUserService(t)
	svc := NewAuthService(mockCache, mockUserService, nil, nil).(*AuthService)
	return &testFixture{
		cache:       mockCache,
		userService: mockUserService,
		authService: svc,
	}
}

func assertBizError(t *testing.T, err error, expectedCode *common.ResultCode) {
	t.Helper()
	assert.Error(t, err)
	var bizErr *common.BizError
	assert.True(t, errors.As(err, &bizErr), "期望 BizError 类型，实际: %T", err)
	assert.Equal(t, expectedCode, bizErr.Code(), "错误码不匹配")
}

func newTestUser() *model.UserAuthInfo {
	return &model.UserAuthInfo{
		UserId:    1001,
		Username:  "zhangsan",
		Nickname:  "张三",
		DeptId:    10,
		Status:    1,
		Roles:     []string{"admin"},
		Perms:     []string{"sys:user:list", "sys:role:list"},
		DataScope: 1,
	}
}

func mockCacheGetNotFound(cache *mocks.MockICache, key string) {
	cache.EXPECT().Get(mock.Anything, key).Return("", errors.New("key not found")).Once()
}

func TestLogin_NilRequest(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	result, err := f.authService.Login(ctx, nil, "192.168.1.100", "")

	assert.Nil(t, result)
	assertBizError(t, err, common.PARAM_ERROR)
	assert.Contains(t, err.Error(), "登录请求不能为空")
}

func TestLogin_IPLocked(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "192.168.1.100"

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Admin@2024",
		CaptchaKey:  "key-001",
		CaptchaCode: "123456",
	}

	f.cache.EXPECT().Get(mock.Anything, "login:fail:ip:"+clientIP).Return("5", nil).Once()

	result, err := f.authService.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.PASSWORD_ENTER_EXCEED_LIMIT)
	assert.Contains(t, err.Error(), "IP已临时锁定")
}

func TestLogin_UsernameLocked(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "192.168.1.100"

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Admin@2024",
		CaptchaKey:  "key-001",
		CaptchaCode: "123456",
	}

	mockCacheGetNotFound(f.cache, "login:fail:ip:"+clientIP)
	f.cache.EXPECT().Get(mock.Anything, "login:fail:user:zhangsan").Return("5", nil).Once()

	result, err := f.authService.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.PASSWORD_ENTER_EXCEED_LIMIT)
	assert.Contains(t, err.Error(), "账户已临时锁定")
}

func TestLogin_UsernameNormalization(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "192.168.1.100"

	req := &bo.LoginRequest{
		Username:    "  ZhangSan  ",
		Password:    "Admin@2024",
		CaptchaKey:  "key-001",
		CaptchaCode: "123456",
	}

	mockCacheGetNotFound(f.cache, "login:fail:ip:"+clientIP)
	f.cache.EXPECT().Get(mock.Anything, "login:fail:user:zhangsan").Return("5", nil).Once()

	result, err := f.authService.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.PASSWORD_ENTER_EXCEED_LIMIT)
}

func TestGetAuthInfo_Success(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	expectedVO := &vo.UserInfoVO{
		UserId:   1001,
		Username: "zhangsan",
		Nickname: "张三",
		Avatar:   "https://cdn.example.com/avatar/zhangsan.png",
		Roles:    []string{"admin"},
		Perms:    []string{"sys:user:list", "sys:role:list"},
	}

	f.userService.EXPECT().GetCurrentUserInfo(ctx, int64(1001)).Return(expectedVO, nil).Once()

	result, err := f.authService.GetAuthInfo(ctx, 1001)

	assert.NoError(t, err)
	assert.Equal(t, expectedVO.UserId, result.UserId)
	assert.Equal(t, "zhangsan", result.Username)
	assert.Equal(t, "张三", result.Nickname)
}

func TestGetAuthInfo_UserNotExist(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	f.userService.EXPECT().GetCurrentUserInfo(ctx, int64(9999)).
		Return(nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")).Once()

	result, err := f.authService.GetAuthInfo(ctx, 9999)

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_NOT_EXIST)
}

func TestLogout_NoSession(t *testing.T) {
	f := setupTest(t)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/api/v1/auth/logout", nil)

	err := f.authService.Logout(c)

	assert.NoError(t, err)
}
