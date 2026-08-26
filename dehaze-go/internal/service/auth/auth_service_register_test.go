package auth

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// newRegisterTestService 构造 Register 成功路径所需 AuthService：
// 缓存用 MockICache 注入（session 写入与失败计数重置走 mock），验证码校验走全局 miniredis 验证码 store（既有 initCaptchaBackend/setCaptcha helper）。
func newRegisterTestService(t *testing.T) (*testFixture, *mocks.MockIMemberService) {
	t.Helper()
	cache := mocks.NewMockICache(t)
	userService := mocks.NewMockIUserService(t)
	memberService := mocks.NewMockIMemberService(t)
	svc := NewAuthService(cache, userService, nil, memberService).(*AuthService)
	return &testFixture{cache: cache, userService: userService, authService: svc}, memberService
}

func TestRegister_Success(t *testing.T) {
	f, memberSvc := newRegisterTestService(t)
	ctx := context.Background()

	captchaKey := "ck-register-success"
	captchaCode := "1234"
	setCaptcha(t, captchaKey, captchaCode)

	registeredUser := &model.SysUser{BaseModel: model.BaseModel{ID: 2001}, Username: "testuser", Nickname: "测试用户"}
	// 用户名归一化：service 内部 ToLower+TrimSpace 后以归一化值调用 Register
	f.userService.EXPECT().Register(ctx, "testuser", "测试用户", "Secret@123").
		Return(registeredUser, int8(1), nil).Once()
	memberSvc.EXPECT().InitDefaultMember(ctx, int64(2001)).Return(nil).Once()
	// 注册成功写 session 到注入的 MockICache；resetLoginFailCount 触发 Delete（ip+user 两次）
	f.cache.EXPECT().Set(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil)
	f.cache.EXPECT().Delete(mock.Anything, mock.Anything).Return(nil)

	req := &bo.RegisterRequest{
		Username:    "  TestUser  ",
		Nickname:    "测试用户",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := f.authService.Register(ctx, req, "10.0.0.20")

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotEmpty(t, result.SessionID)
	assert.Equal(t, int64(2001), result.User.ID)
	// 返回的用户名应为归一化后的小写
	assert.Equal(t, "testuser", result.User.Username)
	assert.Equal(t, "测试用户", result.User.Nickname)
}

func TestRegister_UsernameNormalization_PassedToUserService(t *testing.T) {
	f, memberSvc := newRegisterTestService(t)
	ctx := context.Background()

	captchaKey := "ck-register-norm"
	captchaCode := "5678"
	setCaptcha(t, captchaKey, captchaCode)

	registeredUser := &model.SysUser{BaseModel: model.BaseModel{ID: 2002}, Username: "mixedcase", Nickname: "昵称"}
	// 断言归一化后的值确实传入 UserService.Register，且大小写/空格被清除
	f.userService.EXPECT().Register(ctx, "mixedcase", "昵称", "Secret@123").
		Return(registeredUser, int8(0), nil).Once()
	memberSvc.EXPECT().InitDefaultMember(ctx, int64(2002)).Return(nil).Once()
	f.cache.EXPECT().Set(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil)
	f.cache.EXPECT().Delete(mock.Anything, mock.Anything).Return(nil)

	req := &bo.RegisterRequest{
		Username:    "  MixedCase  ",
		Nickname:    " 昵称 ",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	_, err := f.authService.Register(ctx, req, "10.0.0.21")

	assert.NoError(t, err)
}

func TestRegister_CaptchaWrong_ReturnsError(t *testing.T) {
	f, memberSvc := newRegisterTestService(t)
	ctx := context.Background()

	captchaKey := "ck-register-fail"
	// 不预置验证码，VerifyCaptcha 必然失败
	req := &bo.RegisterRequest{
		Username:    "newuser",
		Nickname:    "新用户",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: "0000",
	}
	result, err := f.authService.Register(ctx, req, "10.0.0.22")

	assert.Nil(t, result)
	assertBizError(t, err, common.VERIFY_CODE_ERROR)
	// 验证码错误时不创建用户、不初始化会员
	f.userService.AssertNotCalled(t, "Register", mock.Anything, mock.Anything, mock.Anything, mock.Anything)
	memberSvc.AssertNotCalled(t, "InitDefaultMember", mock.Anything, mock.Anything)
}

func TestRegister_UserServiceError_Propagates(t *testing.T) {
	f, memberSvc := newRegisterTestService(t)
	ctx := context.Background()

	captchaKey := "ck-register-usererr"
	captchaCode := "4321"
	setCaptcha(t, captchaKey, captchaCode)

	f.userService.EXPECT().Register(ctx, "dupuser", "用户", "Secret@123").
		Return(nil, int8(0), common.NewBizError(common.DATA_EXISTS, "用户名已被注册")).Once()

	req := &bo.RegisterRequest{
		Username:    "dupuser",
		Nickname:    "用户",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := f.authService.Register(ctx, req, "10.0.0.23")

	assert.Nil(t, result)
	assertBizError(t, err, common.DATA_EXISTS)
	// 用户创建失败后不初始化会员
	memberSvc.AssertNotCalled(t, "InitDefaultMember", mock.Anything, mock.Anything)
}
