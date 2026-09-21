package user

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/repository/mocks"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	mysql "github.com/go-sql-driver/mysql"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// TestIsDuplicateKeyError 唯一键冲突识别（注册并发兜底）：
// "先查后插"存在并发窗口，数据库唯一键是最终防线；该场景 python 捕获 IntegrityError → DATA_EXISTS(A0501)，
// go 必须用同一业务码返回（SDK auth-security.test.ts:182「并发注册仅一个成功」即此断言）。
func TestIsDuplicateKeyError(t *testing.T) {
	require.True(t, isDuplicateKeyError(&mysql.MySQLError{Number: 1062, Message: "Duplicate entry 'x' for key 'uk_username'"}))
	require.False(t, isDuplicateKeyError(&mysql.MySQLError{Number: 1054, Message: "Unknown column"}))
	require.False(t, isDuplicateKeyError(nil))
}

// TestRegisterDuplicateKeyRaceIsA0501 注册落库撞唯一键必须返回 A0501：
// 并发注册时两个请求都能穿过 ExistsByUsername 预检，后者由数据库唯一键兜底，
// 这条兜底路径曾经被包成 B0001「创建用户失败」（run6 auth-security.test.ts:182 实测），
// 与 python 捕获 IntegrityError → DATA_EXISTS 的口径不符。
func TestRegisterDuplicateKeyRaceIsA0501(t *testing.T) {
	userRepo := mocks.NewMockIUserRepository(t)
	roleRepo := mocks.NewMockIRoleRepository(t)
	svc := NewUserService(userRepo, roleRepo, nil, nil, nil, nil)

	userRepo.On("ExistsByUsername", mock.Anything, "race_user").Return(false, nil)
	roleRepo.On("FindByCode", mock.Anything, "GUEST").
		Return(&model.SysRole{BaseModel: model.BaseModel{ID: 9}, Status: 1}, nil)
	userRepo.On("CreateWithRoles", mock.Anything, mock.Anything, mock.Anything).
		Return(&mysql.MySQLError{Number: 1062, Message: "Duplicate entry 'race_user' for key 'uk_username'"})

	_, _, err := svc.Register(context.Background(), "race_user", "并发注册", "Pw123456")
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "期望业务错误信封，实际: %v", err)
	require.Equal(t, "A0501", bizErr.Code().Code)
}
