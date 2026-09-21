package role

import (
	"context"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	repomocks "github.com/earthyzinc/dehaze-go/internal/repository/mocks"
	smocks "github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func newTestService(t *testing.T) (*RoleService, *smocks.MockICache, *repomocks.MockIRoleRepository, *repomocks.MockIMenuRepository) {
	t.Helper()
	cache := smocks.NewMockICache(t)
	repo := repomocks.NewMockIRoleRepository(t)
	menuRepo := repomocks.NewMockIMenuRepository(t)
	svc := NewRoleService(cache, repo, menuRepo, nil)
	return svc, cache, repo, menuRepo
}

func TestGetOptions_HidesBuiltinRolesForNonRoot(t *testing.T) {
	svc, cache, repo, _ := newTestService(t)

	options := []read.Option{
		{Value: 1, Label: "超级管理员", Code: "ROOT"},
		{Value: 2, Label: "管理员", Code: "ADMIN"},
		{Value: 3, Label: "访客", Code: "GUEST"},
	}
	cache.EXPECT().Get(mock.Anything, ROLE_OPTIONS_CACHE_KEY).Return("", errors.New("key not found")).Once()
	repo.EXPECT().FindOptions(mock.Anything).Return(options, nil).Once()
	cache.EXPECT().Set(mock.Anything, ROLE_OPTIONS_CACHE_KEY, mock.Anything, ROLE_OPTIONS_CACHE_TTL).Return(nil).Once()

	// 非 root 隐藏内置角色 ROOT/ADMIN（与 Python 口径一致）
	guest, err := svc.GetOptions(context.Background(), false)
	assert.NoError(t, err)
	assert.Len(t, guest, 1)
	assert.Equal(t, int64(3), guest[0].Value)
	assert.Equal(t, "访客", guest[0].Label)

	// root 可见全部（命中缓存）
	cachedJSON := `[{"value":1,"label":"超级管理员","code":"ROOT"},{"value":2,"label":"管理员","code":"ADMIN"},{"value":3,"label":"访客","code":"GUEST"}]`
	cache.EXPECT().Get(mock.Anything, ROLE_OPTIONS_CACHE_KEY).Return(cachedJSON, nil).Once()
	all, err := svc.GetOptions(context.Background(), true)
	assert.NoError(t, err)
	assert.Len(t, all, 3)
}

func TestUpdateStatus_DisableRefreshesCaches(t *testing.T) {
	svc, cache, repo, menuRepo := newTestService(t)

	role := &model.SysRole{Code: "TEST", Status: 1}
	repo.EXPECT().FindByID(mock.Anything, int64(1)).Return(role, nil).Once()
	repo.EXPECT().UpdateStatus(mock.Anything, int64(1), int8(0)).Return(nil).Once()
	menuRepo.EXPECT().FindPermsByRoleCode(mock.Anything, "TEST").Return([]string{"sys:user:list"}, nil).Once()
	cache.EXPECT().Delete(mock.Anything, "role:perms:TEST").Return(nil).Once()
	cache.EXPECT().Set(mock.Anything, "role:perms:TEST", mock.Anything, ROLE_PERMS_TTL).Return(nil).Once()
	cache.EXPECT().Delete(mock.Anything, ROLE_OPTIONS_CACHE_KEY).Return(nil).Once()

	assert.NoError(t, svc.UpdateStatus(context.Background(), 1, 0))
}

func TestUpdateStatus_InvalidValueRejected(t *testing.T) {
	svc, _, _, _ := newTestService(t)
	err := svc.UpdateStatus(context.Background(), 1, 2)
	assert.Error(t, err)
}
