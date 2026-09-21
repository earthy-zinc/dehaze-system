package preset

import (
	"context"
	"fmt"
	"testing"

	presetrepo "github.com/earthyzinc/dehaze-go/internal/repository/preset"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
)

func requireBizCode(t *testing.T, err error, want string) {
	t.Helper()
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "期望业务错误信封，实际: %v", err)
	require.Equal(t, want, bizErr.Code().Code)
}

// TestCreatePresetDuplicateNameIsA0501 同名预设 → A0501：
// python `preset_service.create_preset` 捕获唯一键 uk_user_name 冲突后
// `raise BusinessException(DATA_EXISTS, "预设名称已存在")`，go 曾把它暴露成 C0300（SDK model.test.ts:255）。
func TestCreatePresetDuplicateNameIsA0501(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewPresetService(db, presetrepo.NewPresetRepository(db))
	ctx := context.Background()

	form := &PresetForm{Name: "conflict_regression", AlgorithmID: 1, Params: []byte(`{"gamma":1}`)}
	_, err := svc.CreatePreset(ctx, 960011, form)
	require.NoError(t, err)

	_, err = svc.CreatePreset(ctx, 960011, form)
	requireBizCode(t, err, "A0501")
	require.Contains(t, err.Error(), "预设名称已存在")
}

// TestCreatePresetHasNoCountLimit 自定义预设无数量上限：
// python create_preset 无任何计数校验、schema/python 均无 preset_limit 列，
// go 曾按会员等级硬编码 3/10/20，第 4 条起误报 A0503（SDK model.test.ts:273）。
func TestCreatePresetHasNoCountLimit(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewPresetService(db, presetrepo.NewPresetRepository(db))
	ctx := context.Background()

	for i := 1; i <= 6; i++ {
		_, err := svc.CreatePreset(ctx, 960012, &PresetForm{
			Name:        fmt.Sprintf("no_limit_%d", i),
			AlgorithmID: 1,
			Params:      []byte(`{"gamma":1}`),
		})
		require.NoError(t, err, "第 %d 条自定义预设不应被数量上限拦截", i)
	}
}
