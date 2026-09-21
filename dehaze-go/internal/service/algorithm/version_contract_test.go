package algorithm

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/repository/mocks"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func requireBizCode(t *testing.T, err error, want string) {
	t.Helper()
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "期望业务错误信封，实际: %v", err)
	require.Equal(t, want, bizErr.Code().Code)
}

// TestCreateVersionFormatValidation 版本号格式必须为 vX.Y.Z（A0400）：
// python `AlgorithmVersionForm.validate_version` 用 `^v\d+\.\d+\.\d+$` 校验，
// go 曾直接接受 `not-semver-*` 并落库（SDK algorithm.test.ts「边界：版本号格式非法应失败」）。
func TestCreateVersionFormatValidation(t *testing.T) {
	repo := mocks.NewMockIAlgorithmRepository(t)
	svc := NewAlgorithmService(repo, nil, nil)

	for _, version := range []string{"not-semver-1", "2.0.1", "v2.0", "v2.0.1-beta"} {
		_, err := svc.CreateVersion(context.Background(), 812, &bo.AlgorithmVersionForm{Version: version})
		requireBizCode(t, err, "A0400")
	}
	// 格式校验先于任何仓储调用（python 在表单层拦截）
	repo.AssertNotCalled(t, "FindByID", mock.Anything, mock.Anything)
}

// TestCreateVersionDuplicateIsB0001 版本号重复取 python 默认码 B0001：
// python 该分支为 `BusinessException("版本号 {v} 已存在")`（仅传消息）→
// core/exceptions.py 规则取 SYSTEM_EXECUTION_ERROR(B0001)，go 曾返回 A0502。
func TestCreateVersionDuplicateIsB0001(t *testing.T) {
	repo := mocks.NewMockIAlgorithmRepository(t)
	svc := NewAlgorithmService(repo, nil, nil)

	repo.On("FindByID", mock.Anything, int64(812)).Return(&model.SysAlgorithm{}, nil)
	repo.On("ExistsByVersion", mock.Anything, int64(812), "v2.0.90619").Return(true, nil)

	_, err := svc.CreateVersion(context.Background(), 812, &bo.AlgorithmVersionForm{Version: "v2.0.90619"})
	requireBizCode(t, err, "B0001")
	require.Contains(t, err.Error(), "已存在")
}
