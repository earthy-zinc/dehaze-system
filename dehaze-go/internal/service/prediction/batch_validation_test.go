package prediction

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
)

// TestBatchPredictRejectsEmptyItems 空 items 必须参数校验失败（A0400）：
// python `prediction_service.batch_predict` 首行即 `if not items: raise BusinessException(PARAM_ERROR)`，
// go 曾"成功"返回 {total:0, results:[]}（SDK model.test.ts「边界：空items应失败」）。
func TestBatchPredictRejectsEmptyItems(t *testing.T) {
	svc := &PredictionService{}
	results, err := svc.BatchPredict(context.Background(), 13, nil, 2, nil)
	require.Nil(t, results)
	requireBizCode(t, err, common.PARAM_ERROR)
	require.Contains(t, err.Error(), "批量处理图片列表不能为空")
}
