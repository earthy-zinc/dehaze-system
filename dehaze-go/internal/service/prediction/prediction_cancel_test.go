package prediction

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

func predLogStatus(t *testing.T, db *gorm.DB, id int64) (model.LogStatus, string) {
	t.Helper()
	var row struct {
		Status       model.LogStatus
		ErrorMessage *string
	}
	require.NoError(t, db.Table("sys_pred_log").Select("status, error_message").
		Where("id = ?", id).Scan(&row).Error)
	errMsg := ""
	if row.ErrorMessage != nil {
		errMsg = *row.ErrorMessage
	}
	return row.Status, errMsg
}

// TestCancelTask 取消图像任务契约（对齐 python `cancel_task`）：
// 仅本人可取消（不存在/非本人 → A0401 防枚举）；仅"处理中"可取消（置已取消）；
// 终态任务幂等返回当前状态。
func TestCancelTask(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	seedImageModel(t, db, "aecrnet")
	svc := newPredictionService(t, db, "http://127.0.0.1:1", nil)

	t.Run("处理中→取消：状态置 4", func(t *testing.T) {
		userID := int64(880001)
		log := createPredLog(t, db, userID, "aecrnet")

		result, err := svc.CancelTask(ctx, log.ID, userID)
		require.NoError(t, err)
		require.EqualValues(t, model.LogStatusCancelled, result.Status)

		status, msg := predLogStatus(t, db, log.ID)
		require.EqualValues(t, model.LogStatusCancelled, status)
		require.Equal(t, "任务已取消", msg)
	})

	t.Run("幂等：已取消任务再次取消返回当前状态", func(t *testing.T) {
		userID := int64(880002)
		log := createPredLog(t, db, userID, "aecrnet")

		_, err := svc.CancelTask(ctx, log.ID, userID)
		require.NoError(t, err)

		result, err := svc.CancelTask(ctx, log.ID, userID)
		require.NoError(t, err)
		require.EqualValues(t, model.LogStatusCancelled, result.Status)
	})

	t.Run("已完成任务：幂等返回当前状态", func(t *testing.T) {
		userID := int64(880003)
		log := createPredLog(t, db, userID, "aecrnet")
		require.NoError(t, db.Table("sys_pred_log").Where("id = ?", log.ID).
			Update("status", model.LogStatusCompleted).Error)

		result, err := svc.CancelTask(ctx, log.ID, userID)
		require.NoError(t, err)
		require.EqualValues(t, model.LogStatusCompleted, result.Status)
	})

	t.Run("非本人任务：A0401", func(t *testing.T) {
		log := createPredLog(t, db, 880004, "aecrnet")
		_, err := svc.CancelTask(ctx, log.ID, 880005)
		requireBizCode(t, err, common.RESOURCE_NOT_FOUND)
	})

	t.Run("任务不存在：A0401", func(t *testing.T) {
		_, err := svc.CancelTask(ctx, 999999999, 880006)
		requireBizCode(t, err, common.RESOURCE_NOT_FOUND)
	})
}

func requireBizCode(t *testing.T, err error, code *common.ResultCode) {
	t.Helper()
	require.Error(t, err)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际: %v", err)
	require.Equal(t, code, bizErr.Code())
}
