package compare

import (
	"context"
	"testing"

	"github.com/earthyzinc/clarity-go/internal/model"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	evalrepo "github.com/earthyzinc/clarity-go/internal/repository/eval_log"
	predrepo "github.com/earthyzinc/clarity-go/internal/repository/pred_log"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestGenerateReport_ReportTaskKeepsPredGtSemantics 报告任务的 pred/gt 语义：
// pred=处理结果图、gt=原图（与 python 实现及效果对比后端实现文档一致，不得写反）。
func TestGenerateReport_ReportTaskKeepsPredGtSemantics(t *testing.T) {
	// 报告生成会另起 goroutine 写库，用池化句柄（各自独立连接）而非单事务句柄
	db := testutil.NewPoolTestDB(t)
	userID := int64(900301)
	ctx := context.Background()

	predLog := &model.SysPredLog{
		BaseModel: model.BaseModel{CreateBy: userID},
		ModelID:   "aecrnet",
		OriginMD5: "0cc175b9c0f1b6a831c399e269772661",
		OriginURL: "http://cdn.test/origin.png",
		PredMD5:   "d41d8cd98f00b204e9800998ecf8427e",
		PredURL:   "http://cdn.test/result.png",
		Status:    model.LogStatusCompleted,
	}
	require.NoError(t, predrepo.NewPredLogRepository(db).Create(ctx, predLog))
	t.Cleanup(func() { db.Exec("DELETE FROM sys_pred_log WHERE id = ?", predLog.ID) })

	svc := NewCompareService(
		evalrepo.NewEvalLogRepository(db),
		predrepo.NewPredLogRepository(db),
		airepo.NewModelRepository(db),
	)
	vo, err := svc.GenerateReport(ctx, userID, &CompareReportForm{LogID: predLog.ID})
	require.NoError(t, err)
	t.Cleanup(func() { db.Exec("DELETE FROM sys_eval_log WHERE id = ?", vo.TaskID) })

	var task model.SysEvalLog
	require.NoError(t, db.First(&task, vo.TaskID).Error)
	assert.Equal(t, "report", task.TaskType)
	assert.Equal(t, predLog.PredURL, task.PredURL, "pred_url 应为处理结果图")
	assert.Equal(t, predLog.OriginURL, task.GtURL, "gt_url 应为原图")
	assert.Equal(t, "aecrnet", task.ModelID, "报告任务应引用图像模型标识")
}
