package evaluation

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// algoEnvelope 构造 Python 算法服务的标准信封响应 {"code":...,"msg":...,"data":...}
func algoEnvelope(t *testing.T, data any) string {
	t.Helper()
	raw, err := json.Marshal(data)
	require.NoError(t, err)
	return `{"code":"00000","msg":"一切ok","data":` + string(raw) + `}`
}

// seedEvalImageModel 播种一条可用图像模型（供评估入口的模型存在性校验）
func seedEvalImageModel(t *testing.T, db *gorm.DB, modelID string) {
	t.Helper()
	provider := model.SysAiProvider{ProviderCode: "local-modal-" + modelID, DisplayName: "内置图像引擎", Status: 1}
	require.NoError(t, db.Create(&provider).Error)
	require.NoError(t, db.Create(&model.SysAiModel{
		ProviderID: provider.ID, ModelID: modelID, ModelType: "image_edit", DisplayName: modelID, Status: 1,
	}).Error)
}

func newEvalService(t *testing.T, db *gorm.DB, url string) *EvaluationService {
	t.Helper()
	client, err := algorithm.NewClient(options.Algorithm{ServiceURL: url, MaxRetry: 0, Timeout: 5})
	require.NoError(t, err)
	return NewEvaluationService(evalrepo.NewEvalLogRepository(db), airepo.NewModelRepository(db), client)
}

// TestExecuteAsync_WritesContentMd5FromAlgorithmResponse 评估完成时：
// pred_md5/gt_md5 取算法服务返回的图片内容 MD5，不得用 URL 的 MD5 冒充。
func TestExecuteAsync_WritesContentMd5FromAlgorithmResponse(t *testing.T) {
	db := testutil.NewTestDB(t)
	userID := int64(900401)
	ctx := context.Background()

	const predMd5 = "d41d8cd98f00b204e9800998ecf8427e"
	const gtMd5 = "0cc175b9c0f1b6a831c399e269772661"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(algoEnvelope(t, map[string]any{
			"logId": 1, "status": 2,
			"metrics": map[string]float64{"psnr": 28.5},
			"predMd5": predMd5, "gtMd5": gtMd5, "time": 3,
		})))
	}))
	t.Cleanup(srv.Close)

	repo := evalrepo.NewEvalLogRepository(db)
	svc := newEvalService(t, db, srv.URL)

	predURL := "http://cdn.test/pred.png"
	gtURL := "http://cdn.test/gt.png"
	log := &model.SysEvalLog{
		BaseModel: model.BaseModel{CreateBy: userID},
		ModelID:   "aecrnet",
		PredURL:   predURL,
		GtURL:     gtURL,
		Status:    model.LogStatusProcessing,
		TaskType:  "evaluation",
	}
	require.NoError(t, repo.Create(ctx, log))

	svc.executeAsync(log.ID, "aecrnet", predURL, gtURL)

	var saved model.SysEvalLog
	require.NoError(t, db.First(&saved, log.ID).Error)
	assert.Equal(t, model.LogStatusCompleted, saved.Status)
	assert.Equal(t, predMd5, saved.PredMD5, "pred_md5 应为算法服务返回的预测图内容 MD5")
	assert.Equal(t, gtMd5, saved.GtMD5, "gt_md5 应为算法服务返回的参考图内容 MD5")
	assert.NotEqual(t, utils.MD5Hex(predURL), saved.PredMD5, "不得用预测图 URL 的 MD5 冒充")
	assert.NotEqual(t, utils.MD5Hex(gtURL), saved.GtMD5, "不得用参考图 URL 的 MD5 冒充")
}

// TestEvaluate_ModelNotFound 评估入口模型不存在 → A0401（对齐 python `prediction_service.get_model`）。
func TestEvaluate_ModelNotFound(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := newEvalService(t, db, "http://127.0.0.1:1")

	_, err := svc.Evaluate(context.Background(), "not-exist", "http://cdn.test/pred.png", "http://cdn.test/gt.png", 900402)
	require.Error(t, err)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际: %v", err)
	require.Equal(t, common.RESOURCE_NOT_FOUND, bizErr.Code())
}
