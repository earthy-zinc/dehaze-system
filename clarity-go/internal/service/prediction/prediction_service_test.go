package prediction

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/vo"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	predrepo "github.com/earthyzinc/clarity-go/internal/repository/pred_log"
	"github.com/earthyzinc/clarity-go/internal/service/dict"
	servicemocks "github.com/earthyzinc/clarity-go/internal/service/mocks"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/earthyzinc/clarity-go/pkg/algorithm"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/config/options"
	"github.com/earthyzinc/clarity-go/pkg/lifecycle"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// seedImageModel 播种一条可用图像模型（model_type=image_edit、status=1）及其供应商。
// 先清掉同 model_id/provider_code 的种子行，避免 config 种子里的同名模型抢先被解析。
func seedImageModel(t *testing.T, db *gorm.DB, modelID string) (model.SysAiModel, model.SysAiProvider) {
	t.Helper()
	providerCode := "local-modal-" + modelID
	require.NoError(t, db.Unscoped().Where("model_id = ?", modelID).Delete(&model.SysAiModel{}).Error)
	require.NoError(t, db.Unscoped().Where("provider_code = ?", providerCode).Delete(&model.SysAiProvider{}).Error)

	provider := model.SysAiProvider{
		ProviderCode: providerCode,
		DisplayName:  "内置图像引擎",
		ProtocolType: "openai_image",
		Status:       1,
	}
	require.NoError(t, db.Create(&provider).Error)

	row := model.SysAiModel{
		ProviderID:  provider.ID,
		ModelID:     modelID,
		ModelType:   "image_edit",
		DisplayName: modelID,
		Status:      1,
	}
	require.NoError(t, db.Create(&row).Error)
	return row, provider
}

// newPredictionService 组装 PredictionService：repository 真实；algo.Client 指向 httptest 算法服务；dictSvc 可注入。
func newPredictionService(t *testing.T, db *gorm.DB, algoServerURL string, dictSvc dict.IDictService) *PredictionService {
	t.Helper()
	client, err := algorithm.NewClient(options.Algorithm{
		ServiceURL: algoServerURL,
		MaxRetry:   0,
		Timeout:    5,
	})
	require.NoError(t, err)
	lm := lifecycle.NewManager()
	t.Cleanup(func() { _ = lm.Shutdown(2 * time.Second) })
	return NewPredictionService(
		predrepo.NewPredLogRepository(db),
		airepo.NewModelRepository(db),
		airepo.NewProviderRepository(db),
		dictSvc,
		client,
		lm,
	)
}

// algoResponse 构造 Python 算法服务的标准信封响应 {"code":"...","msg":"...","data":...}
func algoResponse(code string, data any) string {
	raw, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf(`{"code":%q,"msg":"ok","data":%s}`, code, string(raw))
}

func createPredLog(t *testing.T, db *gorm.DB, userID int64, modelID string) *model.SysPredLog {
	t.Helper()
	log := &model.SysPredLog{
		BaseModel: model.BaseModel{CreateBy: userID},
		ModelID:   modelID,
		OriginMD5: "abc123abc123abc123abc123abc123ab",
		OriginURL: "http://test.local/origin.png",
		Status:    model.LogStatusProcessing,
	}
	require.NoError(t, db.Create(log).Error)
	return log
}

// TestResolveModelID_Explicit 请求模型存在、启用、类型 image_edit 且供应商可用 → 解析成功。
func TestResolveModelID_Explicit(t *testing.T) {
	db := testutil.NewTestDB(t)
	_, provider := seedImageModel(t, db, "aecrnet")
	svc := newPredictionService(t, db, "http://127.0.0.1:1", nil)

	modelID, providerID, err := svc.resolveModelID(context.Background(), "aecrnet")
	require.NoError(t, err)
	require.Equal(t, "aecrnet", modelID)
	require.Equal(t, provider.ID, providerID)
}

// TestResolveModelID_FallbackToDefaultDict 请求未带 model 时取平台默认字典 ai_model.default_model.image_edit。
func TestResolveModelID_FallbackToDefaultDict(t *testing.T) {
	db := testutil.NewTestDB(t)
	seedImageModel(t, db, "aecrnet")

	dictSvc := servicemocks.NewMockIDictService(t)
	dictSvc.EXPECT().GetByTypeCode(mock.Anything, "ai_model").
		Return([]vo.Option{{Label: "default_model.image_edit", Value: "aecrnet"}}, nil)

	svc := newPredictionService(t, db, "http://127.0.0.1:1", dictSvc)
	modelID, _, err := svc.resolveModelID(context.Background(), "")
	require.NoError(t, err)
	require.Equal(t, "aecrnet", modelID)
}

// TestResolveModelID_MissingDefault 平台默认未配置 → A0500，不回退代码常量。
func TestResolveModelID_MissingDefault(t *testing.T) {
	db := testutil.NewTestDB(t)
	dictSvc := servicemocks.NewMockIDictService(t)
	dictSvc.EXPECT().GetByTypeCode(mock.Anything, "ai_model").Return([]vo.Option{}, nil)

	svc := newPredictionService(t, db, "http://127.0.0.1:1", dictSvc)
	_, _, err := svc.resolveModelID(context.Background(), "")
	requireBizCode(t, err, common.BUSINESS_ERROR)
}

// TestResolveModelID_TypeMismatch 非 image_edit 类型模型不可用于图像任务 → A0400。
func TestResolveModelID_TypeMismatch(t *testing.T) {
	db := testutil.NewTestDB(t)
	provider := model.SysAiProvider{ProviderCode: "p-chat", DisplayName: "p", Status: 1}
	require.NoError(t, db.Create(&provider).Error)
	require.NoError(t, db.Create(&model.SysAiModel{
		ProviderID: provider.ID, ModelID: "chat-x", ModelType: "chat", DisplayName: "chat-x", Status: 1,
	}).Error)
	svc := newPredictionService(t, db, "http://127.0.0.1:1", nil)

	_, _, err := svc.resolveModelID(context.Background(), "chat-x")
	requireBizCode(t, err, common.PARAM_ERROR)
}

// TestExecuteAsync_Success 成功：日志置 completed，pred_md5/origin_md5 取算法服务返回的内容 MD5（不得用 URL 的 MD5 冒充）。
func TestExecuteAsync_Success(t *testing.T) {
	db := testutil.NewTestDB(t)
	userID := int64(910102)
	seedImageModel(t, db, "aecrnet")

	const resultMd5 = "d41d8cd98f00b204e9800998ecf8427e"
	const originMd5 = "0cc175b9c0f1b6a831c399e269772661"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(algoResponse("00000", map[string]any{
			"taskId": 1, "status": 2, "resultUrl": "http://result.local/out.png",
			"resultMd5": resultMd5, "originMd5": originMd5, "time": 3,
		})))
	}))
	t.Cleanup(srv.Close)

	svc := newPredictionService(t, db, srv.URL, nil)
	log := createPredLog(t, db, userID, "aecrnet")

	svc.executeAsync(context.Background(), log.ID, "aecrnet", "http://test.local/origin.png", "{}")

	var savedLog model.SysPredLog
	require.NoError(t, db.First(&savedLog, log.ID).Error)
	require.EqualValues(t, model.LogStatusCompleted, savedLog.Status)
	require.Equal(t, resultMd5, savedLog.PredMD5, "pred_md5 应为算法服务返回的结果图 MD5")
	require.Equal(t, originMd5, savedLog.OriginMD5, "origin_md5 应为算法服务返回的原图内容 MD5")

	status, err := svc.GetTaskStatus(context.Background(), log.ID, userID)
	require.NoError(t, err)
	require.Equal(t, resultMd5, status.ResultMd5)
	require.Equal(t, originMd5, status.OriginMd5)
}

// TestExecuteAsync_Failure 算法服务业务失败 → 日志置 failed 并写入错误信息。
func TestExecuteAsync_Failure(t *testing.T) {
	db := testutil.NewTestDB(t)
	userID := int64(910101)
	seedImageModel(t, db, "aecrnet")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(algoResponse("E50001", nil)))
	}))
	t.Cleanup(srv.Close)

	svc := newPredictionService(t, db, srv.URL, nil)
	log := createPredLog(t, db, userID, "aecrnet")

	svc.executeAsync(context.Background(), log.ID, "aecrnet", "http://test.local/origin.png", "{}")

	var savedLog model.SysPredLog
	require.NoError(t, db.First(&savedLog, log.ID).Error)
	require.EqualValues(t, model.LogStatusFailed, savedLog.Status)
	require.NotNil(t, savedLog.ErrorMessage)
}

// TestPredict_InvalidModelRejected 提交前模型校验失败 → A0400，且不落任何任务日志。
func TestPredict_InvalidModelRejected(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := newPredictionService(t, db, "http://127.0.0.1:1", nil)

	_, err := svc.Predict(context.Background(), "not-exist", "http://test.local/origin.png", "{}", 910103, nil)
	requireBizCode(t, err, common.PARAM_ERROR)

	var count int64
	require.NoError(t, db.Model(&model.SysPredLog{}).Count(&count).Error)
	require.EqualValues(t, 0, count, "模型校验失败不得落任务日志")
}
