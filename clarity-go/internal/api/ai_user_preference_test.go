package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	aidomainrepo "github.com/earthyzinc/clarity-go/internal/repository/aidomain"
	dictrepo "github.com/earthyzinc/clarity-go/internal/repository/dict"
	aiservice "github.com/earthyzinc/clarity-go/internal/service/ai"
	dictservice "github.com/earthyzinc/clarity-go/internal/service/dict"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/security"
	"github.com/earthyzinc/clarity-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

var preferenceProbeSeq int64

// TestUserPreferenceEndpointsContract 端到端钉住 /ai/user-preference 的对外契约：
// 信封 {code,msg,data}、请求/响应字段名 defaultModel、错误码 A0601。
func TestUserPreferenceEndpointsContract(t *testing.T) {
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	service := aiservice.NewUserPreferenceService(
		airepo.NewUserPreferenceRepository(db),
		airepo.NewModelRepository(db),
		dictservice.NewDictService(dictrepo.NewDictRepository(db), dictrepo.NewDictTypeRepository(db), nil),
		aiservice.NewHealthService(db),
		aidomainrepo.NewAgentRepository(db),
	)
	userPreferenceApi := NewAiUserPreferenceApi(service)

	preferenceProbeSeq++
	userID := time.Now().UnixNano() + preferenceProbeSeq

	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.Use(middleware.ContextErrorHandler())
	engine.Use(func(c *gin.Context) { c.Set("claims", &security.CustomClaims{UserID: userID}) })
	engine.GET("/api/v1/ai/user-preference", userPreferenceApi.Get)
	engine.PUT("/api/v1/ai/user-preference", userPreferenceApi.Update)

	// 无记录 → 不出现 defaultModel 键（三端统一剔除 null，非 {"defaultModel":null}）
	got := callUserPreference(engine, http.MethodGet, "")
	require.Equal(t, common.SUCCESS.Code, got.Code)
	require.Equal(t, "{}", string(got.Data))
	require.NotContains(t, string(got.Data), "defaultModel")

	modelID := fmt.Sprintf("pref_api_probe_%d", userID)
	require.NoError(t, db.Create(&model.SysAiModel{
		ProviderID: 1, ModelID: modelID, ModelType: "chat", DisplayName: modelID,
		MaxContextTokens: 4096, MaxOutputTokens: 2048, SupportsStreaming: 1, Status: 1,
	}).Error)

	// 设置成功返回空 data，且可读回
	set := callUserPreference(engine, http.MethodPut, fmt.Sprintf(`{"defaultModel":%q}`, modelID))
	require.Equal(t, common.SUCCESS.Code, set.Code)
	require.Empty(t, set.Data, "PUT 成功应返回空 data")

	got = callUserPreference(engine, http.MethodGet, "")
	require.Equal(t, common.SUCCESS.Code, got.Code)
	require.JSONEq(t, fmt.Sprintf(`{"defaultModel":%q}`, modelID), string(got.Data))

	// defaultModel=null 清除偏好 → 回到「键不出现」
	cleared := callUserPreference(engine, http.MethodPut, `{"defaultModel":null}`)
	require.Equal(t, common.SUCCESS.Code, cleared.Code)
	got = callUserPreference(engine, http.MethodGet, "")
	require.Equal(t, "{}", string(got.Data))
	require.NotContains(t, string(got.Data), "defaultModel")

	// 不可用模型 → A0601
	denied := callUserPreference(engine, http.MethodPut, `{"defaultModel":"pref_api_missing_model"}`)
	require.Equal(t, common.AI_MODEL_NOT_AVAILABLE.Code, denied.Code)
	require.Equal(t, "模型不存在、未启用或当前等级不可用", denied.Msg)
}

func callUserPreference(engine *gin.Engine, method, body string) envelope {
	request := httptest.NewRequest(method, "/api/v1/ai/user-preference", bytes.NewBufferString(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)

	var env envelope
	_ = json.Unmarshal(recorder.Body.Bytes(), &env)
	return env
}
