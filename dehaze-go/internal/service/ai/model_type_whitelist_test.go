package ai

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
)

// TestModelTypeWhitelistBlocksUnknownValue python `AiModelCreate.model_type` / `AiModelUpdate.model_type`
// 均为 Literal[chat/embedding/rerank]，非法字面量在 python 由 pydantic 拦成 A0400；go 无校验层，
// 必须在 service 层拦成 A0400 且不落库（否则非法类型直接入库，污染模型类型筛选与目录展示）。
func TestModelTypeWhitelistBlocksUnknownValue(t *testing.T) {
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	svc := NewModelService(db, airepo.NewModelRepository(db), airepo.NewModelPriceRepository(db), nil, nil)
	ctx := context.Background()

	for _, modelType := range []string{"CHAT", "llm", "chat ", "rerank/embedding", "chat;drop", "聊天"} {
		modelID := fmt.Sprintf("whitelist_probe_%d", time.Now().UnixNano())
		_, err := svc.CreateModel(ctx, &bo.AiModelCreateForm{
			ProviderID: 1, ModelID: modelID, ModelType: modelType, DisplayName: "白名单探针",
		}, 1)

		bizErr, ok := common.AsBizError(err)
		require.True(t, ok, "model_type=%q 应以业务错误拒绝，实际 err=%v", modelType, err)
		require.Equal(t, common.PARAM_ERROR, bizErr.Code(), "model_type=%q 错误码应为 A0400", modelType)

		var count int64
		require.NoError(t, db.Table("sys_ai_model").Where("model_id = ?", modelID).Count(&count).Error)
		require.Zero(t, count, "model_type=%q 非法值不得落库", modelType)
	}

	// 白名单内取值不误伤：embedding 通过白名单后落到"缺 dimension"的单字段校验
	_, err := svc.CreateModel(ctx, &bo.AiModelCreateForm{
		ProviderID: 1, ModelID: "whitelist_probe_legal", ModelType: "embedding", DisplayName: "合法类型",
	}, 1)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok)
	require.Equal(t, common.PARAM_ERROR, bizErr.Code())
	require.Contains(t, bizErr.Message(), "dimension")
}

// TestListModelsRejectsUnknownModelTypeFilter python `AiModelPageQuery.model_type` 同为
// Literal[chat/embedding/rerank]，非法筛选值在 python 是请求校验阶段的 A0400（而非"过滤后空列表"）；
// go 无该校验层，须在 service 层拦成同码错误，避免拼错类型被静默当作"无匹配数据"。
func TestListModelsRejectsUnknownModelTypeFilter(t *testing.T) {
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	svc := NewModelService(db, airepo.NewModelRepository(db), airepo.NewModelPriceRepository(db), nil, nil)
	ctx := context.Background()

	for _, modelType := range []string{"CHAT", "llm", "chat ", "rerank/embedding", "聊天"} {
		query := &bo.AiModelQuery{ModelType: modelType}
		query.PageNum, query.PageSize = 1, 10

		_, err := svc.ListModels(ctx, query)

		bizErr, ok := common.AsBizError(err)
		require.True(t, ok, "modelType=%q 应被拒绝，实际 err=%v", modelType, err)
		require.Equal(t, common.PARAM_ERROR, bizErr.Code(), "modelType=%q 错误码应为 A0400", modelType)
	}

	// 空串＝不筛选（显式空串与不传在 go 侧不可区分，沿用既有空串差异登记口径）；合法值正常返回
	for _, modelType := range []string{"", "chat", "embedding", "rerank"} {
		query := &bo.AiModelQuery{ModelType: modelType}
		query.PageNum, query.PageSize = 1, 10

		result, err := svc.ListModels(ctx, query)

		require.NoError(t, err, "modelType=%q 应放行", modelType)
		require.NotNil(t, result)
	}
}
