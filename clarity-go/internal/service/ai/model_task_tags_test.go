package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model/bo"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// TestTaskTagsRoundTripAndFilter task_tags 落库/回显 + 列表过滤（JSON 数组含任一即命中）
func TestTaskTagsRoundTripAndFilter(t *testing.T) {
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	svc := NewModelService(db, airepo.NewModelRepository(db), airepo.NewModelPriceRepository(db), nil, nil, newTestFavoriteService(db), nil)
	ctx := context.Background()
	stamp := time.Now().UnixNano()

	dehazeID := fmt.Sprintf("task_tags_dehaze_%d", stamp)
	superID := fmt.Sprintf("task_tags_super_%d", stamp)
	plainID := fmt.Sprintf("task_tags_plain_%d", stamp)

	created, err := svc.CreateModel(ctx, &bo.AiModelCreateForm{
		ProviderID: 1, ModelID: dehazeID, ModelType: "image_edit",
		DisplayName: "去雾模型", TaskTags: json.RawMessage(`["dehaze"]`),
	}, 1)
	require.NoError(t, err)
	require.JSONEq(t, `["dehaze"]`, string(created.TaskTags))

	_, err = svc.CreateModel(ctx, &bo.AiModelCreateForm{
		ProviderID: 1, ModelID: superID, ModelType: "image_edit",
		DisplayName: "超分模型", TaskTags: json.RawMessage(`["super_resolution"]`),
	}, 1)
	require.NoError(t, err)

	plain, err := svc.CreateModel(ctx, &bo.AiModelCreateForm{
		ProviderID: 1, ModelID: plainID, ModelType: "image_edit", DisplayName: "无标签模型",
	}, 1)
	require.NoError(t, err)
	require.Nil(t, plain.TaskTags, "未传 taskTags 应保持 NULL")

	listByTag := func(tags string) map[string]bool {
		query := &bo.AiModelQuery{Keyword: "task_tags_", TaskTags: tags}
		query.PageNum, query.PageSize = 1, 50
		result, err := svc.ListModels(ctx, query)
		require.NoError(t, err)
		found := map[string]bool{}
		for _, item := range result.List {
			found[item.ModelID] = true
		}
		return found
	}

	single := listByTag("dehaze")
	require.True(t, single[dehazeID])
	require.False(t, single[superID])
	require.False(t, single[plainID], "无标签行不应命中任务标签筛选")

	either := listByTag("dehaze, super_resolution")
	require.True(t, either[dehazeID])
	require.True(t, either[superID])
	require.False(t, either[plainID])
}

func TestParseTaskTagsAndHit(t *testing.T) {
	require.Nil(t, parseTaskTags(""))
	require.Nil(t, parseTaskTags("  ,  "))
	require.Equal(t, []string{"a", "b"}, parseTaskTags(" a , ,b "))

	require.True(t, taskTagsHit(json.RawMessage(`["dehaze","stream"]`), []string{"stream"}))
	require.False(t, taskTagsHit(json.RawMessage(`["dehaze"]`), []string{"stream"}))
	require.False(t, taskTagsHit(nil, []string{"dehaze"}))
	require.False(t, taskTagsHit(json.RawMessage(`"dehaze"`), []string{"dehaze"}))
}
