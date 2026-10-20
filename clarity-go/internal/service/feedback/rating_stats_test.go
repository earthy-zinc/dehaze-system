package feedback

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/vo"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	fbrepo "github.com/earthyzinc/clarity-go/internal/repository/feedback"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// TestRatingStatsModelStats 评价统计对齐 python feedback_repository：
// 全局 averageRating 与 modelStats 的 averageRating/lowRatingRate 均保留 2 位小数，
// 低分为 ≤2 星占比，modelName 取 sys_ai_model.display_name（无匹配模型时空串）。
// 统计以本用例写入的评分行为准，用窄时间窗隔离测试库中的历史数据。
func TestRatingStatsModelStats(t *testing.T) {
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	ratingRepo := fbrepo.NewRatingRepository(db)
	svc := NewRatingService(db, ratingRepo, nil, nil, nil, nil, nil, nil)

	stamp := time.Now().UnixNano()
	modelID := fmt.Sprintf("stats_model_%d", stamp)
	require.NoError(t, airepo.NewModelRepository(db).Create(ctx, &model.SysAiModel{
		ProviderID: 1, ModelID: modelID, ModelType: "image_edit", DisplayName: "统计模型",
		MaxContextTokens: 4096, MaxOutputTokens: 4096, SupportsStreaming: 1, Status: 1,
	}))
	orphanModelID := fmt.Sprintf("stats_orphan_%d", stamp)

	// 模型 5/4/1 → 均值 3.33、低分率 33.33；孤儿 model_id 3/3/3 → 均值 3、低分率 0
	createRatings(t, ctx, ratingRepo, modelID, []int8{5, 4, 1}, stamp)
	createRatings(t, ctx, ratingRepo, orphanModelID, []int8{3, 3, 3}, stamp+10)

	// 全局 6 评合计 19 分 → 未舍入为 3.1666…，必须下发 3.17
	stats, err := svc.GetRatingStats(ctx,
		time.Now().Add(-time.Minute).Format(timeFormat), time.Now().Add(time.Minute).Format(timeFormat))
	require.NoError(t, err)
	require.EqualValues(t, 6, stats.TotalRatings)
	require.Equal(t, 3.17, stats.AverageRating)
	require.Equal(t, map[int]int64{1: 1, 2: 0, 3: 3, 4: 1, 5: 1}, stats.RatingDistribution)

	byModelID := make(map[string]vo.ModelStat, len(stats.ModelStats))
	for _, item := range stats.ModelStats {
		byModelID[item.ModelID] = item
	}
	require.Len(t, stats.ModelStats, 2)

	stat, ok := byModelID[modelID]
	require.True(t, ok, "模型维度统计应含 %s，实际 %v", modelID, stats.ModelStats)
	require.Equal(t, "统计模型", stat.ModelName)
	require.EqualValues(t, 3, stat.TotalRatings)
	require.Equal(t, 3.33, stat.AverageRating)
	require.Equal(t, 33.33, stat.LowRatingRate)

	orphan, ok := byModelID[orphanModelID]
	require.True(t, ok, "无模型记录的评分仍需参与聚合（python 用 outerjoin）")
	require.Empty(t, orphan.ModelName)
	require.EqualValues(t, 3, orphan.TotalRatings)
	require.Equal(t, 3.0, orphan.AverageRating)
	require.Equal(t, 0.0, orphan.LowRatingRate)

	// 无数据区间：全局为 0、modelStats 仍为非 nil 空数组
	empty, err := svc.GetRatingStats(ctx, time.Now().Add(time.Hour).Format(timeFormat), "")
	require.NoError(t, err)
	require.EqualValues(t, 0, empty.TotalRatings)
	require.Equal(t, 0.0, empty.AverageRating)
	require.NotNil(t, empty.ModelStats)
	require.Empty(t, empty.ModelStats)

	for _, s := range []*vo.RatingStatsVO{stats, empty} {
		raw, err := json.Marshal(s)
		require.NoError(t, err)
		var payload map[string]any
		require.NoError(t, json.Unmarshal(raw, &payload))
		_, isArray := payload["modelStats"].([]any)
		require.True(t, isArray, "modelStats 必须序列化为数组，实际 %s", raw)
	}
}

func createRatings(
	t *testing.T, ctx context.Context, repo fbrepo.IRatingRepository, modelID string, ratings []int8, seq int64,
) {
	t.Helper()
	for i, rating := range ratings {
		require.NoError(t, repo.Create(ctx, &model.SysRating{
			UserID: 1, PredLogID: seq + int64(i), ModelID: modelID, Rating: rating,
			Tags: "[]", ImageUrls: "[]",
		}))
	}
}
