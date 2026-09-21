package aidomain

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTotalScoreOf(t *testing.T) {
	summary := map[string]any{
		"dimensions": map[string]any{
			"result_quality":     90.0,
			"process_compliance": 80.0,
			"safety_boundary":    100.0,
			"efficiency":         70.0,
		},
	}
	total := totalScoreOf(summary)
	require.NotNil(t, total)
	assert.Equal(t, 85.0, *total)

	assert.Nil(t, totalScoreOf(map[string]any{}))
	assert.Nil(t, totalScoreOf(map[string]any{"dimensions": map[string]any{}}))
}

func TestIsDegraded(t *testing.T) {
	current := 80.0
	previous := 90.0
	// 下降 11.1% > 阈值 5% → 退化
	assert.True(t, isDegraded(&current, &previous, 5))
	// 阈值 20% 时不判退化
	assert.False(t, isDegraded(&current, &previous, 20))
	// 无基准不判退化
	assert.False(t, isDegraded(&current, nil, 5))
	// 提升不判退化
	higher := 95.0
	assert.False(t, isDegraded(&higher, &previous, 5))
}

func TestSampleHitIsDeterministic(t *testing.T) {
	first := sampleHit(12, 34, 50)
	for i := 0; i < 5; i++ {
		assert.Equal(t, first, sampleHit(12, 34, 50))
	}
	// 抽样比例 100 时全部命中，比例 0 时全部不命中
	assert.True(t, sampleHit(7, 9, 100))
	assert.False(t, sampleHit(7, 9, 0))
}

func TestSampleDiffOf(t *testing.T) {
	current := []map[string]any{
		{"sample_id": 1.0, "task_goal": "A", "passed": true, "scores": map[string]any{"result_quality": 90.0}},
		{"sample_id": 2.0, "task_goal": "B", "passed": false, "scores": map[string]any{"result_quality": 40.0}},
		{"sample_id": 3.0, "task_goal": "C", "passed": true, "scores": map[string]any{"result_quality": 60.0}},
	}
	base := []map[string]any{
		{"sample_id": 1.0, "task_goal": "A", "passed": true, "scores": map[string]any{"result_quality": 90.0}},
		{"sample_id": 2.0, "task_goal": "B", "passed": true, "scores": map[string]any{"result_quality": 80.0}},
		{"sample_id": 4.0, "task_goal": "D", "passed": true, "scores": map[string]any{"result_quality": 70.0}},
	}
	diff := sampleDiffOf(current, base)
	assert.Len(t, diff.Added, 1)
	assert.Equal(t, int64(3), diff.Added[0].SampleID)
	assert.Len(t, diff.Removed, 1)
	assert.Equal(t, int64(4), diff.Removed[0].SampleID)
	assert.Len(t, diff.Changed, 1)
	assert.Equal(t, int64(2), diff.Changed[0].SampleID)
	require.NotNil(t, diff.Changed[0].ScoreDelta)
	assert.Equal(t, -40.0, *diff.Changed[0].ScoreDelta)
	assert.Equal(t, 1, diff.UnchangedCount)
}

func TestDimensionDiffOf(t *testing.T) {
	current := map[string]any{"dimensions": map[string]any{"result_quality": 90.0, "efficiency": 60.0}}
	base := map[string]any{"dimensions": map[string]any{"result_quality": 80.0, "efficiency": 70.0}}
	diff := dimensionDiffOf(current, base)
	assert.Equal(t, 10.0, diff["result_quality"])
	assert.Equal(t, -10.0, diff["efficiency"])
	assert.Equal(t, 0.0, diff["process_compliance"])
	assert.Equal(t, 0.0, diff["safety_boundary"])
}

func TestRound2(t *testing.T) {
	assert.Equal(t, 1.23, round2(1.2349))
	assert.Equal(t, -1.23, round2(-1.2349))
	assert.Equal(t, 2.0, round2(2.0))
}
