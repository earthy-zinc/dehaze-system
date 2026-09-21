package aidomain

import (
	"encoding/json"
	"math"
)

// EvalAgentOverviewVO 评测总览项。
type EvalAgentOverviewVO struct {
	AgentID        int64              `json:"agentId"`
	AgentCode      string             `json:"agentCode"`
	AgentName      string             `json:"agentName"`
	RunID          *int64             `json:"runId,omitempty"`
	RunTime        string             `json:"runTime,omitempty"`
	TriggerType    string             `json:"triggerType,omitempty"`
	GateStatus     string             `json:"gateStatus"`
	TotalScore     *float64           `json:"totalScore,omitempty"`
	Dimensions     map[string]float64 `json:"dimensions,omitempty"`
	Degraded       bool               `json:"degraded"`
	HighRiskFailed bool               `json:"highRiskFailed"`
}

// EvalTrendVO 评测趋势项。
type EvalTrendVO struct {
	RunID       int64              `json:"runId"`
	AgentID     int64              `json:"agentId"`
	AgentName   string             `json:"agentName,omitempty"`
	TriggerType string             `json:"triggerType"`
	Status      int                `json:"status"`
	TotalScore  *float64           `json:"totalScore,omitempty"`
	Dimensions  map[string]float64 `json:"dimensions,omitempty"`
	CreateTime  string             `json:"createTime,omitempty"`
}

// EvalRunScoreSnapshotVO 单次评测得分快照。
type EvalRunScoreSnapshotVO struct {
	RunID       int64              `json:"runId"`
	TotalScore  *float64           `json:"totalScore,omitempty"`
	Dimensions  map[string]float64 `json:"dimensions,omitempty"`
	SampleCount int                `json:"sampleCount"`
	PassRate    *float64           `json:"passRate,omitempty"`
	CreateTime  string             `json:"createTime,omitempty"`
}

// EvalSampleDiffItemVO 样本级差异项。
type EvalSampleDiffItemVO struct {
	SampleID      int64    `json:"sampleId"`
	TaskGoal      string   `json:"taskGoal"`
	CurrentPassed *bool    `json:"currentPassed,omitempty"`
	BasePassed    *bool    `json:"basePassed,omitempty"`
	CurrentScore  *float64 `json:"currentScore,omitempty"`
	BaseScore     *float64 `json:"baseScore,omitempty"`
	ScoreDelta    *float64 `json:"scoreDelta,omitempty"`
}

// EvalSampleDiffVO 样本级差异汇总。
type EvalSampleDiffVO struct {
	Added          []EvalSampleDiffItemVO `json:"added"`
	Removed        []EvalSampleDiffItemVO `json:"removed"`
	Changed        []EvalSampleDiffItemVO `json:"changed"`
	UnchangedCount int                    `json:"unchangedCount"`
}

// EvalRunCompareVO 两次评测对比结果。
type EvalRunCompareVO struct {
	RunID         int64                  `json:"runId"`
	BaseRunID     int64                  `json:"baseRunId"`
	AgentID       int64                  `json:"agentId"`
	Current       EvalRunScoreSnapshotVO `json:"current"`
	Base          EvalRunScoreSnapshotVO `json:"base"`
	DimensionDiff map[string]float64     `json:"dimensionDiff"`
	SampleDiff    EvalSampleDiffVO       `json:"sampleDiff"`
}

// JudgeReviewStatsVO 人工复核统计。
type JudgeReviewStatsVO struct {
	Total         int     `json:"total"`
	Pending       int     `json:"pending"`
	Reviewed      int     `json:"reviewed"`
	AgreeCount    int     `json:"agreeCount"`
	DisagreeCount int     `json:"disagreeCount"`
	AgreementRate float64 `json:"agreementRate"`
}

// JudgeStatusVO 判分模型状态。
type JudgeStatusVO struct {
	ConsistencyState     string             `json:"consistencyState"`
	DriftPaused          bool               `json:"driftPaused"`
	ConsistencyThreshold int                `json:"consistencyThreshold"`
	ReviewStats          JudgeReviewStatsVO `json:"reviewStats"`
}

// EvalReviewItemVO 复核项。
type EvalReviewItemVO struct {
	ID          int64  `json:"id"`
	RunID       int64  `json:"runId"`
	SampleID    int64  `json:"sampleId"`
	AgentID     int64  `json:"agentId"`
	AgentName   string `json:"agentName,omitempty"`
	JudgePassed bool   `json:"judgePassed"`
	RiskLevel   string `json:"riskLevel"`
	Status      int    `json:"status"`
	Agree       *bool  `json:"agree,omitempty"`
	Remark      string `json:"remark,omitempty"`
	CreateTime  string `json:"createTime,omitempty"`
}

// EvalReviewQueueVO 复核队列。
type EvalReviewQueueVO struct {
	Items    []EvalReviewItemVO `json:"items"`
	Pending  int                `json:"pending"`
	Reviewed int                `json:"reviewed"`
}

// EvalReviewSubmitForm 复核回填表单。
type EvalReviewSubmitForm struct {
	Agree  bool    `json:"agree"`
	Remark *string `json:"remark"`
}

// EvalReviewDetailVO 复核详情。
type EvalReviewDetailVO struct {
	RunID             int64              `json:"runId"`
	AgentID           int64              `json:"agentId"`
	AgentName         string             `json:"agentName,omitempty"`
	SampleID          int64              `json:"sampleId"`
	TaskGoal          string             `json:"taskGoal"`
	AllowedInput      string             `json:"allowedInput,omitempty"`
	ExpectedResult    string             `json:"expectedResult,omitempty"`
	ExpectedProcess   string             `json:"expectedProcess,omitempty"`
	ForbiddenBehavior string             `json:"forbiddenBehavior,omitempty"`
	Tools             []string           `json:"tools,omitempty"`
	RiskLevel         string             `json:"riskLevel"`
	JudgePassed       bool               `json:"judgePassed"`
	ActualOutput      string             `json:"actualOutput,omitempty"`
	Error             string             `json:"error,omitempty"`
	Scores            map[string]float64 `json:"scores"`
	Notes             map[string]string  `json:"notes"`
}

var evalDimensions = []string{"result_quality", "process_compliance", "safety_boundary", "efficiency"}

// parseScoreSummary 解析 score_summary JSON。
func parseScoreSummary(raw string) map[string]any {
	if raw == "" {
		return map[string]any{}
	}
	var summary map[string]any
	if err := json.Unmarshal([]byte(raw), &summary); err != nil {
		return map[string]any{}
	}
	return summary
}

// parseRunResults 解析 run.results JSON 数组。
func parseRunResults(raw string) []map[string]any {
	if raw == "" {
		return nil
	}
	var results []map[string]any
	if err := json.Unmarshal([]byte(raw), &results); err != nil {
		return nil
	}
	return results
}

// dimensionsOf 取四维得分。
func dimensionsOf(summary map[string]any) map[string]float64 {
	raw, ok := summary["dimensions"].(map[string]any)
	if !ok {
		return nil
	}
	result := make(map[string]float64, len(raw))
	for key, value := range raw {
		if number, ok := toFloat(value); ok {
			result[key] = number
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// totalScoreOf 四维均值（无维度返回 nil）。
func totalScoreOf(summary map[string]any) *float64 {
	dimensions := dimensionsOf(summary)
	if len(dimensions) == 0 {
		return nil
	}
	sum := 0.0
	for _, value := range dimensions {
		sum += value
	}
	total := round2(sum / float64(len(dimensions)))
	return &total
}

// sampleTotalOf 单样本四维均值。
func sampleTotalOf(result map[string]any) *float64 {
	raw, ok := result["scores"].(map[string]any)
	if !ok || len(raw) == 0 {
		return nil
	}
	sum := 0.0
	count := 0
	for _, value := range raw {
		if number, ok := toFloat(value); ok {
			sum += number
			count++
		}
	}
	if count == 0 {
		return nil
	}
	total := round2(sum / float64(count))
	return &total
}

// isDegraded 相对上次评测总分下降超阈值即退化。
func isDegraded(current, previous *float64, threshold int) bool {
	if current == nil || previous == nil || *previous <= 0 {
		return false
	}
	return (*previous-*current) / *previous * 100 > float64(threshold)
}

func hasHighRiskFailed(results []map[string]any) bool {
	for _, result := range results {
		if result["risk_level"] == "high" && result["passed"] != true {
			return true
		}
	}
	return false
}

func runSnapshotOf(runID int64, summary map[string]any, createTime string) EvalRunScoreSnapshotVO {
	snapshot := EvalRunScoreSnapshotVO{
		RunID:      runID,
		TotalScore: totalScoreOf(summary),
		Dimensions: dimensionsOf(summary),
		CreateTime: createTime,
	}
	if count, ok := toFloat(summary["sample_count"]); ok {
		snapshot.SampleCount = int(count)
	}
	if rate, ok := toFloat(summary["pass_rate"]); ok {
		snapshot.PassRate = &rate
	}
	return snapshot
}

func dimensionDiffOf(current, base map[string]any) map[string]float64 {
	currentDimensions := dimensionsOf(current)
	baseDimensions := dimensionsOf(base)
	result := make(map[string]float64, len(evalDimensions))
	for _, dim := range evalDimensions {
		result[dim] = round2(currentDimensions[dim] - baseDimensions[dim])
	}
	return result
}

func sampleDiffOf(current, base []map[string]any) EvalSampleDiffVO {
	toMap := func(results []map[string]any) map[int64]map[string]any {
		mapped := make(map[int64]map[string]any, len(results))
		for _, result := range results {
			if id, ok := toFloat(result["sample_id"]); ok {
				mapped[int64(id)] = result
			}
		}
		return mapped
	}
	currentMap := toMap(current)
	baseMap := toMap(base)

	item := func(sampleID int64, result, baseResult map[string]any) EvalSampleDiffItemVO {
		entry := EvalSampleDiffItemVO{SampleID: sampleID}
		if goal, ok := result["task_goal"].(string); ok {
			entry.TaskGoal = goal
		}
		if passed, ok := result["passed"].(bool); ok {
			entry.CurrentPassed = &passed
		}
		if baseResult != nil {
			if passed, ok := baseResult["passed"].(bool); ok {
				entry.BasePassed = &passed
			}
		}
		entry.CurrentScore = sampleTotalOf(result)
		if baseResult != nil {
			entry.BaseScore = sampleTotalOf(baseResult)
			if entry.CurrentScore != nil && entry.BaseScore != nil {
				delta := round2(*entry.CurrentScore - *entry.BaseScore)
				entry.ScoreDelta = &delta
			}
		}
		return entry
	}

	diff := EvalSampleDiffVO{
		Added:   []EvalSampleDiffItemVO{},
		Removed: []EvalSampleDiffItemVO{},
		Changed: []EvalSampleDiffItemVO{},
	}
	for sampleID, result := range currentMap {
		if _, ok := baseMap[sampleID]; !ok {
			diff.Added = append(diff.Added, item(sampleID, result, nil))
		}
	}
	for sampleID, result := range baseMap {
		if _, ok := currentMap[sampleID]; !ok {
			diff.Removed = append(diff.Removed, item(sampleID, result, nil))
		}
	}
	for sampleID, result := range currentMap {
		baseResult, ok := baseMap[sampleID]
		if !ok {
			continue
		}
		currentTotal := sampleTotalOf(result)
		baseTotal := sampleTotalOf(baseResult)
		samePassed := result["passed"] == baseResult["passed"]
		sameScore := (currentTotal == nil && baseTotal == nil) ||
			(currentTotal != nil && baseTotal != nil && *currentTotal == *baseTotal)
		if samePassed && sameScore {
			diff.UnchangedCount++
			continue
		}
		diff.Changed = append(diff.Changed, item(sampleID, result, baseResult))
	}
	return diff
}

// sampleHit 确定性抽样：同一 (run_id, sample_id) 结果恒定。
func sampleHit(runID, sampleID int64, ratio int) bool {
	return (runID*1000003+sampleID)%100 < int64(ratio)
}

func round2(v float64) float64 {
	return math.Round(v*100) / 100
}

func toFloat(value any) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, true
	case int:
		return float64(number), true
	case int64:
		return float64(number), true
	case json.Number:
		parsed, err := number.Float64()
		return parsed, err == nil
	}
	return 0, false
}

func parseJSONMap(raw string) map[string]any {
	if raw == "" {
		return nil
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil
	}
	return parsed
}

func parseJSONStringMap(raw string) map[string]string {
	if raw == "" {
		return nil
	}
	var parsed map[string]string
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil
	}
	return parsed
}

func parseJSONFloatMap(raw string) map[string]float64 {
	if raw == "" {
		return nil
	}
	var parsed map[string]float64
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil
	}
	return parsed
}
