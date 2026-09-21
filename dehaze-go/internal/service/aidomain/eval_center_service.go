package aidomain

import (
	"context"
	"encoding/json"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// sys_dict 读取失败时的回退默认值（与 config/sql/data/sys_dict.sql ai_eval 种子一致）。
const (
	regressionThresholdDefault  = 5
	consistencyThresholdDefault = 90
	reviewRatioDefault          = 1

	// 复核队列扫描的最近已完成评测数与复核明细聚合上限（纯技术参数）
	reviewScanRunLimit   = 50
	reviewAggregateLimit = 1000

	evalDictType = "ai_eval"
)

// EvalCenterService 评测中心跨 Agent 聚合与人工复核。
type EvalCenterService struct {
	eval   *repo.EvalRepository
	agents *repo.AgentRepository
}

// NewEvalCenterService 构造 EvalCenterService。
func NewEvalCenterService(eval *repo.EvalRepository, agents *repo.AgentRepository) *EvalCenterService {
	return &EvalCenterService{eval: eval, agents: agents}
}

// dictInt 读取 ai_eval 字典整型值，缺键/解析失败回退默认值。
func (s *EvalCenterService) dictInt(ctx context.Context, key string, fallback int) int {
	values, err := s.agents.LoadDictValues(ctx, evalDictType)
	if err != nil {
		return fallback
	}
	raw, ok := values[key]
	if !ok {
		return fallback
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return parsed
}

// Overview 评测总览（各 Agent 最近得分/门禁状态/退化标识）。
func (s *EvalCenterService) Overview(ctx context.Context) ([]EvalAgentOverviewVO, error) {
	agents, err := s.agents.ListAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 列表失败", err)
	}
	runs, err := s.eval.ListLatestRunsPerAgent(ctx, 2)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测记录失败", err)
	}
	runsByAgent := map[int64][]model.SysAiAgentEvalRun{}
	for _, run := range runs {
		runsByAgent[run.AgentID] = append(runsByAgent[run.AgentID], run)
	}
	threshold := s.dictInt(ctx, "regression_threshold", regressionThresholdDefault)

	items := make([]EvalAgentOverviewVO, 0, len(agents))
	for _, agent := range agents {
		agentRuns := runsByAgent[agent.ID]
		var latest, previous *model.SysAiAgentEvalRun
		if len(agentRuns) > 0 {
			latest = &agentRuns[0]
		}
		if len(agentRuns) > 1 {
			previous = &agentRuns[1]
		}
		item := EvalAgentOverviewVO{
			AgentID:    agent.ID,
			AgentCode:  agent.AgentCode,
			AgentName:  agent.Name,
			GateStatus: "none",
		}
		if latest != nil {
			summary := parseScoreSummary(latest.ScoreSummary)
			total := totalScoreOf(summary)
			item.RunID = &latest.ID
			item.RunTime = formatTime(latest.CreateTime)
			item.TriggerType = latest.TriggerType
			if latest.Status == 2 {
				item.GateStatus = "passed"
			} else {
				item.GateStatus = "failed"
			}
			item.TotalScore = total
			item.Dimensions = dimensionsOf(summary)
			if previous != nil {
				item.Degraded = isDegraded(total, totalScoreOf(parseScoreSummary(previous.ScoreSummary)), threshold)
			}
			item.HighRiskFailed = hasHighRiskFailed(parseRunResults(latest.Results))
		}
		items = append(items, item)
	}
	return items, nil
}

// Trends 评测历史趋势（按 Agent/时间范围过滤）。
func (s *EvalCenterService) Trends(ctx context.Context, agentID *int64, startTime, endTime *time.Time, limit int) ([]EvalTrendVO, error) {
	runs, err := s.eval.ListCompletedRuns(ctx, agentID, startTime, endTime, limit)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测记录失败", err)
	}
	agentIDSet := map[int64]struct{}{}
	ids := []int64{}
	for _, run := range runs {
		if _, ok := agentIDSet[run.AgentID]; ok {
			continue
		}
		agentIDSet[run.AgentID] = struct{}{}
		ids = append(ids, run.AgentID)
	}
	agents, err := s.agents.GetByIDs(ctx, ids)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 列表失败", err)
	}
	names := map[int64]string{}
	for _, agent := range agents {
		names[agent.ID] = agent.Name
	}
	items := make([]EvalTrendVO, 0, len(runs))
	for _, run := range runs {
		summary := parseScoreSummary(run.ScoreSummary)
		items = append(items, EvalTrendVO{
			RunID:       run.ID,
			AgentID:     run.AgentID,
			AgentName:   names[run.AgentID],
			TriggerType: run.TriggerType,
			Status:      run.Status,
			TotalScore:  totalScoreOf(summary),
			Dimensions:  dimensionsOf(summary),
			CreateTime:  formatTime(run.CreateTime),
		})
	}
	return items, nil
}

// CompareRuns 两次评测 run 得分对比。
func (s *EvalCenterService) CompareRuns(ctx context.Context, runID, baseRunID int64) (*EvalRunCompareVO, error) {
	run, err := s.eval.GetRun(ctx, runID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测记录失败", err)
	}
	base, err := s.eval.GetRun(ctx, baseRunID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测记录失败", err)
	}
	if run == nil || base == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "评测记录不存在")
	}
	if run.AgentID != base.AgentID {
		return nil, common.NewBizError(common.PARAM_ERROR, "两次评测不属于同一 Agent，无法对比")
	}
	currentSummary := parseScoreSummary(run.ScoreSummary)
	baseSummary := parseScoreSummary(base.ScoreSummary)
	return &EvalRunCompareVO{
		RunID:         run.ID,
		BaseRunID:     base.ID,
		AgentID:       run.AgentID,
		Current:       runSnapshotOf(run.ID, currentSummary, formatTime(run.CreateTime)),
		Base:          runSnapshotOf(base.ID, baseSummary, formatTime(base.CreateTime)),
		DimensionDiff: dimensionDiffOf(currentSummary, baseSummary),
		SampleDiff:    sampleDiffOf(parseRunResults(run.Results), parseRunResults(base.Results)),
	}, nil
}

// JudgeStatus 判分模型状态（人工复核一致率推导漂移）。
func (s *EvalCenterService) JudgeStatus(ctx context.Context) (*JudgeStatusVO, error) {
	threshold := s.dictInt(ctx, "judge_consistency_threshold", consistencyThresholdDefault)
	reviews, err := s.eval.ListReviews(ctx, reviewAggregateLimit)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询复核记录失败", err)
	}
	reviewed := 0
	agreeCount := 0
	for _, review := range reviews {
		if review.Status != 2 {
			continue
		}
		reviewed++
		if review.Agree != nil && *review.Agree == 1 {
			agreeCount++
		}
	}
	stats := JudgeReviewStatsVO{
		Total:         len(reviews),
		Pending:       len(reviews) - reviewed,
		Reviewed:      reviewed,
		AgreeCount:    agreeCount,
		DisagreeCount: reviewed - agreeCount,
	}
	if reviewed > 0 {
		stats.AgreementRate = round2(float64(agreeCount) / float64(reviewed) * 100)
	}
	result := &JudgeStatusVO{ConsistencyThreshold: threshold, ReviewStats: stats}
	if reviewed == 0 {
		result.ConsistencyState = "insufficient_data"
		return result, nil
	}
	if stats.AgreementRate >= float64(threshold) {
		result.ConsistencyState = "normal"
	} else {
		result.ConsistencyState = "drifted"
		result.DriftPaused = true
	}
	return result, nil
}

// ListReviews 复核队列（先按抽样规则幂等补齐待复核项）。
func (s *EvalCenterService) ListReviews(ctx context.Context, status *int) (*EvalReviewQueueVO, error) {
	runs, err := s.eval.ListCompletedRuns(ctx, nil, nil, nil, reviewScanRunLimit)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测记录失败", err)
	}
	if err := s.materializeReviews(ctx, runs); err != nil {
		return nil, err
	}
	reviews, err := s.eval.ListReviews(ctx, reviewAggregateLimit)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询复核记录失败", err)
	}
	if status != nil {
		filtered := make([]model.SysAiEvalReview, 0, len(reviews))
		for _, review := range reviews {
			if review.Status == *status {
				filtered = append(filtered, review)
			}
		}
		reviews = filtered
	}
	agentIDSet := map[int64]struct{}{}
	ids := []int64{}
	for _, review := range reviews {
		if _, ok := agentIDSet[review.AgentID]; ok {
			continue
		}
		agentIDSet[review.AgentID] = struct{}{}
		ids = append(ids, review.AgentID)
	}
	agents, err := s.agents.GetByIDs(ctx, ids)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 列表失败", err)
	}
	names := map[int64]string{}
	for _, agent := range agents {
		names[agent.ID] = agent.Name
	}
	result := &EvalReviewQueueVO{Items: make([]EvalReviewItemVO, 0, len(reviews))}
	for _, review := range reviews {
		var agree *bool
		if review.Agree != nil {
			value := *review.Agree == 1
			agree = &value
		}
		result.Items = append(result.Items, EvalReviewItemVO{
			ID:          review.ID,
			RunID:       review.RunID,
			SampleID:    review.SampleID,
			AgentID:     review.AgentID,
			AgentName:   names[review.AgentID],
			JudgePassed: review.JudgePassed == 1,
			RiskLevel:   review.RiskLevel,
			Status:      review.Status,
			Agree:       agree,
			Remark:      derefString(review.Remark),
			CreateTime:  formatTime(review.CreateTime),
		})
		if review.Status == 1 {
			result.Pending++
		} else {
			result.Reviewed++
		}
	}
	return result, nil
}

// materializeReviews 按抽样规则生成待复核项（唯一键幂等）。
func (s *EvalCenterService) materializeReviews(ctx context.Context, runs []model.SysAiAgentEvalRun) error {
	if len(runs) == 0 {
		return nil
	}
	ratio := s.dictInt(ctx, "judge_review_ratio", reviewRatioDefault)
	runIDs := make([]int64, 0, len(runs))
	for _, run := range runs {
		runIDs = append(runIDs, run.ID)
	}
	existing, err := s.eval.ListReviewsByRunIDs(ctx, runIDs)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询复核记录失败", err)
	}
	known := map[[2]int64]struct{}{}
	for _, review := range existing {
		known[[2]int64{review.RunID, review.SampleID}] = struct{}{}
	}
	pending := []model.SysAiEvalReview{}
	for _, run := range runs {
		for _, item := range parseRunResults(run.Results) {
			sampleID, ok := toFloat(item["sample_id"])
			if !ok {
				continue
			}
			passed, _ := item["passed"].(bool)
			if passed && !sampleHit(run.ID, int64(sampleID), ratio) {
				continue
			}
			key := [2]int64{run.ID, int64(sampleID)}
			if _, ok := known[key]; ok {
				continue
			}
			riskLevel, _ := item["risk_level"].(string)
			if riskLevel == "" {
				riskLevel = "low"
			}
			judgePassed := 0
			if passed {
				judgePassed = 1
			}
			pending = append(pending, model.SysAiEvalReview{
				RunID:       run.ID,
				SampleID:    int64(sampleID),
				AgentID:     run.AgentID,
				JudgePassed: judgePassed,
				RiskLevel:   riskLevel,
				Status:      1,
			})
			known[key] = struct{}{}
		}
	}
	if len(pending) == 0 {
		return nil
	}
	if err := s.eval.CreateReviews(ctx, pending); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "生成复核项失败", err)
	}
	return nil
}

// ReviewDetail 复核详情（样本定义 + 实际输出 + 四维得分）。
func (s *EvalCenterService) ReviewDetail(ctx context.Context, runID, sampleID int64) (*EvalReviewDetailVO, error) {
	run, err := s.eval.GetRun(ctx, runID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测记录失败", err)
	}
	if run == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "评测记录不存在")
	}
	var matched map[string]any
	for _, item := range parseRunResults(run.Results) {
		if id, ok := toFloat(item["sample_id"]); ok && int64(id) == sampleID {
			matched = item
			break
		}
	}
	if matched == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "该评测记录中没有此样本的执行结果")
	}
	sample, err := s.eval.GetSample(ctx, sampleID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测样本失败", err)
	}
	agent, err := s.agents.GetByID(ctx, run.AgentID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 失败", err)
	}

	result := &EvalReviewDetailVO{
		RunID:        run.ID,
		AgentID:      run.AgentID,
		SampleID:     sampleID,
		TaskGoal:     stringField(matched, "task_goal"),
		RiskLevel:    stringField(matched, "risk_level"),
		JudgePassed:  boolField(matched, "passed"),
		ActualOutput: stringField(matched, "actual_output"),
		Error:        stringField(matched, "error"),
		Scores:       floatMapField(matched, "scores"),
		Notes:        stringMapField(matched, "notes"),
	}
	if agent != nil {
		result.AgentName = agent.Name
	}
	if sample != nil {
		if result.TaskGoal == "" {
			result.TaskGoal = sample.TaskGoal
		}
		result.AllowedInput = derefString(sample.AllowedInput)
		result.ExpectedResult = derefString(sample.ExpectedResult)
		result.ExpectedProcess = derefString(sample.ExpectedProcess)
		result.ForbiddenBehavior = derefString(sample.ForbiddenBehavior)
		result.Tools = parseJSONStringSlice(sample.Tools)
		if result.RiskLevel == "" {
			result.RiskLevel = sample.RiskLevel
		}
	}
	if result.RiskLevel == "" {
		result.RiskLevel = "low"
	}
	return result, nil
}

// SubmitReview 复核结果回填（判定一致/不一致 + 备注），重复复核不允许。
func (s *EvalCenterService) SubmitReview(ctx context.Context, reviewID int64, form *EvalReviewSubmitForm, reviewerID int64) (*EvalReviewItemVO, error) {
	review, err := s.eval.GetReview(ctx, reviewID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询复核项失败", err)
	}
	if review == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "复核项不存在")
	}
	if review.Status == 2 {
		return nil, common.NewBizError(common.OPERATION_NOT_ALLOW, "该复核项已完成复核，不允许重复回填")
	}
	agree := 0
	if form.Agree {
		agree = 1
	}
	fields := map[string]any{
		"agree":       agree,
		"status":      2,
		"reviewer_id": reviewerID,
		"remark":      form.Remark,
	}
	if err := s.eval.UpdateReviewFields(ctx, reviewID, fields); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "回填复核结果失败", err)
	}
	agreeValue := form.Agree
	return &EvalReviewItemVO{
		ID:          review.ID,
		RunID:       review.RunID,
		SampleID:    review.SampleID,
		AgentID:     review.AgentID,
		JudgePassed: review.JudgePassed == 1,
		RiskLevel:   review.RiskLevel,
		Status:      2,
		Agree:       &agreeValue,
		Remark:      derefString(form.Remark),
	}, nil
}

func stringField(source map[string]any, key string) string {
	if value, ok := source[key].(string); ok {
		return value
	}
	return ""
}

func boolField(source map[string]any, key string) bool {
	value, _ := source[key].(bool)
	return value
}

func floatMapField(source map[string]any, key string) map[string]float64 {
	raw, ok := source[key].(map[string]any)
	if !ok {
		return map[string]float64{}
	}
	result := make(map[string]float64, len(raw))
	for field, value := range raw {
		if number, ok := toFloat(value); ok {
			result[field] = number
		}
	}
	return result
}

func stringMapField(source map[string]any, key string) map[string]string {
	raw, ok := source[key].(map[string]any)
	if !ok {
		return map[string]string{}
	}
	result := make(map[string]string, len(raw))
	for field, value := range raw {
		if text, ok := value.(string); ok {
			result[field] = text
		}
	}
	return result
}

func parseJSONStringSlice(raw string) []string {
	if raw == "" {
		return nil
	}
	var parsed []string
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil
	}
	return parsed
}
