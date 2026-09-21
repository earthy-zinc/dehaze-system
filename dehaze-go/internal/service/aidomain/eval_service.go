package aidomain

import (
	"context"
	"encoding/json"

	"github.com/earthyzinc/dehaze-go/internal/model"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

var validDatasetTypes = map[string]struct{}{"dev": {}, "regression": {}, "heldout": {}}
var validRiskLevels = map[string]struct{}{"low": {}, "medium": {}, "high": {}}

// EvalDatasetVO 评测集。
type EvalDatasetVO struct {
	ID          int64  `json:"id"`
	AgentID     int64  `json:"agentId"`
	Name        string `json:"name"`
	Description string `json:"description"`
	DatasetType string `json:"datasetType"`
	CreateTime  string `json:"createTime,omitempty"`
}

// EvalSampleVO 评测样本。
type EvalSampleVO struct {
	ID                int64           `json:"id"`
	DatasetID         int64           `json:"datasetId"`
	TaskGoal          string          `json:"taskGoal"`
	AllowedInput      string          `json:"allowedInput,omitempty"`
	Tools             json.RawMessage `json:"tools,omitempty"`
	ExpectedProcess   string          `json:"expectedProcess,omitempty"`
	ExpectedResult    string          `json:"expectedResult,omitempty"`
	ForbiddenBehavior string          `json:"forbiddenBehavior,omitempty"`
	RiskLevel         string          `json:"riskLevel"`
	CreateTime        string          `json:"createTime,omitempty"`
}

// EvalDatasetCreateForm 创建评测集表单。
// 线格式与 python 一致：`EvalDatasetCreate` 是纯 BaseModel（无 camelCase 别名），wire 字段为 snake_case。
type EvalDatasetCreateForm struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	DatasetType string `json:"dataset_type"`
}

// EvalDatasetUpdateForm 更新评测集表单。
type EvalDatasetUpdateForm struct {
	Name        *string `json:"name"`
	Description *string `json:"description"`
}

// EvalSampleCreateForm 创建样本表单（python 侧同为纯 BaseModel，wire 字段 snake_case）。
type EvalSampleCreateForm struct {
	DatasetID         int64    `json:"dataset_id"`
	TaskGoal          string   `json:"task_goal"`
	AllowedInput      *string  `json:"allowed_input"`
	Tools             []string `json:"tools"`
	ExpectedProcess   *string  `json:"expected_process"`
	ExpectedResult    *string  `json:"expected_result"`
	ForbiddenBehavior *string  `json:"forbidden_behavior"`
	RiskLevel         string   `json:"risk_level"`
}

// EvalSampleUpdateForm 更新样本表单（wire 字段同 Create）。
type EvalSampleUpdateForm struct {
	TaskGoal          *string  `json:"task_goal"`
	AllowedInput      *string  `json:"allowed_input"`
	Tools             []string `json:"tools"`
	ExpectedProcess   *string  `json:"expected_process"`
	ExpectedResult    *string  `json:"expected_result"`
	ForbiddenBehavior *string  `json:"forbidden_behavior"`
	RiskLevel         *string  `json:"risk_level"`
}

// EvalService 评测集与样本 CRUD（评测执行为 B 类，由 python 承接）。
type EvalService struct {
	eval     *repo.EvalRepository
	agents   *repo.AgentRepository
	auditLog *auditlogservice.AuditLogService
}

// NewEvalService 构造 EvalService。
func NewEvalService(eval *repo.EvalRepository, agents *repo.AgentRepository, auditLog *auditlogservice.AuditLogService) *EvalService {
	return &EvalService{eval: eval, agents: agents, auditLog: auditLog}
}

// CreateDataset 创建评测集（同 Agent 同类型唯一，软删行复活）。
func (s *EvalService) CreateDataset(ctx context.Context, agentID int64, form *EvalDatasetCreateForm) (*EvalDatasetVO, error) {
	if _, ok := validDatasetTypes[form.DatasetType]; !ok {
		return nil, common.NewBizError(common.PARAM_ERROR, "评测集类型取值非法")
	}
	if form.Name == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "评测集名称不能为空")
	}
	if err := s.requireAgent(ctx, agentID); err != nil {
		return nil, err
	}
	existing, err := s.eval.GetDatasetByAgentAndType(ctx, agentID, form.DatasetType)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测集失败", err)
	}
	if existing != nil {
		if existing.Deleted == 0 {
			return nil, common.NewBizError(common.DATA_EXISTS, "该 Agent 已存在同类型评测集")
		}
		fields := map[string]any{
			"name":        form.Name,
			"description": form.Description,
			"deleted":     0,
		}
		if err := s.eval.UpdateDatasetFields(ctx, existing.ID, fields); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "恢复评测集失败", err)
		}
		existing.Name = form.Name
		existing.Description = form.Description
		existing.Deleted = 0
		return toDatasetVO(existing), nil
	}
	dataset := &model.SysAiAgentEvalDataset{
		AgentID:     agentID,
		Name:        form.Name,
		Description: form.Description,
		DatasetType: form.DatasetType,
	}
	if err := s.eval.CreateDataset(ctx, dataset); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建评测集失败", err)
	}
	return toDatasetVO(dataset), nil
}

// ListDatasets 评测集列表。
func (s *EvalService) ListDatasets(ctx context.Context, agentID int64) ([]EvalDatasetVO, error) {
	items, err := s.eval.ListDatasetsByAgent(ctx, agentID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测集失败", err)
	}
	result := make([]EvalDatasetVO, 0, len(items))
	for i := range items {
		result = append(result, *toDatasetVO(&items[i]))
	}
	return result, nil
}

// UpdateDataset 更新评测集。
func (s *EvalService) UpdateDataset(ctx context.Context, agentID, datasetID int64, form *EvalDatasetUpdateForm) (*EvalDatasetVO, error) {
	dataset, err := s.requireDatasetOfAgent(ctx, agentID, datasetID)
	if err != nil {
		return nil, err
	}
	fields := map[string]any{}
	if form.Name != nil {
		fields["name"] = *form.Name
		dataset.Name = *form.Name
	}
	if form.Description != nil {
		fields["description"] = *form.Description
		dataset.Description = *form.Description
	}
	if len(fields) > 0 {
		if err := s.eval.UpdateDatasetFields(ctx, datasetID, fields); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新评测集失败", err)
		}
	}
	return toDatasetVO(dataset), nil
}

// DeleteDataset 删除评测集（级联物理清理样本，留审计）。
func (s *EvalService) DeleteDataset(ctx context.Context, agentID, datasetID, operatorID int64) error {
	dataset, err := s.requireDatasetOfAgent(ctx, agentID, datasetID)
	if err != nil {
		return err
	}
	if _, err := s.eval.DeleteSamplesByDatasets(ctx, []int64{datasetID}); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "清理评测样本失败", err)
	}
	if err := s.eval.SoftDeleteDatasets(ctx, []int64{datasetID}, operatorID); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除评测集失败", err)
	}
	if s.auditLog != nil {
		s.auditLog.RecordAuditAsync(ctx, operatorID, "ai_eval_dataset", datasetID, "delete", "ai_eval",
			map[string]any{"agent_id": agentID, "name": dataset.Name, "dataset_type": dataset.DatasetType}, nil, "", "")
	}
	return nil
}

// CreateSample 创建评测样本。
func (s *EvalService) CreateSample(ctx context.Context, agentID, datasetID int64, form *EvalSampleCreateForm) (*EvalSampleVO, error) {
	if form.DatasetID != datasetID {
		return nil, common.NewBizError(common.PARAM_ERROR, "样本所属评测集与路径不一致")
	}
	if _, err := s.requireDatasetOfAgent(ctx, agentID, datasetID); err != nil {
		return nil, err
	}
	if form.TaskGoal == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "任务目标不能为空")
	}
	riskLevel := form.RiskLevel
	if riskLevel == "" {
		riskLevel = "low"
	}
	if _, ok := validRiskLevels[riskLevel]; !ok {
		return nil, common.NewBizError(common.PARAM_ERROR, "风险等级取值非法")
	}
	sample := &model.SysAiAgentEvalSample{
		DatasetID:         datasetID,
		TaskGoal:          form.TaskGoal,
		AllowedInput:      form.AllowedInput,
		Tools:             marshalJSON(form.Tools),
		ExpectedProcess:   form.ExpectedProcess,
		ExpectedResult:    form.ExpectedResult,
		ForbiddenBehavior: form.ForbiddenBehavior,
		RiskLevel:         riskLevel,
	}
	if err := s.eval.CreateSample(ctx, sample); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建评测样本失败", err)
	}
	return toSampleVO(sample), nil
}

// ListSamples 评测样本列表。
func (s *EvalService) ListSamples(ctx context.Context, agentID, datasetID int64) ([]EvalSampleVO, error) {
	if _, err := s.requireDatasetOfAgent(ctx, agentID, datasetID); err != nil {
		return nil, err
	}
	items, err := s.eval.ListSamplesByDataset(ctx, datasetID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测样本失败", err)
	}
	result := make([]EvalSampleVO, 0, len(items))
	for i := range items {
		result = append(result, *toSampleVO(&items[i]))
	}
	return result, nil
}

// UpdateSample 更新评测样本。
func (s *EvalService) UpdateSample(ctx context.Context, agentID, sampleID int64, form *EvalSampleUpdateForm) (*EvalSampleVO, error) {
	sample, err := s.requireSampleOfAgent(ctx, agentID, sampleID)
	if err != nil {
		return nil, err
	}
	fields := map[string]any{}
	if form.TaskGoal != nil {
		fields["task_goal"] = *form.TaskGoal
		sample.TaskGoal = *form.TaskGoal
	}
	if form.AllowedInput != nil {
		fields["allowed_input"] = *form.AllowedInput
		sample.AllowedInput = form.AllowedInput
	}
	if form.Tools != nil {
		tools := marshalJSON(form.Tools)
		fields["tools"] = tools
		sample.Tools = tools
	}
	if form.ExpectedProcess != nil {
		fields["expected_process"] = *form.ExpectedProcess
		sample.ExpectedProcess = form.ExpectedProcess
	}
	if form.ExpectedResult != nil {
		fields["expected_result"] = *form.ExpectedResult
		sample.ExpectedResult = form.ExpectedResult
	}
	if form.ForbiddenBehavior != nil {
		fields["forbidden_behavior"] = *form.ForbiddenBehavior
		sample.ForbiddenBehavior = form.ForbiddenBehavior
	}
	if form.RiskLevel != nil {
		if _, ok := validRiskLevels[*form.RiskLevel]; !ok {
			return nil, common.NewBizError(common.PARAM_ERROR, "风险等级取值非法")
		}
		fields["risk_level"] = *form.RiskLevel
		sample.RiskLevel = *form.RiskLevel
	}
	if len(fields) > 0 {
		if err := s.eval.UpdateSampleFields(ctx, sampleID, fields); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新评测样本失败", err)
		}
	}
	return toSampleVO(sample), nil
}

// DeleteSample 删除评测样本（物理删除，样本表无逻辑删除）。
func (s *EvalService) DeleteSample(ctx context.Context, agentID, sampleID, operatorID int64) error {
	sample, err := s.requireSampleOfAgent(ctx, agentID, sampleID)
	if err != nil {
		return err
	}
	if _, err := s.eval.DeleteSamples(ctx, []int64{sampleID}); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除评测样本失败", err)
	}
	if s.auditLog != nil {
		s.auditLog.RecordAuditAsync(ctx, operatorID, "ai_eval_sample", sampleID, "delete", "ai_eval",
			map[string]any{"dataset_id": sample.DatasetID, "task_goal": sample.TaskGoal}, nil, "", "")
	}
	return nil
}

func (s *EvalService) requireAgent(ctx context.Context, agentID int64) error {
	agent, err := s.agents.GetByID(ctx, agentID)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 失败", err)
	}
	if agent == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "Agent 不存在")
	}
	return nil
}

// requireDatasetOfAgent 评测集必须归属路径中的 Agent（跨 Agent 一律 404，不暴露存在性）。
func (s *EvalService) requireDatasetOfAgent(ctx context.Context, agentID, datasetID int64) (*model.SysAiAgentEvalDataset, error) {
	dataset, err := s.eval.GetDataset(ctx, datasetID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测集失败", err)
	}
	if dataset == nil || dataset.AgentID != agentID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "评测集不存在")
	}
	return dataset, nil
}

func (s *EvalService) requireSampleOfAgent(ctx context.Context, agentID, sampleID int64) (*model.SysAiAgentEvalSample, error) {
	sample, err := s.eval.GetSample(ctx, sampleID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测样本失败", err)
	}
	if sample == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "评测样本不存在")
	}
	if _, err := s.requireDatasetOfAgent(ctx, agentID, sample.DatasetID); err != nil {
		return nil, err
	}
	return sample, nil
}

func toDatasetVO(dataset *model.SysAiAgentEvalDataset) *EvalDatasetVO {
	return &EvalDatasetVO{
		ID:          dataset.ID,
		AgentID:     dataset.AgentID,
		Name:        dataset.Name,
		Description: dataset.Description,
		DatasetType: dataset.DatasetType,
		CreateTime:  formatTime(dataset.CreateTime),
	}
}

func toSampleVO(sample *model.SysAiAgentEvalSample) *EvalSampleVO {
	return &EvalSampleVO{
		ID:                sample.ID,
		DatasetID:         sample.DatasetID,
		TaskGoal:          sample.TaskGoal,
		AllowedInput:      derefString(sample.AllowedInput),
		Tools:             rawJSON(sample.Tools),
		ExpectedProcess:   derefString(sample.ExpectedProcess),
		ExpectedResult:    derefString(sample.ExpectedResult),
		ForbiddenBehavior: derefString(sample.ForbiddenBehavior),
		RiskLevel:         sample.RiskLevel,
		CreateTime:        formatTime(sample.CreateTime),
	}
}
