package aidomain

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// completedRunStatuses 已完成评测状态（2:通过;3:失败），1 为执行中。
var completedRunStatuses = []int{2, 3}

// EvalRepository 评测集/样本/执行记录/人工复核的数据访问。
type EvalRepository struct {
	db *gorm.DB
}

func NewEvalRepository(db *gorm.DB) *EvalRepository {
	return &EvalRepository{db: db}
}

// ── 评测集 ────────────────────────────────────────────────────

func (r *EvalRepository) CreateDataset(ctx context.Context, d *model.SysAiAgentEvalDataset) error {
	return r.db.WithContext(ctx).Create(d).Error
}

func (r *EvalRepository) GetDataset(ctx context.Context, id int64) (*model.SysAiAgentEvalDataset, error) {
	var d model.SysAiAgentEvalDataset
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&d).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &d, err
}

// GetDatasetByAgentAndType 按 (agent_id, dataset_type) 查（含软删行，用于唯一键复活）。
// 必须 Unscoped：否则全局软删回调追加的 deleted = 0 会让调用方的"软删行复活"分支永远走不到，
// 退化成直接 INSERT → 撞唯一键。
func (r *EvalRepository) GetDatasetByAgentAndType(ctx context.Context, agentID int64, datasetType string) (*model.SysAiAgentEvalDataset, error) {
	var d model.SysAiAgentEvalDataset
	err := r.db.WithContext(ctx).Unscoped().
		Where("agent_id = ? AND dataset_type = ?", agentID, datasetType).First(&d).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &d, err
}

func (r *EvalRepository) ListDatasetsByAgent(ctx context.Context, agentID int64) ([]model.SysAiAgentEvalDataset, error) {
	var items []model.SysAiAgentEvalDataset
	err := r.db.WithContext(ctx).
		Where("agent_id = ? AND deleted = 0", agentID).
		Order("id DESC").Find(&items).Error
	return items, err
}

func (r *EvalRepository) UpdateDatasetFields(ctx context.Context, id int64, fields map[string]any) error {
	fields["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiAgentEvalDataset{}).
		Where("id = ?", id).Updates(fields).Error
}

func (r *EvalRepository) SoftDeleteDatasets(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiAgentEvalDataset{}).
		Where("id IN ?", ids).
		Updates(map[string]any{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// ── 样本 ──────────────────────────────────────────────────────

func (r *EvalRepository) CreateSample(ctx context.Context, s *model.SysAiAgentEvalSample) error {
	return r.db.WithContext(ctx).Create(s).Error
}

func (r *EvalRepository) GetSample(ctx context.Context, id int64) (*model.SysAiAgentEvalSample, error) {
	var s model.SysAiAgentEvalSample
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &s, err
}

func (r *EvalRepository) ListSamplesByDataset(ctx context.Context, datasetID int64) ([]model.SysAiAgentEvalSample, error) {
	var items []model.SysAiAgentEvalSample
	err := r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Order("id ASC").Find(&items).Error
	return items, err
}

func (r *EvalRepository) UpdateSampleFields(ctx context.Context, id int64, fields map[string]any) error {
	fields["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiAgentEvalSample{}).
		Where("id = ?", id).Updates(fields).Error
}

func (r *EvalRepository) DeleteSamples(ctx context.Context, ids []int64) (int64, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	res := r.db.WithContext(ctx).Where("id IN ?", ids).Delete(&model.SysAiAgentEvalSample{})
	return res.RowsAffected, res.Error
}

// DeleteSamplesByDatasets 评测集删除时级联物理清理样本（样本表无逻辑删除）。
func (r *EvalRepository) DeleteSamplesByDatasets(ctx context.Context, datasetIDs []int64) (int64, error) {
	if len(datasetIDs) == 0 {
		return 0, nil
	}
	res := r.db.WithContext(ctx).Where("dataset_id IN ?", datasetIDs).Delete(&model.SysAiAgentEvalSample{})
	return res.RowsAffected, res.Error
}

// ── 执行记录 ──────────────────────────────────────────────────

func (r *EvalRepository) CreateRun(ctx context.Context, run *model.SysAiAgentEvalRun) error {
	return r.db.WithContext(ctx).Create(run).Error
}

func (r *EvalRepository) GetRun(ctx context.Context, id int64) (*model.SysAiAgentEvalRun, error) {
	var run model.SysAiAgentEvalRun
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&run).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &run, err
}

func (r *EvalRepository) PaginateRunsByAgent(ctx context.Context, agentID int64, page, size int, datasetID *int64) ([]model.SysAiAgentEvalRun, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiAgentEvalRun{}).Where("agent_id = ?", agentID)
	if datasetID != nil {
		db = db.Where("dataset_id = ?", *datasetID)
	}
	db = db.Order("id DESC")
	return countAndFind[model.SysAiAgentEvalRun](db, page, size)
}

func (r *EvalRepository) DeleteRunsByAgent(ctx context.Context, agentID int64) (int64, error) {
	res := r.db.WithContext(ctx).Where("agent_id = ?", agentID).Delete(&model.SysAiAgentEvalRun{})
	return res.RowsAffected, res.Error
}

// ListCompletedRuns 已完成评测（时间升序，可按 Agent 与时间范围过滤）。
func (r *EvalRepository) ListCompletedRuns(ctx context.Context, agentID *int64, startTime, endTime *time.Time, limit int) ([]model.SysAiAgentEvalRun, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiAgentEvalRun{}).
		Where("status IN ?", completedRunStatuses)
	if agentID != nil {
		db = db.Where("agent_id = ?", *agentID)
	}
	if startTime != nil {
		db = db.Where("create_time >= ?", *startTime)
	}
	if endTime != nil {
		db = db.Where("create_time <= ?", *endTime)
	}
	var items []model.SysAiAgentEvalRun
	err := db.Order("create_time ASC, id ASC").Limit(limit).Find(&items).Error
	return items, err
}

// ListLatestRunsPerAgent 各 Agent 最近 perAgent 次已完成评测（窗口函数，供总览退化判定）。
func (r *EvalRepository) ListLatestRunsPerAgent(ctx context.Context, perAgent int) ([]model.SysAiAgentEvalRun, error) {
	var items []model.SysAiAgentEvalRun
	err := r.db.WithContext(ctx).Raw(`
		SELECT * FROM (
			SELECT r.*, ROW_NUMBER() OVER (PARTITION BY r.agent_id ORDER BY r.id DESC) AS rn
			FROM sys_ai_agent_eval_run r
			WHERE r.status IN (2, 3)
		) t WHERE t.rn <= ? ORDER BY t.agent_id ASC, t.id DESC`, perAgent).
		Scan(&items).Error
	return items, err
}

// ── 人工复核 ──────────────────────────────────────────────────

func (r *EvalRepository) GetReview(ctx context.Context, id int64) (*model.SysAiEvalReview, error) {
	var review model.SysAiEvalReview
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&review).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &review, err
}

// ListReviews 复核记录（待复核优先，id 倒序）。
func (r *EvalRepository) ListReviews(ctx context.Context, limit int) ([]model.SysAiEvalReview, error) {
	var items []model.SysAiEvalReview
	err := r.db.WithContext(ctx).
		Order("status ASC, id DESC").Limit(limit).Find(&items).Error
	return items, err
}

func (r *EvalRepository) ListReviewsByRunIDs(ctx context.Context, runIDs []int64) ([]model.SysAiEvalReview, error) {
	if len(runIDs) == 0 {
		return nil, nil
	}
	var items []model.SysAiEvalReview
	err := r.db.WithContext(ctx).Where("run_id IN ?", runIDs).Find(&items).Error
	return items, err
}

func (r *EvalRepository) CreateReviews(ctx context.Context, reviews []model.SysAiEvalReview) error {
	if len(reviews) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Create(&reviews).Error
}

func (r *EvalRepository) UpdateReviewFields(ctx context.Context, id int64, fields map[string]any) error {
	fields["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiEvalReview{}).
		Where("id = ?", id).Updates(fields).Error
}
