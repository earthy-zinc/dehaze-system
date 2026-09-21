package ai

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// A2ARepository A2A 协议（Agent Card）数据访问
type A2ARepository struct {
	db *gorm.DB
}

func NewA2ARepository(db *gorm.DB) *A2ARepository {
	return &A2ARepository{db: db}
}

func (r *A2ARepository) GetAgent(ctx context.Context, agentID int64) (*model.SysAiAgent, error) {
	var agent model.SysAiAgent
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", agentID).First(&agent).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &agent, err
}

// LatestPublishedVersionNo 取当前已发布版本号（同一 Agent 同一时刻至多一条 status=2）
func (r *A2ARepository) LatestPublishedVersionNo(ctx context.Context, agentID int64) (int, bool, error) {
	var versionNo *int
	err := r.db.WithContext(ctx).Model(&model.SysAiAgentVersion{}).
		Where("agent_id = ? AND status = 2", agentID).
		Order("version_no DESC").Limit(1).
		Pluck("version_no", &versionNo).Error
	if err != nil {
		return 0, false, err
	}
	if versionNo == nil {
		return 0, false, nil
	}
	return *versionNo, true, nil
}
