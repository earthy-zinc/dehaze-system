package aidomain

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// ConversationRepository AI 会话数据访问。
type ConversationRepository struct {
	db *gorm.DB
}

func NewConversationRepository(db *gorm.DB) *ConversationRepository {
	return &ConversationRepository{db: db}
}

func (r *ConversationRepository) Create(ctx context.Context, conv *model.SysAiConversation) error {
	return r.db.WithContext(ctx).Create(conv).Error
}

func (r *ConversationRepository) GetByID(ctx context.Context, id int64) (*model.SysAiConversation, error) {
	var conv model.SysAiConversation
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&conv).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &conv, err
}

func (r *ConversationRepository) GetByIDAndUser(ctx context.Context, id, userID int64) (*model.SysAiConversation, error) {
	var conv model.SysAiConversation
	err := r.db.WithContext(ctx).
		Where("id = ? AND user_id = ? AND deleted = 0", id, userID).
		First(&conv).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &conv, err
}

// GetInTrash 取回收站中未超恢复窗口的会话（软删行，需绕过 deleted=0 过滤）。
// 必须 Unscoped：全局软删回调（pkg/database/soft_delete.go）会给所有含 Deleted 字段的模型
// 追加 deleted = 0，与这里的 deleted <> 0 互斥，不加则永久查不到软删行（恢复必报"已超出恢复窗口"）。
func (r *ConversationRepository) GetInTrash(ctx context.Context, id, userID int64, windowStart time.Time) (*model.SysAiConversation, error) {
	var conv model.SysAiConversation
	err := r.db.WithContext(ctx).Unscoped().
		Where("id = ? AND user_id = ? AND deleted <> 0 AND delete_time >= ?", id, userID, windowStart).
		First(&conv).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &conv, err
}

// PaginateUser 用户视角会话分页（置顶优先 → 置顶时间 → 最后消息时间 → id 倒序）。
func (r *ConversationRepository) PaginateUser(ctx context.Context, userID int64, page, size int, status *int) ([]model.SysAiConversation, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("user_id = ? AND deleted = 0", userID)
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	db = db.Order("pinned DESC, pinned_at DESC, last_message_at DESC, id DESC")
	return countAndFind[model.SysAiConversation](db, page, size)
}

// PaginateAll 管理端审计视角：全量用户会话分页。
func (r *ConversationRepository) PaginateAll(ctx context.Context, page, size int, keyword string, status *int) ([]model.SysAiConversation, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiConversation{}).Where("deleted = 0")
	if keyword != "" {
		db = db.Where("title LIKE ? ESCAPE '\\\\'", "%"+escapeLike(keyword)+"%")
	}
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	db = db.Order("pinned DESC, pinned_at DESC, last_message_at DESC, id DESC")
	return countAndFind[model.SysAiConversation](db, page, size)
}

// PaginateTrash 回收站分页（软删行，delete_time 倒序）。
// 同 GetInTrash：需 Unscoped 绕过全局软删回调追加的 deleted = 0。
func (r *ConversationRepository) PaginateTrash(ctx context.Context, userID int64, page, size int, windowStart time.Time) ([]model.SysAiConversation, int64, error) {
	db := r.db.WithContext(ctx).Unscoped().Model(&model.SysAiConversation{}).
		Where("user_id = ? AND deleted <> 0 AND delete_time >= ?", userID, windowStart).
		Order("delete_time DESC, id DESC")
	return countAndFind[model.SysAiConversation](db, page, size)
}

// GetByIDs 按 ID 列表查询（保持传入顺序）。
func (r *ConversationRepository) GetByIDs(ctx context.Context, userID int64, ids []int64) ([]model.SysAiConversation, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var convs []model.SysAiConversation
	if err := r.db.WithContext(ctx).
		Where("id IN ? AND user_id = ? AND deleted = 0", ids, userID).
		Find(&convs).Error; err != nil {
		return nil, err
	}
	order := make(map[int64]int, len(ids))
	for i, id := range ids {
		order[id] = i
	}
	for i := 1; i < len(convs); i++ {
		for j := i; j > 0 && order[convs[j].ID] < order[convs[j-1].ID]; j-- {
			convs[j], convs[j-1] = convs[j-1], convs[j]
		}
	}
	return convs, nil
}

func (r *ConversationRepository) UpdateFields(ctx context.Context, id int64, fields map[string]any) error {
	return r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id = ?", id).Updates(fields).Error
}

// SetPinned 置顶/取消置顶（pinned_at 为 nil 表示取消）。
func (r *ConversationRepository) SetPinned(ctx context.Context, id int64, pinned int, pinnedAt *time.Time) error {
	return r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id = ?", id).
		Updates(map[string]any{"pinned": pinned, "pinned_at": pinnedAt}).Error
}

func (r *ConversationRepository) UpdateStatusByIDs(ctx context.Context, ids []int64, status int, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id IN ?", ids).
		Updates(map[string]any{"status": status, "update_by": updateBy}).Error
}

func (r *ConversationRepository) MarkRead(ctx context.Context, id, messageID, userID int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id = ?", id).
		Updates(map[string]any{"last_read_message_id": messageID, "update_by": userID}).Error
}

func (r *ConversationRepository) UpdateCurrentBranch(ctx context.Context, id, messageID, userID int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id = ?", id).
		Updates(map[string]any{"current_branch_message_id": messageID, "update_by": userID}).Error
}

// SoftDeleteByIDs 软删（deleted=id + delete_time）。
// delete_time 截断到秒：列为 DATETIME（秒精度），直接写 time.Now() 会被 MySQL 进位成下一刻，
// 与接口回显/窗口比较出现亚秒偏差（与 create_time 同类的精度问题）。
func (r *ConversationRepository) SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id IN ? AND deleted = 0", ids).
		Updates(map[string]any{
			"deleted":     gorm.Expr("id"),
			"delete_time": time.Now().Truncate(time.Second),
			"update_by":   updateBy,
		}).Error
}

func (r *ConversationRepository) RestoreByIDs(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id IN ? AND deleted <> 0", ids).
		Updates(map[string]any{"deleted": 0, "delete_time": nil, "update_by": updateBy}).Error
}

func (r *ConversationRepository) CountActivePinned(ctx context.Context, userID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("user_id = ? AND deleted = 0 AND pinned = 1", userID).
		Count(&count).Error
	return count, err
}

// UpdateLastMessage 更新最后消息指针并累加 message_count。
func (r *ConversationRepository) UpdateLastMessage(ctx context.Context, id, messageID int64, at time.Time) error {
	return r.db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id = ?", id).
		Updates(map[string]any{
			"last_message_at":           at,
			"current_branch_message_id": messageID,
			"message_count":             gorm.Expr("message_count + 1"),
		}).Error
}

// ConsumptionRow 会话计费聚合行。
type ConsumptionRow struct {
	ConversationID int64 `gorm:"column:conversation_id"`
	Token          int64 `gorm:"column:token"`
	Credits        int64 `gorm:"column:credits"`
}

// SumConsumptionByConversation 按会话聚合 Token/积分（只读 sys_ai_billing）。
func (r *ConversationRepository) SumConsumptionByConversation(ctx context.Context, convIDs []int64) (map[int64]ConsumptionRow, error) {
	result := make(map[int64]ConsumptionRow)
	if len(convIDs) == 0 {
		return result, nil
	}
	var rows []ConsumptionRow
	err := r.db.WithContext(ctx).Table("sys_ai_billing").
		Select("conversation_id, COALESCE(SUM(input_tokens + output_tokens), 0) AS token, COALESCE(SUM(credits), 0) AS credits").
		Where("conversation_id IN ?", convIDs).
		Group("conversation_id").
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		if row.ConversationID != 0 {
			result[row.ConversationID] = row
		}
	}
	return result, nil
}

// ListQuotaAnomalyConversationIDs 存在"连续配额不足"异常的会话 ID 集合。
func (r *ConversationRepository) ListQuotaAnomalyConversationIDs(ctx context.Context, convIDs []int64) (map[int64]struct{}, error) {
	result := make(map[int64]struct{})
	if len(convIDs) == 0 {
		return result, nil
	}
	var ids []int64
	err := r.db.WithContext(ctx).Table("sys_ai_billing b").
		Select("b.conversation_id").
		Joins("JOIN sys_ai_billing_anomaly a ON a.billing_id = b.id").
		Where("b.conversation_id IN ? AND a.anomaly_type = ?", convIDs, "consecutive_quota_fail").
		Distinct("b.conversation_id").
		Pluck("b.conversation_id", &ids).Error
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		result[id] = struct{}{}
	}
	return result, nil
}

// ListRiskyToolConversationIDs 存在高风险工具调用（工具调用失败/超时）的会话 ID 集合。
func (r *ConversationRepository) ListRiskyToolConversationIDs(ctx context.Context, convIDs []int64) (map[int64]struct{}, error) {
	result := make(map[int64]struct{})
	if len(convIDs) == 0 {
		return result, nil
	}
	var ids []int64
	err := r.db.WithContext(ctx).Table("sys_ai_trace t").
		Select("t.conversation_id").
		Joins("JOIN sys_ai_llm_call c ON c.trace_id = t.trace_id").
		Where("t.conversation_id IN ? AND c.tool_call IS NOT NULL AND c.status <> 1", convIDs).
		Distinct("t.conversation_id").
		Pluck("t.conversation_id", &ids).Error
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		result[id] = struct{}{}
	}
	return result, nil
}

// MatchMessageRow 会话内消息关键词命中的最新消息。
type MatchMessageRow struct {
	ConversationID int64  `gorm:"column:conversation_id"`
	ID             int64  `gorm:"column:id"`
	Content        string `gorm:"column:content"`
}

// ListMatchedMessageIDs 返回各会话中内容命中关键词的最新消息（供审计视角回填 matchedMessageId）。
func (r *ConversationRepository) ListMatchedMessageIDs(ctx context.Context, convIDs []int64, keyword string) (map[int64]int64, error) {
	result := make(map[int64]int64)
	if len(convIDs) == 0 || keyword == "" {
		return result, nil
	}
	var rows []MatchMessageRow
	err := r.db.WithContext(ctx).Table("sys_ai_message").
		Select("conversation_id, MAX(id) AS id").
		Where("conversation_id IN ? AND deleted = 0 AND content LIKE ? ESCAPE '\\\\'", convIDs, "%"+escapeLike(keyword)+"%").
		Group("conversation_id").
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		result[row.ConversationID] = row.ID
	}
	return result, nil
}

// UserNameRow 用户名查询行。
type UserNameRow struct {
	ID       int64  `gorm:"column:id"`
	Username string `gorm:"column:username"`
}

// ListUserNames 批量取用户显示名（审计视角）。
func (r *ConversationRepository) ListUserNames(ctx context.Context, userIDs []int64) (map[int64]string, error) {
	result := make(map[int64]string)
	if len(userIDs) == 0 {
		return result, nil
	}
	var rows []UserNameRow
	err := r.db.WithContext(ctx).Table("sys_user").
		Select("id, COALESCE(nickname, username) AS username").
		Where("id IN ?", userIDs).
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		result[row.ID] = row.Username
	}
	return result, nil
}
