package aidomain

import (
	"context"
	"errors"

	"github.com/earthyzinc/clarity-go/internal/model"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// MessageRepository AI 消息数据访问。
type MessageRepository struct {
	db *gorm.DB
}

func NewMessageRepository(db *gorm.DB) *MessageRepository {
	return &MessageRepository{db: db}
}

// Create 插入消息。blocks 零值由 model 的 default:'[]' 兜底为合法 JSON 数组（列 NOT NULL）。
func (r *MessageRepository) Create(ctx context.Context, msg *model.SysAiMessage) error {
	return r.db.WithContext(ctx).Create(msg).Error
}

// UpdateFields 按列更新消息（状态、块、计划、待恢复中断等增量写入）。
// 走 map 更新：JSON 列的值须由调用方给出合法的 JSON 文本或 nil（落 SQL NULL）。
func (r *MessageRepository) UpdateFields(ctx context.Context, id int64, fields map[string]any) error {
	if len(fields) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Where("id = ? AND deleted = 0", id).Updates(fields).Error
}

func (r *MessageRepository) GetByID(ctx context.Context, id int64) (*model.SysAiMessage, error) {
	var msg model.SysAiMessage
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&msg).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &msg, err
}

func (r *MessageRepository) GetByIDAndUser(ctx context.Context, id, userID int64) (*model.SysAiMessage, error) {
	var msg model.SysAiMessage
	err := r.db.WithContext(ctx).Raw(
		"SELECT m.* FROM sys_ai_message m JOIN sys_ai_conversation c ON c.id = m.conversation_id "+
			"WHERE m.id = ? AND m.deleted = 0 AND c.user_id = ? AND c.deleted = 0", id, userID).
		Scan(&msg).Error
	if err != nil {
		return nil, err
	}
	if msg.ID == 0 {
		return nil, nil
	}
	return &msg, nil
}

// ListByConversationCursor 会话消息游标分页（id 倒序等价时间倒序）：
//   - 返回 total＝该会话消息总数（不受 before 影响，供前端展示总数）；
//   - before 非空时只取 id < before，否则取最新一页；
//   - 多取一条（limit+1）供调用方判定 hasMore，无需额外一次存在性往返。
func (r *MessageRepository) ListByConversationCursor(ctx context.Context, convID int64, before *int64, limit int) ([]model.SysAiMessage, int64, error) {
	base := r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Where("conversation_id = ? AND deleted = 0", convID)
	var total int64
	if err := base.Session(&gorm.Session{}).Count(&total).Error; err != nil {
		return nil, 0, err
	}
	query := base.Order("id DESC")
	if before != nil {
		query = query.Where("id < ?", *before)
	}
	var messages []model.SysAiMessage
	if err := query.Limit(limit + 1).Find(&messages).Error; err != nil {
		return nil, 0, err
	}
	return messages, total, nil
}

// ListAssistantSiblingIDs 批量取指定父消息下未删 assistant 兄弟 ID：{parent_message_id: [id 升序]}。
// 一次 IN 查询按父消息分组，避免逐条消息各查一次兄弟（N+1）。siblingIndex 即列表下标 + 1，
// siblingCount 即列表长度；root 级（parent_message_id IS NULL）助手消息不参与，调用方相应字段保持 null。
func (r *MessageRepository) ListAssistantSiblingIDs(ctx context.Context, convID int64, parentIDs []int64) (map[int64][]int64, error) {
	grouped := make(map[int64][]int64)
	if len(parentIDs) == 0 {
		return grouped, nil
	}
	var rows []struct {
		ParentMessageID int64 `gorm:"column:parent_message_id"`
		ID              int64 `gorm:"column:id"`
	}
	err := r.db.WithContext(ctx).Table("sys_ai_message").
		Select("parent_message_id, id").
		Where("conversation_id = ? AND parent_message_id IN ? AND role = ? AND deleted = 0",
			convID, parentIDs, "assistant").
		Order("parent_message_id ASC, id ASC").
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		grouped[row.ParentMessageID] = append(grouped[row.ParentMessageID], row.ID)
	}
	return grouped, nil
}

// GetChildren 某消息的分支列表（时间倒序）。
//
// 只返回 role=assistant 的子消息：同一 parent 下还可能挂着用户后续提问（删尾后继续发送，
// 新 user 消息会以旧尾的 parent 为 parent），它们不是可切换的兄弟分支，混进来会让前端
// 分支序号与 ListAssistantSiblingIDs 的口径错位。
func (r *MessageRepository) GetChildren(ctx context.Context, convID, parentID int64) ([]model.SysAiMessage, error) {
	var msgs []model.SysAiMessage
	err := r.db.WithContext(ctx).
		Where("conversation_id = ? AND parent_message_id = ? AND role = ? AND deleted = 0",
			convID, parentID, "assistant").
		Order("id DESC").Find(&msgs).Error
	return msgs, err
}

// GetChainByID 沿 parent 链回溯到根（按 id 正序返回，供会话导出）。
func (r *MessageRepository) GetChainByID(ctx context.Context, convID, tailID int64) ([]model.SysAiMessage, error) {
	var all []model.SysAiMessage
	if err := r.db.WithContext(ctx).
		Where("conversation_id = ? AND deleted = 0", convID).
		Order("id ASC").Find(&all).Error; err != nil {
		return nil, err
	}
	byID := make(map[int64]model.SysAiMessage, len(all))
	for _, m := range all {
		byID[m.ID] = m
	}
	var chain []model.SysAiMessage
	visited := make(map[int64]struct{})
	for cur := tailID; cur != 0; {
		if _, seen := visited[cur]; seen {
			break
		}
		visited[cur] = struct{}{}
		msg, ok := byID[cur]
		if !ok {
			break
		}
		chain = append(chain, msg)
		if msg.ParentMessageID == nil {
			break
		}
		cur = *msg.ParentMessageID
	}
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}
	return chain, nil
}

func (r *MessageRepository) CountMessagesAfter(ctx context.Context, convID, msgID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Where("conversation_id = ? AND deleted = 0 AND id > ?", convID, msgID).
		Count(&count).Error
	return count, err
}

func (r *MessageRepository) GetLastMessageID(ctx context.Context, convID int64) (*int64, error) {
	var row struct {
		MaxID *int64 `gorm:"column:max_id"`
	}
	err := r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Select("MAX(id) AS max_id").
		Where("conversation_id = ? AND deleted = 0", convID).
		Scan(&row).Error
	return row.MaxID, err
}

func (r *MessageRepository) SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Where("id IN ? AND deleted = 0", ids).
		Updates(map[string]any{"deleted": gorm.Expr("id"), "update_by": updateBy}).Error
}

// DeleteLeafOutcome 助手消息删除的落地结果，供调用方映射业务错误。
type DeleteLeafOutcome int

const (
	// DeleteLeafDone 已软删（指针是否回退由 CAS 是否命中决定）。
	DeleteLeafDone DeleteLeafOutcome = iota
	// DeleteLeafNotFound 会话或消息已不存在（含并发软删）。
	DeleteLeafNotFound
	// DeleteLeafHasChildren 仍有未删子消息，拒绝删除。
	DeleteLeafHasChildren
)

// DeleteAssistantLeaf 事务内删除助手叶子消息，返回落地结果。
//
// 会话行上的排他锁（SELECT ... FOR UPDATE）是该会话写路径的串行点：children 检查、软删、
// 指针 CAS 全部落在同一临界区，避免"检查时仍是叶子、写入时已长出子消息"把后续对话的父链
// 删断，也避免回退旧指针覆盖并发 send 刚推进的新尾。锁用 DB 行锁而非 Redis：Redis 锁在
// 进程外失效（过期/丢键）就无法保证正确性，而这两步写入本就在同一事务里。
//
// 指针更新带 CAS（current_branch_message_id = msgID）：仅当指针仍指向被删消息时回退到
// parent；并发 send 已推进指针时条件不命中，指针保持不动。非叶子/消息消失时不写任何数据，
// 事务空提交返回对应结果。
func (r *MessageRepository) DeleteAssistantLeaf(ctx context.Context, convID, msgID, updateBy int64) (DeleteLeafOutcome, error) {
	outcome := DeleteLeafNotFound
	err := r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var conv model.SysAiConversation
		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Select("id").Where("id = ? AND deleted = 0", convID).Take(&conv).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil
		}
		if err != nil {
			return err
		}
		var msg model.SysAiMessage
		err = tx.Where("id = ? AND conversation_id = ? AND deleted = 0", msgID, convID).Take(&msg).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil
		}
		if err != nil {
			return err
		}
		var children int64
		if err := tx.Model(&model.SysAiMessage{}).
			Where("conversation_id = ? AND parent_message_id = ? AND deleted = 0", convID, msgID).
			Count(&children).Error; err != nil {
			return err
		}
		if children > 0 {
			outcome = DeleteLeafHasChildren
			return nil
		}
		if err := tx.Model(&model.SysAiMessage{}).
			Where("id = ? AND deleted = 0", msgID).
			Updates(map[string]any{"deleted": gorm.Expr("id"), "update_by": updateBy}).Error; err != nil {
			return err
		}
		if err := tx.Model(&model.SysAiConversation{}).
			Where("id = ? AND current_branch_message_id = ?", convID, msgID).
			Updates(map[string]any{"current_branch_message_id": msg.ParentMessageID, "update_by": updateBy}).Error; err != nil {
			return err
		}
		outcome = DeleteLeafDone
		return nil
	})
	if err != nil {
		return DeleteLeafNotFound, err
	}
	return outcome, nil
}

// ListAnomalyStatusByConversations 按会话汇总异常消息状态（failed/canceled）。
// status 为字符串枚举，异常口径只看这两种终态。
func (r *MessageRepository) ListAnomalyStatusByConversations(ctx context.Context, convIDs []int64) (map[int64]map[string]struct{}, error) {
	result := make(map[int64]map[string]struct{})
	if len(convIDs) == 0 {
		return result, nil
	}
	type row struct {
		ConversationID int64  `gorm:"column:conversation_id"`
		Status         string `gorm:"column:status"`
	}
	var rows []row
	err := r.db.WithContext(ctx).Table("sys_ai_message").
		Select("conversation_id, status").
		Where("conversation_id IN ? AND status IN ? AND deleted = 0", convIDs, []string{"failed", "canceled"}).
		Group("conversation_id, status").
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, item := range rows {
		if result[item.ConversationID] == nil {
			result[item.ConversationID] = make(map[string]struct{})
		}
		result[item.ConversationID][item.Status] = struct{}{}
	}
	return result, nil
}

// ThoughtRepository 推理步骤数据访问。
type ThoughtRepository struct {
	db *gorm.DB
}

func NewThoughtRepository(db *gorm.DB) *ThoughtRepository {
	return &ThoughtRepository{db: db}
}

func (r *ThoughtRepository) ListByMessage(ctx context.Context, messageID int64) ([]model.SysAiAgentThought, error) {
	var items []model.SysAiAgentThought
	err := r.db.WithContext(ctx).
		Where("message_id = ?", messageID).
		Order("position ASC").Find(&items).Error
	return items, err
}

// ListByMessages 批量取多条消息的推理步骤（position 正序），避免逐消息 N+1。
func (r *ThoughtRepository) ListByMessages(ctx context.Context, messageIDs []int64) (map[int64][]model.SysAiAgentThought, error) {
	result := make(map[int64][]model.SysAiAgentThought)
	if len(messageIDs) == 0 {
		return result, nil
	}
	var items []model.SysAiAgentThought
	err := r.db.WithContext(ctx).
		Where("message_id IN ?", messageIDs).
		Order("message_id ASC, position ASC").Find(&items).Error
	if err != nil {
		return nil, err
	}
	for _, item := range items {
		result[item.MessageID] = append(result[item.MessageID], item)
	}
	return result, nil
}
