package aidomain

import (
	"context"
	"encoding/json"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// MessageVO 消息响应（含推理步骤）。
type MessageVO struct {
	ID                int64            `json:"id"`
	ConversationID    int64            `json:"conversationId"`
	ParentMessageID   *int64           `json:"parentMessageId,omitempty"`
	Role              string           `json:"role"`
	Content           string           `json:"content,omitempty"`
	ToolCalls         json.RawMessage  `json:"toolCalls,omitempty"`
	ToolCallID        string           `json:"toolCallId,omitempty"`
	Model             string           `json:"model,omitempty"`
	Status            int              `json:"status"`
	Error             string           `json:"error,omitempty"`
	Metadata          json.RawMessage  `json:"metadata,omitempty"`
	InputTokens       int              `json:"inputTokens"`
	OutputTokens      int              `json:"outputTokens"`
	CachedInputTokens int              `json:"cachedInputTokens"`
	Credits           int64            `json:"credits"`
	TaskID            string           `json:"taskId,omitempty"`
	Edited            int              `json:"edited"`
	OriginalContent   string           `json:"originalContent,omitempty"`
	UsedMemoryIDs     json.RawMessage  `json:"usedMemoryIds,omitempty"`
	CreateTime        string           `json:"createTime,omitempty"`
	Thoughts          []AgentThoughtVO `json:"thoughts,omitempty"`
}

// AgentThoughtVO 推理步骤响应。
type AgentThoughtVO struct {
	ID             int64           `json:"id"`
	MessageID      int64           `json:"messageId"`
	ConversationID int64           `json:"conversationId"`
	Position       int             `json:"position"`
	AgentCode      string          `json:"agentCode,omitempty"`
	IsSubagent     int             `json:"isSubagent"`
	Thought        string          `json:"thought,omitempty"`
	Tool           string          `json:"tool,omitempty"`
	ToolInput      json.RawMessage `json:"toolInput,omitempty"`
	Observation    string          `json:"observation,omitempty"`
	Status         int             `json:"status"`
	LatencyMs      int             `json:"latencyMs"`
	Error          string          `json:"error,omitempty"`
	CreateTime     string          `json:"createTime,omitempty"`
}

// MessageService 消息查询与分支管理（非推理类）。
type MessageService struct {
	conversations *repo.ConversationRepository
	messages      *repo.MessageRepository
	thoughts      *repo.ThoughtRepository
}

func NewMessageService(
	conversations *repo.ConversationRepository,
	messages *repo.MessageRepository,
	thoughts *repo.ThoughtRepository,
) *MessageService {
	return &MessageService{conversations: conversations, messages: messages, thoughts: thoughts}
}

// List 会话消息游标分页（id 倒序，before 缺省取最新一页）。
// hasMore 用 limit+1 探测：仓储多取一条，命中即说明还有更早消息，裁掉后不回填，
// 单次查询即可判定，避免额外 COUNT 存在性往返。assistant 消息批量附带推理步骤。
func (s *MessageService) List(ctx context.Context, convID, userID int64, before *int64, limit int, admin bool) (*vo.CursorResult[MessageVO], error) {
	if err := s.ensureConversation(ctx, convID, userID, admin); err != nil {
		return nil, err
	}
	msgs, total, err := s.messages.ListByConversationCursor(ctx, convID, before, limit)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息列表失败", err)
	}
	hasMore := len(msgs) > limit
	if hasMore {
		msgs = msgs[:limit]
	}
	assistantIDs := make([]int64, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == "assistant" {
			assistantIDs = append(assistantIDs, m.ID)
		}
	}
	thoughtsMap, err := s.thoughts.ListByMessages(ctx, assistantIDs)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询推理步骤失败", err)
	}
	items := make([]MessageVO, 0, len(msgs))
	for i := range msgs {
		item := s.toVO(&msgs[i])
		item.Thoughts = toThoughtVOs(thoughtsMap[msgs[i].ID])
		items = append(items, *item)
	}
	return &vo.CursorResult[MessageVO]{List: items, Total: total, HasMore: hasMore}, nil
}

// Get 消息详情（含推理步骤）。
func (s *MessageService) Get(ctx context.Context, msgID, userID int64, admin bool) (*MessageVO, error) {
	var (
		msg *model.SysAiMessage
		err error
	)
	if admin {
		msg, err = s.messages.GetByID(ctx, msgID)
	} else {
		msg, err = s.messages.GetByIDAndUser(ctx, msgID, userID)
	}
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
	}
	if msg == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	}
	thoughts, err := s.thoughts.ListByMessage(ctx, msgID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询推理步骤失败", err)
	}
	result := s.toVO(msg)
	result.Thoughts = toThoughtVOs(thoughts)
	return result, nil
}

// Delete 删除助手消息（软删）。
func (s *MessageService) Delete(ctx context.Context, msgID, userID int64) error {
	msg, err := s.messages.GetByIDAndUser(ctx, msgID, userID)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
	}
	if msg == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	}
	if msg.Role != "assistant" {
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "仅助手消息可删除")
	}
	if err := s.messages.SoftDeleteByIDs(ctx, []int64{msgID}, userID); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除消息失败", err)
	}
	return nil
}

// GetBranches 查询消息的分支列表（时间倒序）。
func (s *MessageService) GetBranches(ctx context.Context, convID, msgID, userID int64) ([]MessageVO, error) {
	if err := s.ensureConversation(ctx, convID, userID, false); err != nil {
		return nil, err
	}
	msg, err := s.messages.GetByIDAndUser(ctx, msgID, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
	}
	if msg == nil || msg.ConversationID != convID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	}
	children, err := s.messages.GetChildren(ctx, convID, msgID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询分支失败", err)
	}
	items := make([]MessageVO, 0, len(children))
	for i := range children {
		items = append(items, *s.toVO(&children[i]))
	}
	return items, nil
}

// SwitchBranch 切换当前激活分支。
func (s *MessageService) SwitchBranch(ctx context.Context, convID, msgID, userID int64) (*ConversationVO, error) {
	conv, err := s.conversations.GetByIDAndUser(ctx, convID, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	msg, err := s.messages.GetByIDAndUser(ctx, msgID, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
	}
	if msg == nil || msg.ConversationID != convID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	}
	if err := s.conversations.UpdateCurrentBranch(ctx, convID, msgID, userID); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "切换分支失败", err)
	}
	conv.CurrentBranchMessageID = &msgID
	if conv.LastReadMessageID != nil {
		unread, countErr := s.messages.CountMessagesAfter(ctx, convID, *conv.LastReadMessageID)
		if countErr != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "统计未读消息失败", countErr)
		}
		result := s.toConversationVO(conv)
		result.UnreadCount = unread
		return result, nil
	}
	result := s.toConversationVO(conv)
	result.UnreadCount = int64(conv.MessageCount)
	return result, nil
}

func (s *MessageService) ensureConversation(ctx context.Context, convID, userID int64, admin bool) error {
	var (
		conv *model.SysAiConversation
		err  error
	)
	if admin {
		conv, err = s.conversations.GetByID(ctx, convID)
	} else {
		conv, err = s.conversations.GetByIDAndUser(ctx, convID, userID)
	}
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	return nil
}

// toConversationVO 复用会话 VO 构造（未读数由调用方按已读水位计算）。
func (s *MessageService) toConversationVO(conv *model.SysAiConversation) *ConversationVO {
	return &ConversationVO{
		ID:                     conv.ID,
		UserID:                 conv.UserID,
		Title:                  conv.Title,
		Model:                  derefString(conv.Model),
		AgentCode:              derefString(conv.AgentCode),
		AgentVersion:           conv.AgentVersion,
		Summary:                derefString(conv.Summary),
		SystemPrompt:           derefString(conv.SystemPrompt),
		ModelConfig:            rawJSON(conv.ModelConfig),
		SuggestionsEnabled:     conv.SuggestionsEnabled,
		APIKeyID:               conv.APIKeyID,
		MessageCount:           conv.MessageCount,
		LastMessageAt:          formatTimePtr(conv.LastMessageAt),
		CurrentBranchMessageID: conv.CurrentBranchMessageID,
		LastReadMessageID:      conv.LastReadMessageID,
		Pinned:                 conv.Pinned,
		PinnedAt:               formatTimePtr(conv.PinnedAt),
		DeleteTime:             formatTimePtr(conv.DeleteTime),
		TitleSource:            conv.TitleSource,
		Status:                 conv.Status,
		CreateTime:             formatTime(conv.CreateTime),
		UpdateTime:             formatTimePtr(conv.UpdateTime),
	}
}

func (s *MessageService) toVO(msg *model.SysAiMessage) *MessageVO {
	return &MessageVO{
		ID:                msg.ID,
		ConversationID:    msg.ConversationID,
		ParentMessageID:   msg.ParentMessageID,
		Role:              msg.Role,
		Content:           derefString(msg.Content),
		ToolCalls:         rawJSON(msg.ToolCalls),
		ToolCallID:        derefString(msg.ToolCallID),
		Model:             derefString(msg.Model),
		Status:            msg.Status,
		Error:             derefString(msg.Error),
		Metadata:          rawJSON(msg.Metadata),
		InputTokens:       msg.InputTokens,
		OutputTokens:      msg.OutputTokens,
		CachedInputTokens: msg.CachedInputTokens,
		Credits:           msg.Credits,
		TaskID:            derefString(msg.TaskID),
		Edited:            msg.Edited,
		OriginalContent:   derefString(msg.OriginalContent),
		UsedMemoryIDs:     rawJSON(msg.UsedMemoryIDs),
		CreateTime:        formatTime(msg.CreateTime),
	}
}

func toThoughtVOs(items []model.SysAiAgentThought) []AgentThoughtVO {
	if len(items) == 0 {
		return nil
	}
	result := make([]AgentThoughtVO, 0, len(items))
	for _, item := range items {
		latency := 0
		if item.LatencyMs != nil {
			latency = *item.LatencyMs
		}
		result = append(result, AgentThoughtVO{
			ID:             item.ID,
			MessageID:      item.MessageID,
			ConversationID: item.ConversationID,
			Position:       item.Position,
			AgentCode:      derefString(item.AgentCode),
			IsSubagent:     item.IsSubagent,
			Thought:        derefString(item.Thought),
			Tool:           derefString(item.Tool),
			ToolInput:      rawJSON(item.ToolInput),
			Observation:    derefString(item.Observation),
			Status:         item.Status,
			LatencyMs:      latency,
			Error:          derefString(item.Error),
			CreateTime:     formatTime(item.CreateTime),
		})
	}
	return result
}
