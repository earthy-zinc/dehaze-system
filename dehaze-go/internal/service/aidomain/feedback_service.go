package aidomain

import (
	"context"
	"encoding/json"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// feedbackValidDays 反馈时效（天）。
const feedbackValidDays = 30

var likeTags = map[string]struct{}{"accurate": {}, "detailed": {}, "concise": {}, "creative": {}}
var dislikeTags = map[string]struct{}{
	"incorrect": {}, "irrelevant": {}, "incomplete": {}, "too_long": {}, "bad_citation": {}, "harmful": {},
}

// dislikePreferenceMemories 点踩标签 → 用户偏好语义记忆。
var dislikePreferenceMemories = map[string]string{
	"too_long":   "用户偏好简洁回复",
	"incomplete": "用户期望回复完整、覆盖全部要点",
	"irrelevant": "用户期望回复紧扣主题、避免无关内容",
}

// FeedbackVO 消息反馈响应。
type FeedbackVO struct {
	ID         int64           `json:"id"`
	MessageID  int64           `json:"messageId"`
	UserID     int64           `json:"userId"`
	Rating     int             `json:"rating"`
	Tags       json.RawMessage `json:"tags,omitempty"`
	Comment    string          `json:"comment,omitempty"`
	CreateTime string          `json:"createTime,omitempty"`
	UpdateTime string          `json:"updateTime,omitempty"`
}

// FeedbackCreateForm 提交反馈表单。
type FeedbackCreateForm struct {
	Rating  int      `json:"rating" binding:"required,oneof=1 -1"`
	Tags    []string `json:"tags"`
	Comment *string  `json:"comment" binding:"omitempty,max=2000"`
}

// FeedbackService 消息反馈业务逻辑。
type FeedbackService struct {
	messages  *repo.MessageRepository
	feedbacks *repo.MessageFeedbackRepository
	memories  *repo.MemoryRepository
}

func NewFeedbackService(
	messages *repo.MessageRepository,
	feedbacks *repo.MessageFeedbackRepository,
	memories *repo.MemoryRepository,
) *FeedbackService {
	return &FeedbackService{messages: messages, feedbacks: feedbacks, memories: memories}
}

// Submit 提交/更新反馈（同一用户同一消息唯一）。
func (s *FeedbackService) Submit(ctx context.Context, messageID, userID int64, form *FeedbackCreateForm) (*FeedbackVO, error) {
	msg, err := s.messages.GetByIDAndUser(ctx, messageID, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
	}
	if msg == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	}
	if msg.Role != "assistant" {
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "仅助手消息可反馈")
	}
	if time.Since(msg.CreateTime) > feedbackValidDays*24*time.Hour {
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "已超过反馈时效(30天)")
	}
	if form.Rating == 1 {
		for _, tag := range form.Tags {
			if _, ok := likeTags[tag]; !ok {
				return nil, common.NewBizError(common.PARAM_ERROR, "不支持的标签类型")
			}
		}
	} else {
		if len(form.Tags) == 0 {
			return nil, common.NewBizError(common.PARAM_ERROR, "点踩必须选择问题标签")
		}
		for _, tag := range form.Tags {
			if _, ok := dislikeTags[tag]; !ok {
				return nil, common.NewBizError(common.PARAM_ERROR, "不支持的标签类型")
			}
		}
	}

	conversationID := msg.ConversationID
	feedback := &model.SysAiMessageFeedback{
		MessageID:      messageID,
		UserID:         userID,
		ConversationID: &conversationID,
		Model:          msg.Model,
		Source:         "internal",
		Rating:         form.Rating,
		Tags:           marshalJSON(form.Tags),
		Comment:        form.Comment,
	}
	saved, err := s.feedbacks.Upsert(ctx, feedback)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "提交反馈失败", err)
	}
	if form.Rating != 1 {
		s.spawnPreferenceMemory(userID, form.Tags, form.Comment)
	}
	return toFeedbackVO(saved), nil
}

// Get 查询反馈状态（无反馈返回 nil）。
func (s *FeedbackService) Get(ctx context.Context, messageID, userID int64) (*FeedbackVO, error) {
	feedback, err := s.feedbacks.GetByUserAndMessage(ctx, messageID, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询反馈失败", err)
	}
	if feedback == nil {
		return nil, nil
	}
	return toFeedbackVO(feedback), nil
}

// Revoke 撤销反馈（软删）。
func (s *FeedbackService) Revoke(ctx context.Context, messageID, userID int64) error {
	feedback, err := s.feedbacks.GetByUserAndMessage(ctx, messageID, userID)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询反馈失败", err)
	}
	if feedback == nil {
		return common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}
	if err := s.feedbacks.SoftDelete(ctx, messageID, userID); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "撤销反馈失败", err)
	}
	return nil
}

// spawnPreferenceMemory 点踩标签异步沉淀为用户偏好语义记忆，不阻塞反馈提交。
func (s *FeedbackService) spawnPreferenceMemory(userID int64, tags []string, comment *string) {
	content := ""
	for _, tag := range tags {
		if mapped, ok := dislikePreferenceMemories[tag]; ok {
			content = mapped
			break
		}
	}
	if content == "" {
		return
	}
	if comment != nil && *comment != "" {
		content = content + "（用户补充：" + *comment + "）"
	}
	go func() {
		defer func() {
			if r := recover(); r != nil {
				logger.Warn("反馈记忆沉淀 panic", zap.Any("recover", r))
			}
		}()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		memory := &model.SysAiMemory{
			UserID:     userID,
			MemoryType: "semantic",
			Content:    content,
			Metadata:   marshalJSON(map[string]any{"category": "preference", "is_preference": 1}),
			Importance: 100,
			Source:     "feedback",
			Status:     1,
		}
		if err := s.memories.Create(ctx, memory); err != nil {
			logger.Warn("反馈记忆沉淀失败", zap.Int64("userId", userID), zap.Error(err))
		}
	}()
}

func toFeedbackVO(feedback *model.SysAiMessageFeedback) *FeedbackVO {
	return &FeedbackVO{
		ID:         feedback.ID,
		MessageID:  feedback.MessageID,
		UserID:     feedback.UserID,
		Rating:     feedback.Rating,
		Tags:       rawJSON(feedback.Tags),
		Comment:    derefString(feedback.Comment),
		CreateTime: formatTime(feedback.CreateTime),
		UpdateTime: formatTimePtr(feedback.UpdateTime),
	}
}
