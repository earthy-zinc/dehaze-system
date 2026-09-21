package aidomain

import (
	"context"
	"encoding/json"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
)

// ArtifactVO 产物响应。
type ArtifactVO struct {
	ID             int64           `json:"id"`
	ConversationID int64           `json:"conversationId"`
	MessageID      int64           `json:"messageId"`
	Type           string          `json:"type"`
	RefType        string          `json:"refType,omitempty"`
	RefID          *int64          `json:"refId,omitempty"`
	Summary        json.RawMessage `json:"summary,omitempty"`
	IsInvalid      int             `json:"isInvalid"`
	CreateTime     string          `json:"createTime,omitempty"`
}

// ArtifactDetailVO 产物详情（含运行时拼接的图片 URL）。
type ArtifactDetailVO struct {
	Artifact *ArtifactVO `json:"artifact"`
	ImageURL string      `json:"imageUrl,omitempty"`
}

// ArtifactService 中间产物查询业务逻辑。
type ArtifactService struct {
	artifacts     *repo.ArtifactRepository
	conversations *repo.ConversationRepository
	messages      *repo.MessageRepository
	storage       *storage.Registry
}

func NewArtifactService(
	artifacts *repo.ArtifactRepository,
	conversations *repo.ConversationRepository,
	messages *repo.MessageRepository,
	storageRegistry *storage.Registry,
) *ArtifactService {
	return &ArtifactService{
		artifacts:     artifacts,
		conversations: conversations,
		messages:      messages,
		storage:       storageRegistry,
	}
}

// ListByConversation 会话产物分页（校验会话归属）。
func (s *ArtifactService) ListByConversation(ctx context.Context, convID, userID int64, page, size int) (*vo.PageResult[ArtifactVO], error) {
	conv, err := s.conversations.GetByIDAndUser(ctx, convID, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	items, total, err := s.artifacts.ListByConversation(ctx, convID, page, size)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询产物列表失败", err)
	}
	result := make([]ArtifactVO, 0, len(items))
	for i := range items {
		result = append(result, *toArtifactVO(&items[i]))
	}
	return &vo.PageResult[ArtifactVO]{List: result, Total: total}, nil
}

// ListByMessage 消息关联产物列表（校验消息归属）。
func (s *ArtifactService) ListByMessage(ctx context.Context, messageID, userID int64) ([]ArtifactVO, error) {
	msg, err := s.messages.GetByIDAndUser(ctx, messageID, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
	}
	if msg == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	}
	items, err := s.artifacts.ListByMessage(ctx, messageID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询产物列表失败", err)
	}
	result := make([]ArtifactVO, 0, len(items))
	for i := range items {
		result = append(result, *toArtifactVO(&items[i]))
	}
	return result, nil
}

// ListByRef 按业务引用反查产物（仅返回当前用户所属会话的产物）。
func (s *ArtifactService) ListByRef(ctx context.Context, refType string, refID, userID int64) ([]ArtifactVO, error) {
	items, err := s.artifacts.ListByRef(ctx, refType, refID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询产物列表失败", err)
	}
	result := make([]ArtifactVO, 0, len(items))
	for i := range items {
		conv, convErr := s.conversations.GetByIDAndUser(ctx, items[i].ConversationID, userID)
		if convErr != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", convErr)
		}
		if conv != nil {
			result = append(result, *toArtifactVO(&items[i]))
		}
	}
	return result, nil
}

// GetDetail 产物详情（含运行时图片 URL）。
func (s *ArtifactService) GetDetail(ctx context.Context, artifactID, userID int64) (*ArtifactDetailVO, error) {
	artifact, err := s.artifacts.GetByID(ctx, artifactID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询产物失败", err)
	}
	if artifact == nil || artifact.IsInvalid != 0 {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "产物不存在或已失效")
	}
	conv, err := s.conversations.GetByIDAndUser(ctx, artifact.ConversationID, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "产物所属会话不存在")
	}
	imageURL, err := s.resolveImageURL(ctx, artifact)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "解析产物图片失败", err)
	}
	return &ArtifactDetailVO{Artifact: toArtifactVO(artifact), ImageURL: imageURL}, nil
}

// resolveImageURL 经 ref 链路解析图片运行时 URL（URL 不落库，按需拼接）。
func (s *ArtifactService) resolveImageURL(ctx context.Context, artifact *model.SysAiArtifact) (string, error) {
	if artifact.RefID == nil || artifact.RefType == nil {
		return "", nil
	}
	var fileID *int64
	switch *artifact.RefType {
	case "sys_file":
		fileID = artifact.RefID
	case "sys_pred_log", "sys_eval_log":
		resolved, err := s.artifacts.GetRefFileID(ctx, *artifact.RefType, *artifact.RefID)
		if err != nil {
			return "", err
		}
		fileID = resolved
	}
	if fileID == nil {
		return "", nil
	}
	file, err := s.artifacts.GetFileByID(ctx, *fileID)
	if err != nil || file == nil {
		return "", err
	}
	if s.storage == nil {
		return "", nil
	}
	service, err := s.storage.Get(file.Storage)
	if err != nil {
		return "", nil
	}
	return service.GetURL(ctx, file.ObjectName)
}

func toArtifactVO(artifact *model.SysAiArtifact) *ArtifactVO {
	return &ArtifactVO{
		ID:             artifact.ID,
		ConversationID: artifact.ConversationID,
		MessageID:      artifact.MessageID,
		Type:           artifact.Type,
		RefType:        derefString(artifact.RefType),
		RefID:          artifact.RefID,
		Summary:        rawJSON(artifact.Summary),
		IsInvalid:      artifact.IsInvalid,
		CreateTime:     formatTime(artifact.CreateTime),
	}
}
