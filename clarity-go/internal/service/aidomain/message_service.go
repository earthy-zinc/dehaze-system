package aidomain

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/aicontract"
	repo "github.com/earthyzinc/clarity-go/internal/repository/aidomain"
	"github.com/earthyzinc/clarity-go/pkg/cache/redis"
	"github.com/earthyzinc/clarity-go/pkg/common"
	goredis "github.com/redis/go-redis/v9"
)

// interruptKeyFmt 中断点 Redis 键，与 clarity-python InterruptHandler.INTERRUPT_KEY 一致：
// thread_id 为 "{conversation_id}:{current_branch_message_id}"，由推理侧写入、resume 成功后清除。
// Go 侧只读不写。
const interruptKeyFmt = "ai:interrupt:%d:%d"

// MessageService 消息查询与分支管理（非推理类）。
//
// 响应直接使用契约 DTO（aicontract.Message）：消息只有 blocks/plan/pendingInterrupt 这一种
// 形状，推理步骤不再随消息下发（旧 thoughts 字段已随契约删除），历史一致性由 blocks 保证。
type MessageService struct {
	conversations *repo.ConversationRepository
	messages      *repo.MessageRepository
	artifacts     *repo.ArtifactRepository
}

func NewMessageService(
	conversations *repo.ConversationRepository,
	messages *repo.MessageRepository,
	artifacts *repo.ArtifactRepository,
) *MessageService {
	return &MessageService{conversations: conversations, messages: messages, artifacts: artifacts}
}

// List 会话消息列表。
//
// self 视角（默认）只返回当前激活链：以会话 current_branch_message_id 为链尾沿
// parent_message_id 回溯（指针为空/悬空/指向已删消息时回退到会话最大 id 的未删消息），
// 链即用户实际看到的对话。分页在链上切：before 之前的最后 limit 条按 id 倒序返回，
// hasMore 表示链上是否还有更早消息，total 为链长——分支消息不再混入列表，前端无需再做
// "每个分叉取最新"的选择算法。
//
// admin 视角保持全量分支分页（审计需要看到所有分叉），total 为会话消息总数。
// siblingIndex/siblingCount 仅在 self 视角的 assistant 消息上填充。
func (s *MessageService) List(ctx context.Context, convID, userID int64, before *int64, limit int, admin bool) (*aicontract.MessageList, error) {
	conv, err := s.loadConversation(ctx, convID, userID, admin)
	if err != nil {
		return nil, err
	}
	if admin {
		msgs, total, err := s.messages.ListByConversationCursor(ctx, convID, before, limit)
		if err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息列表失败", err)
		}
		hasMore := len(msgs) > limit
		if hasMore {
			msgs = msgs[:limit]
		}
		items, err := s.toMessages(ctx, msgs)
		if err != nil {
			return nil, err
		}
		return &aicontract.MessageList{List: items, Total: total, HasMore: hasMore}, nil
	}

	chain, err := s.loadActiveChain(ctx, conv)
	if err != nil {
		return nil, err
	}
	eligible := chain
	if before != nil {
		eligible = make([]model.SysAiMessage, 0, len(chain))
		for i := range chain {
			if chain[i].ID < *before {
				eligible = append(eligible, chain[i])
			}
		}
	}
	hasMore := len(eligible) > limit
	start := len(eligible) - limit
	if start < 0 {
		start = 0
	}
	// 链回溯按 id 升序，接口契约仍按 id 倒序返回（最新一页在前）
	page := make([]model.SysAiMessage, 0, len(eligible)-start)
	for i := len(eligible) - 1; i >= start; i-- {
		page = append(page, eligible[i])
	}
	positions, err := s.siblingPositions(ctx, convID, page)
	if err != nil {
		return nil, err
	}
	items, err := s.toMessages(ctx, page)
	if err != nil {
		return nil, err
	}
	for i := range items {
		if position, ok := positions[items[i].ID]; ok {
			index, count := position[0], position[1]
			items[i].SiblingIndex = &index
			items[i].SiblingCount = &count
		}
	}
	return &aicontract.MessageList{List: items, Total: int64(len(chain)), HasMore: hasMore}, nil
}

// loadActiveChain 当前激活链（id 升序）：指针为空/悬空/指向已删消息时回退到最新未删消息。
func (s *MessageService) loadActiveChain(ctx context.Context, conv *model.SysAiConversation) ([]model.SysAiMessage, error) {
	tailID := conv.CurrentBranchMessageID
	if tailID != nil {
		tail, err := s.messages.GetByID(ctx, *tailID)
		if err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
		}
		// 悬空、已软删（GetByID 过滤 deleted=0）或不属于本会话，一律回退
		if tail == nil || tail.ConversationID != conv.ID {
			tailID = nil
		}
	}
	if tailID == nil {
		maxID, err := s.messages.GetLastMessageID(ctx, conv.ID)
		if err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
		}
		tailID = maxID
	}
	if tailID == nil {
		return nil, nil
	}
	chain, err := s.messages.GetChainByID(ctx, conv.ID, *tailID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询激活分支失败", err)
	}
	return chain, nil
}

// siblingPositions 批量计算本页 assistant 消息的兄弟序号：{message_id: [1 起序号, 兄弟总数]}。
// 仅 parentMessageId 非空（分叉下的助手回复）参与，root 级与非 assistant 保持 null；
// 一次 IN 查询取全部兄弟，避免逐条消息查询的 N+1。
func (s *MessageService) siblingPositions(ctx context.Context, convID int64, msgs []model.SysAiMessage) (map[int64][2]int64, error) {
	pageIDs := make(map[int64]struct{}, len(msgs))
	parentSet := make(map[int64]struct{})
	for i := range msgs {
		pageIDs[msgs[i].ID] = struct{}{}
		if msgs[i].Role == aicontract.RoleAssistant && msgs[i].ParentMessageID != nil {
			parentSet[*msgs[i].ParentMessageID] = struct{}{}
		}
	}
	if len(parentSet) == 0 {
		return nil, nil
	}
	parentIDs := make([]int64, 0, len(parentSet))
	for parentID := range parentSet {
		parentIDs = append(parentIDs, parentID)
	}
	grouped, err := s.messages.ListAssistantSiblingIDs(ctx, convID, parentIDs)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询兄弟消息失败", err)
	}
	positions := make(map[int64][2]int64)
	for _, siblingIDs := range grouped {
		count := int64(len(siblingIDs))
		for i, siblingID := range siblingIDs {
			if _, onPage := pageIDs[siblingID]; onPage {
				positions[siblingID] = [2]int64{int64(i) + 1, count}
			}
		}
	}
	return positions, nil
}

// Get 消息详情（完整 Message：块、计划、待恢复中断、用量与产物引用）。
//
// 与列表同一口径回填 siblingIndex/siblingCount：详情是单条消息，仍走批量兄弟查询（一次 IN），
// root 级与非 assistant 保持 null；软删兄弟不占位。
func (s *MessageService) Get(ctx context.Context, msgID, userID int64, admin bool) (*aicontract.Message, error) {
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
	artifactIDs, err := s.artifactIDs(ctx, []int64{msg.ID})
	if err != nil {
		return nil, err
	}
	result := toContractMessage(msg, artifactIDs[msg.ID])
	positions, err := s.siblingPositions(ctx, msg.ConversationID, []model.SysAiMessage{*msg})
	if err != nil {
		return nil, err
	}
	if position, ok := positions[msg.ID]; ok {
		index, count := position[0], position[1]
		result.SiblingIndex = &index
		result.SiblingCount = &count
	}
	return &result, nil
}

// GetInterrupt 查询消息当前待恢复的中断；无中断 payload 返回 nil（前端按 null 处理）。
// 与列表/详情走同一个 normalizePendingInterrupt：过期或消息已不是 awaiting_input 时
// 照常返回 payload，但 actionable=false 明确标注不可续。
func (s *MessageService) GetInterrupt(ctx context.Context, msgID, userID int64, admin bool) (*aicontract.Interrupt, error) {
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
	return normalizePendingInterrupt(msg), nil
}

// Delete 删除助手消息（软删）。
//
// 分支一致性：
//   - 非叶子（仍有未删子消息，不区分 role）拒绝删除：删掉中间节点会让后续消息的 parent
//     悬空、激活链从断点处被截断，且无法通过桥接恢复对话；提示用户先删除其后的对话；
//   - 被删消息正是 current_branch_message_id 时，指针在同一事务内回退到 parent（可为 NULL），
//     之后继续发送会挂到 parent 下，历史消息仍在；
//   - 非激活链上的旧分支消息照删，指针不动（仓库层用 CAS 判定指针是否指向被删消息）。
//
// children 检查、软删与指针回退都在会话行锁内完成（DeleteAssistantLeaf）：并发 send
// 推进指针或挂上子消息时，删除要么看到完整现状被拒，要么让 CAS 落空，不会覆盖新指针。
func (s *MessageService) Delete(ctx context.Context, msgID, userID int64) error {
	msg, err := s.messages.GetByIDAndUser(ctx, msgID, userID)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
	}
	if msg == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	}
	if msg.Role != aicontract.RoleAssistant {
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "仅助手消息可删除")
	}
	outcome, err := s.messages.DeleteAssistantLeaf(ctx, msg.ConversationID, msgID, userID)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除消息失败", err)
	}
	switch outcome {
	case repo.DeleteLeafNotFound:
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	case repo.DeleteLeafHasChildren:
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "请先删除该回复之后的对话")
	}
	return nil
}

// GetBranches 查询消息的分支列表（时间倒序）。
//
// 分支是同父的助手回复：只返回 role=assistant 的子消息，与列表/详情的 siblingIndex 口径
// 一致。删尾后继续发送会让用户新提问也挂在同一 parent 下，它不是可切换的分支。
func (s *MessageService) GetBranches(ctx context.Context, convID, msgID, userID int64) ([]aicontract.Message, error) {
	if _, err := s.loadConversation(ctx, convID, userID, false); err != nil {
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
	return s.toMessages(ctx, children)
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
	// 中断挂起时拒绝切换：当前指针上的 resume 句柄仍有效，切走后恢复请求会落到另一条链上。
	// 事实源是消息的 pending_interrupt（DB），Redis 键只是推理侧运行态句柄的补充信号。
	if conv.CurrentBranchMessageID != nil {
		pending, err := s.hasPendingInterrupt(ctx, conv.ID, *conv.CurrentBranchMessageID)
		if err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话中断状态失败", err)
		}
		if pending {
			return nil, common.NewBizError(common.BUSINESS_ERROR, "会话有未完成的中断确认，请先确认或停止")
		}
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

// hasPendingInterrupt 当前分支指针上是否仍有可恢复的中断。
//
// pending_interrupt 以消息行（DB）为事实源：消息仍属本会话、status=awaiting_input 且
// 未过期（actionable）才算挂起，与列表/详情/中断查询同一口径；指针悬空或消息已软删时
// 不阻断。Redis 键 ai:interrupt:{conv}:{msg} 是推理侧运行态句柄，作为补充信号一并检查：
// 任一命中都视为挂起，Redis 未初始化时只依赖 DB，不阻断非推理主流程。
func (s *MessageService) hasPendingInterrupt(ctx context.Context, convID, msgID int64) (bool, error) {
	msg, err := s.messages.GetByID(ctx, msgID)
	if err != nil {
		return false, err
	}
	if msg != nil && msg.ConversationID == convID {
		if interrupt := normalizePendingInterrupt(msg); interrupt != nil && interrupt.Actionable {
			return true, nil
		}
	}
	client := redis.GetClient()
	if client == nil {
		return false, nil
	}
	raw, err := client.Get(ctx, fmt.Sprintf(interruptKeyFmt, convID, msgID)).Result()
	if errors.Is(err, goredis.Nil) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return raw != "", nil
}

// loadConversation 取会话并校验归属：admin 视角不限归属，self 视角须属于当前用户。
func (s *MessageService) loadConversation(ctx context.Context, convID, userID int64, admin bool) (*model.SysAiConversation, error) {
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
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	return conv, nil
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

// toMessages 批量组装契约消息：产物 ID 一次查询回填，避免逐消息往返。
func (s *MessageService) toMessages(ctx context.Context, msgs []model.SysAiMessage) ([]aicontract.Message, error) {
	items := make([]aicontract.Message, 0, len(msgs))
	if len(msgs) == 0 {
		return items, nil
	}
	ids := make([]int64, 0, len(msgs))
	for i := range msgs {
		ids = append(ids, msgs[i].ID)
	}
	artifactIDs, err := s.artifactIDs(ctx, ids)
	if err != nil {
		return nil, err
	}
	for i := range msgs {
		items = append(items, toContractMessage(&msgs[i], artifactIDs[msgs[i].ID]))
	}
	return items, nil
}

func (s *MessageService) artifactIDs(ctx context.Context, messageIDs []int64) (map[int64][]int64, error) {
	ids, err := s.artifacts.ListIDsByMessages(ctx, messageIDs)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息产物失败", err)
	}
	return ids, nil
}

// toContractMessage 库中行 → 契约 Message。所有数组字段在无值时给空切片，
// 保证序列化结果是 [] 而不是 null（契约要求数组）。
func toContractMessage(msg *model.SysAiMessage, artifactIDs []int64) aicontract.Message {
	return aicontract.Message{
		ID:               msg.ID,
		ConversationID:   msg.ConversationID,
		ParentMessageID:  msg.ParentMessageID,
		Role:             msg.Role,
		Status:           msg.Status,
		Model:            msg.Model,
		Error:            msg.Error,
		Blocks:           aicontract.DecodeBlocks(msg.Blocks),
		Plan:             aicontract.DecodePlan(msg.Plan),
		PendingInterrupt: normalizePendingInterrupt(msg),
		Usage: aicontract.Usage{
			InputTokens:       int64(msg.InputTokens),
			OutputTokens:      int64(msg.OutputTokens),
			CachedInputTokens: int64(msg.CachedInputTokens),
			Credits:           msg.Credits,
			SubAgents:         aicontract.DecodeSubAgentUsage(msg.SubAgentUsage),
		},
		ArtifactIDs: nonNilIDs(artifactIDs),
		MemoryIDs:   aicontract.DecodeIDList(msg.UsedMemoryIDs),
		Suggestions: aicontract.DecodeSuggestions(msg.Suggestions),
		Edited:      msg.Edited != 0,
		CreateTime:  formatContractTime(msg.CreateTime),
		UpdateTime:  formatContractTimePtr(msg.UpdateTime),
	}
}

// normalizePendingInterrupt 按当前时间与消息状态重算可续性：过 expiresAt 或消息已不是
// awaiting_input 时 actionable=false——历史里残留的中断不给出"可续"的错误信号。
func normalizePendingInterrupt(msg *model.SysAiMessage) *aicontract.Interrupt {
	interrupt := aicontract.NormalizeInterrupt(aicontract.DecodeInterrupt(msg.PendingInterrupt), time.Now())
	if interrupt == nil {
		return nil
	}
	if msg.Status != aicontract.MessageStatusAwaitingInput {
		interrupt.Actionable = false
	}
	return interrupt
}

// nonNilIDs 空集合返回空切片：artifactIds 为契约必填数组，不能下发 null。
func nonNilIDs(ids []int64) []int64 {
	if ids == nil {
		return []int64{}
	}
	return ids
}
