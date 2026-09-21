package aidomain

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// defaultAIModel 与 dehaze-python settings.AI_DEFAULT_MODEL 一致（会话未指定模型时的默认值）。
const defaultAIModel = "qwen3-0.6b"

// pinnedConversationLimit 置顶会话上限。
const pinnedConversationLimit = 10

// recycleWindowDays 软删恢复窗口（天）。
const recycleWindowDays = 30

// scenePrompts 场景默认提示词（对齐 python app/service/ai/strategies/scene_templates.py）。
var scenePrompts = map[string]string{
	"general": "【角色】你是用户亲切、可靠的对话助手。\n" +
		"【任务】回应用户的一般性提问与闲聊，给出清晰、准确的解答。\n" +
		"【指令】先理解意图再作答；不确定时坦承并给出进一步澄清；涉及专业领域时提供必要的背景。\n" +
		"【格式】分点或分段组织回答，语言简洁友好。",
	"image_dispatch": "【角色】你是图像处理任务的调度专家。\n" +
		"【任务】接收用户的图像处理请求，识别处理目标并调度合适的算法完成处理。\n" +
		"【指令】明确输入与期望输出；选择合适的处理算法与参数；处理结果仅以产物引用形式反馈，不展开处理过程细节。\n" +
		"【格式】先复述任务理解，再说明所选算法与参数，最后给出结果引用。",
	"multi_step": "【角色】你是擅长拆解复杂任务的推理专家。\n" +
		"【任务】将复杂问题分解为若干可执行的步骤，逐步求解并给出最终结论。\n" +
		"【指令】先规划步骤再执行；每步说明依据；遇到依赖前置结果的步骤须先取得结果；避免跳跃式推断。\n" +
		"【格式】以编号步骤呈现推理过程，最后以「结论」区块汇总。",
	"algorithm_recommend": "【角色】你是图像处理算法的推荐顾问。\n" +
		"【任务】根据用户提供的图像特征与处理诉求，推荐最合适的算法及参数。\n" +
		"【指令】结合用户偏好与历史处理习惯给出推荐；说明推荐理由与适用场景；提供备选方案。\n" +
		"【格式】列出推荐算法（含理由与匹配度），再给出参数建议与备选。",
	"scheduled_task": "【角色】你是可靠的任务编排与定时调度助手。\n" +
		"【任务】帮助用户设定、调整、查询定时处理任务，并确认任务已正确配置。\n" +
		"【指令】明确任务内容、执行频率与目标对象；校验参数合法性；反馈任务创建/变更结果。\n" +
		"【格式】以任务概览形式列出任务要素（内容/频率/状态）。",
}

// ConversationVO 会话响应（字段对齐 python ConversationResult，camelCase 输出）。
type ConversationVO struct {
	ID                     int64           `json:"id"`
	UserID                 int64           `json:"userId"`
	UserName               string          `json:"userName,omitempty"`
	TokenConsumed          *int64          `json:"tokenConsumed,omitempty"`
	CreditsConsumed        *int64          `json:"creditsConsumed,omitempty"`
	AnomalyType            string          `json:"anomalyType,omitempty"`
	AnomalyLabel           string          `json:"anomalyLabel,omitempty"`
	Title                  string          `json:"title"`
	Model                  string          `json:"model,omitempty"`
	AgentCode              string          `json:"agentCode,omitempty"`
	AgentVersion           *int            `json:"agentVersion,omitempty"`
	Summary                string          `json:"summary,omitempty"`
	SystemPrompt           string          `json:"systemPrompt,omitempty"`
	ModelConfig            json.RawMessage `json:"modelConfig,omitempty"`
	SuggestionsEnabled     int             `json:"suggestionsEnabled"`
	APIKeyID               *int64          `json:"apiKeyId,omitempty"`
	MessageCount           int             `json:"messageCount"`
	LastMessageAt          string          `json:"lastMessageAt,omitempty"`
	CurrentBranchMessageID *int64          `json:"currentBranchMessageId,omitempty"`
	LastReadMessageID      *int64          `json:"lastReadMessageId,omitempty"`
	Pinned                 int             `json:"pinned"`
	PinnedAt               string          `json:"pinnedAt,omitempty"`
	DeleteTime             string          `json:"deleteTime,omitempty"`
	UnreadCount            int64           `json:"unreadCount"`
	TitleSource            string          `json:"titleSource"`
	Status                 int             `json:"status"`
	MatchedMessageID       *int64          `json:"matchedMessageId,omitempty"`
	CreateTime             string          `json:"createTime,omitempty"`
	UpdateTime             string          `json:"updateTime,omitempty"`
}

// ConversationCreateForm 创建会话表单。
type ConversationCreateForm struct {
	Title              *string        `json:"title" binding:"omitempty,max=255"`
	Model              *string        `json:"model" binding:"omitempty,max=64"`
	SystemPrompt       *string        `json:"systemPrompt"`
	ModelConfig        map[string]any `json:"modelConfig"`
	APIKeyID           *int64         `json:"apiKeyId"`
	AgentCode          *string        `json:"agentCode" binding:"omitempty,max=64"`
	SuggestionsEnabled *bool          `json:"suggestionsEnabled"`
	Scene              *string        `json:"scene" binding:"omitempty,max=32"`
}

// ConversationUpdateForm 更新会话表单（指针区分未传字段）。
// 注意 pinned/status 在 python 侧为无界裸 int，此处刻意不加区间（勿"顺手"补约束）。
type ConversationUpdateForm struct {
	Title              *string        `json:"title" binding:"omitempty,max=255"`
	Model              *string        `json:"model" binding:"omitempty,max=64"`
	SystemPrompt       *string        `json:"systemPrompt"`
	ModelConfig        map[string]any `json:"modelConfig"`
	Pinned             *int           `json:"pinned"`
	Status             *int           `json:"status"`
	AgentCode          *string        `json:"agentCode" binding:"omitempty,max=64"`
	SuggestionsEnabled *bool          `json:"suggestionsEnabled"`
}

// ConversationBatchForm 批量操作表单。
type ConversationBatchForm struct {
	Action  string  `json:"action"`
	IDs     []int64 `json:"ids"`
	Confirm bool    `json:"confirm"`
}

// ConversationService 会话与消息（非推理类）业务逻辑。
type ConversationService struct {
	conversations *repo.ConversationRepository
	messages      *repo.MessageRepository
	agents        *repo.AgentRepository
}

func NewConversationService(
	conversations *repo.ConversationRepository,
	messages *repo.MessageRepository,
	agents *repo.AgentRepository,
) *ConversationService {
	return &ConversationService{conversations: conversations, messages: messages, agents: agents}
}

// resolveAgentAnchor 解析会话锚定的 (agent_code, agent_version)。
func (s *ConversationService) resolveAgentAnchor(ctx context.Context, agentCode *string) (string, *int) {
	code := "default"
	if agentCode != nil && strings.TrimSpace(*agentCode) != "" {
		code = strings.TrimSpace(*agentCode)
	}
	agent, err := s.agents.GetByCode(ctx, code)
	if err != nil || agent == nil {
		return code, nil
	}
	published, err := s.agents.GetLatestPublished(ctx, agent.ID)
	if err != nil || published == nil {
		return code, nil
	}
	versionNo := published.VersionNo
	return code, &versionNo
}

func scenePrompt(scene *string) string {
	if scene != nil {
		if prompt, ok := scenePrompts[*scene]; ok {
			return prompt
		}
	}
	return scenePrompts["general"]
}

// Create 创建会话。
func (s *ConversationService) Create(ctx context.Context, userID int64, form *ConversationCreateForm) (*ConversationVO, error) {
	agentCode, agentVersion := s.resolveAgentAnchor(ctx, form.AgentCode)

	systemPrompt := scenePrompt(form.Scene)
	if form.SystemPrompt != nil && *form.SystemPrompt != "" {
		systemPrompt = *form.SystemPrompt
	}
	title := "新对话"
	if form.Title != nil && *form.Title != "" {
		title = *form.Title
	}
	modelID := defaultAIModel
	if form.Model != nil && *form.Model != "" {
		modelID = *form.Model
	}
	suggestionsEnabled := 1
	if form.SuggestionsEnabled != nil && !*form.SuggestionsEnabled {
		suggestionsEnabled = 0
	}

	conv := &model.SysAiConversation{
		UserID:             userID,
		Title:              title,
		Model:              &modelID,
		AgentCode:          &agentCode,
		AgentVersion:       agentVersion,
		SystemPrompt:       &systemPrompt,
		ModelConfig:        marshalJSON(form.ModelConfig),
		SuggestionsEnabled: suggestionsEnabled,
		APIKeyID:           form.APIKeyID,
		Status:             1,
		TitleSource:        "auto",
	}
	if err := s.conversations.Create(ctx, conv); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建会话失败", err)
	}
	return s.toVO(conv), nil
}

// List 会话列表（admin=true 为管理端审计视角，需调用方已校验 ai:conversation:audit）。
func (s *ConversationService) List(ctx context.Context, userID int64, page, size int, keyword string, status *int, admin bool) (*vo.PageResult[ConversationVO], error) {
	var (
		convs []model.SysAiConversation
		total int64
		err   error
	)
	statusFilter := status
	if !admin {
		// 三态范围过滤：缺省仅活跃(1)，0=全部，1=活跃，2=归档
		if status == nil {
			active := 1
			statusFilter = &active
		} else if *status == 0 {
			statusFilter = nil
		}
	} else if statusFilter != nil && *statusFilter == 0 {
		statusFilter = nil
	}

	if keyword != "" {
		convs, total, err = s.conversations.PaginateWithKeyword(ctx, userID, page, size, keyword, statusFilter, admin)
	} else if admin {
		convs, total, err = s.conversations.PaginateAll(ctx, page, size, "", statusFilter)
	} else {
		convs, total, err = s.conversations.PaginateUser(ctx, userID, page, size, statusFilter)
	}
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话列表失败", err)
	}

	items := make([]ConversationVO, 0, len(convs))
	for i := range convs {
		item, err := s.toVOWithUnread(ctx, &convs[i])
		if err != nil {
			return nil, err
		}
		items = append(items, *item)
	}
	if admin {
		if err := s.attachAuditFields(ctx, items); err != nil {
			return nil, err
		}
	} else if keyword != "" {
		if err := s.attachMatchedMessages(ctx, items, keyword); err != nil {
			return nil, err
		}
	}
	return &vo.PageResult[ConversationVO]{List: items, Total: total}, nil
}

// Get 会话详情。
func (s *ConversationService) Get(ctx context.Context, id, userID int64, admin bool) (*ConversationVO, error) {
	var conv *model.SysAiConversation
	var err error
	if admin {
		conv, err = s.conversations.GetByID(ctx, id)
	} else {
		conv, err = s.conversations.GetByIDAndUser(ctx, id, userID)
	}
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	result, err := s.toVOWithUnread(ctx, conv)
	if err != nil {
		return nil, err
	}
	if admin {
		items := []ConversationVO{*result}
		if err := s.attachAuditFields(ctx, items); err != nil {
			return nil, err
		}
		result = &items[0]
	}
	return result, nil
}

// Update 更新会话（标题/模型/提示词/模型参数/置顶/状态/Agent 切换）。
func (s *ConversationService) Update(ctx context.Context, id, userID int64, form *ConversationUpdateForm) (*ConversationVO, error) {
	conv, err := s.conversations.GetByIDAndUser(ctx, id, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}

	fields := map[string]any{"update_by": userID}
	if form.Title != nil {
		fields["title"] = *form.Title
		fields["title_source"] = "manual"
		conv.Title = *form.Title
		conv.TitleSource = "manual"
	}
	if form.Model != nil {
		fields["model"] = *form.Model
		conv.Model = form.Model
	}
	if form.SystemPrompt != nil {
		fields["system_prompt"] = *form.SystemPrompt
		conv.SystemPrompt = form.SystemPrompt
	}
	if form.ModelConfig != nil {
		fields["model_config"] = marshalJSON(form.ModelConfig)
		conv.ModelConfig = marshalJSON(form.ModelConfig)
	}
	if form.SuggestionsEnabled != nil {
		value := 0
		if *form.SuggestionsEnabled {
			value = 1
		}
		fields["suggestions_enabled"] = value
		conv.SuggestionsEnabled = value
	}
	if form.Status != nil {
		fields["status"] = *form.Status
		conv.Status = *form.Status
	}
	// 置顶状态须在字段回写前取原值：仅"未置顶→置顶"时占用名额并写 pinned_at
	if form.Pinned != nil {
		wasPinned := conv.Pinned == 1
		fields["pinned"] = *form.Pinned
		conv.Pinned = *form.Pinned
		if *form.Pinned == 0 {
			fields["pinned_at"] = nil
			conv.PinnedAt = nil
		} else if !wasPinned {
			pinnedAt, err := s.pinWithLimit(ctx, id, userID)
			if err != nil {
				return nil, err
			}
			fields["pinned_at"] = pinnedAt
			conv.PinnedAt = &pinnedAt
		}
	}
	if form.AgentCode != nil {
		agentCode, agentVersion := s.resolveAgentAnchor(ctx, form.AgentCode)
		fields["agent_code"] = agentCode
		fields["agent_version"] = agentVersion
		conv.AgentCode = &agentCode
		conv.AgentVersion = agentVersion
	}
	if err := s.conversations.UpdateFields(ctx, id, fields); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新会话失败", err)
	}
	return s.toVOWithUnread(ctx, conv)
}

// Delete 删除会话（软删，30 天内可恢复）。
func (s *ConversationService) Delete(ctx context.Context, id, userID int64) error {
	conv, err := s.conversations.GetByIDAndUser(ctx, id, userID)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	if err := s.conversations.SoftDeleteByIDs(ctx, []int64{id}, userID); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除会话失败", err)
	}
	return nil
}

// Restore 恢复回收站会话（仅 30 天窗口内）。
func (s *ConversationService) Restore(ctx context.Context, id, userID int64) (*ConversationVO, error) {
	windowStart := time.Now().AddDate(0, 0, -recycleWindowDays)
	conv, err := s.conversations.GetInTrash(ctx, id, userID, windowStart)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在或已超出恢复窗口")
	}
	if err := s.conversations.RestoreByIDs(ctx, []int64{id}, userID); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "恢复会话失败", err)
	}
	conv.Deleted = 0
	conv.DeleteTime = nil
	return s.toVOWithUnread(ctx, conv)
}

// ListTrash 回收站列表。
func (s *ConversationService) ListTrash(ctx context.Context, userID int64, page, size int) (*vo.PageResult[ConversationVO], error) {
	windowStart := time.Now().AddDate(0, 0, -recycleWindowDays)
	convs, total, err := s.conversations.PaginateTrash(ctx, userID, page, size, windowStart)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询回收站失败", err)
	}
	items := make([]ConversationVO, 0, len(convs))
	for i := range convs {
		item, err := s.toVOWithUnread(ctx, &convs[i])
		if err != nil {
			return nil, err
		}
		items = append(items, *item)
	}
	return &vo.PageResult[ConversationVO]{List: items, Total: total}, nil
}

// BatchOperate 批量操作（archive/restore/delete），任一失败整体拒绝。
func (s *ConversationService) BatchOperate(ctx context.Context, userID int64, form *ConversationBatchForm) (int, error) {
	if len(form.IDs) == 0 {
		return 0, common.NewBizError(common.PARAM_ERROR, "会话ID列表不能为空")
	}
	switch form.Action {
	case "archive", "restore", "delete":
	default:
		return 0, common.NewBizError(common.PARAM_ERROR, "不支持的批量操作类型")
	}
	if form.Action == "delete" && !form.Confirm {
		return 0, common.NewBizError(common.PARAM_ERROR, "批量删除需二次确认")
	}

	convs := make([]*model.SysAiConversation, 0, len(form.IDs))
	for _, id := range form.IDs {
		conv, err := s.conversations.GetByIDAndUser(ctx, id, userID)
		if err != nil {
			return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
		}
		if conv == nil {
			return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
		}
		convs = append(convs, conv)
	}

	switch form.Action {
	case "archive":
		for _, conv := range convs {
			if conv.Status != 1 {
				return 0, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "仅活跃会话可归档")
			}
		}
		if err := s.conversations.UpdateStatusByIDs(ctx, form.IDs, 2, userID); err != nil {
			return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "归档会话失败", err)
		}
	case "restore":
		for _, conv := range convs {
			if conv.Status != 2 {
				return 0, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "仅已归档会话可恢复")
			}
		}
		if err := s.conversations.UpdateStatusByIDs(ctx, form.IDs, 1, userID); err != nil {
			return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "恢复会话失败", err)
		}
	case "delete":
		if err := s.conversations.SoftDeleteByIDs(ctx, form.IDs, userID); err != nil {
			return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除会话失败", err)
		}
	}
	return len(form.IDs), nil
}

// Pin 置顶会话。
func (s *ConversationService) Pin(ctx context.Context, id, userID int64) (*ConversationVO, error) {
	resp, err := s.ownedActive(ctx, id, userID)
	if err != nil {
		return nil, err
	}
	if resp.Pinned != 1 {
		pinnedAt, err := s.pinWithLimit(ctx, id, userID)
		if err != nil {
			return nil, err
		}
		resp.Pinned = 1
		resp.PinnedAt = formatTime(pinnedAt)
	}
	return resp, nil
}

// Unpin 取消置顶。
func (s *ConversationService) Unpin(ctx context.Context, id, userID int64) (*ConversationVO, error) {
	resp, err := s.ownedActive(ctx, id, userID)
	if err != nil {
		return nil, err
	}
	if err := s.conversations.SetPinned(ctx, id, 0, nil); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "取消置顶失败", err)
	}
	resp.Pinned = 0
	resp.PinnedAt = ""
	return resp, nil
}

// MarkRead 标记已读（last_read_message_id 置为会话最后一条消息 ID）。
func (s *ConversationService) MarkRead(ctx context.Context, id, userID int64) (*ConversationVO, error) {
	if _, err := s.ownedActive(ctx, id, userID); err != nil {
		return nil, err
	}
	lastMsgID, err := s.messages.GetLastMessageID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", err)
	}
	if lastMsgID != nil {
		if err := s.conversations.MarkRead(ctx, id, *lastMsgID, userID); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "标记已读失败", err)
		}
	}
	conv, err := s.conversations.GetByIDAndUser(ctx, id, userID)
	if err != nil || conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	return s.toVOWithUnread(ctx, conv)
}

// Export 导出会话（沿当前激活分支回溯 user/assistant 消息）。
func (s *ConversationService) Export(ctx context.Context, id, userID int64, format string) (contentType, filename, payload string, err error) {
	conv, err := s.conversations.GetByIDAndUser(ctx, id, userID)
	if err != nil {
		return "", "", "", common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return "", "", "", common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	var messages []model.SysAiMessage
	if conv.CurrentBranchMessageID != nil {
		chain, chainErr := s.messages.GetChainByID(ctx, id, *conv.CurrentBranchMessageID)
		if chainErr != nil {
			return "", "", "", common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询消息失败", chainErr)
		}
		for _, m := range chain {
			if m.Role == "user" || m.Role == "assistant" {
				messages = append(messages, m)
			}
		}
	}

	if format == "json" {
		type exportMessage struct {
			Role       string `json:"role"`
			Content    string `json:"content"`
			CreateTime string `json:"create_time"`
		}
		items := make([]exportMessage, 0, len(messages))
		for _, m := range messages {
			content := ""
			if m.Content != nil {
				content = *m.Content
			}
			items = append(items, exportMessage{Role: m.Role, Content: content, CreateTime: formatTime(m.CreateTime)})
		}
		body := map[string]any{
			"conversation": map[string]any{
				"id":          conv.ID,
				"title":       conv.Title,
				"model":       derefString(conv.Model),
				"agent_code":  derefString(conv.AgentCode),
				"create_time": formatTime(conv.CreateTime),
			},
			"messages": items,
		}
		b, marshalErr := json.MarshalIndent(body, "", "  ")
		if marshalErr != nil {
			return "", "", "", common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "导出会话失败", marshalErr)
		}
		return "application/json", fmt.Sprintf("conversation_%d.json", id), string(b), nil
	}

	roleLabel := map[string]string{"user": "用户", "assistant": "助手"}
	lines := []string{"# " + conv.Title, ""}
	for _, m := range messages {
		lines = append(lines, "## "+roleLabel[m.Role], "", derefString(m.Content), "")
	}
	return "text/markdown", fmt.Sprintf("conversation_%d.md", id), strings.Join(lines, "\n"), nil
}

// ownedActive 取归属当前用户且未删除的会话（不存在即 404）。
func (s *ConversationService) ownedActive(ctx context.Context, id, userID int64) (*ConversationVO, error) {
	conv, err := s.conversations.GetByIDAndUser(ctx, id, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话失败", err)
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}
	return s.toVOWithUnread(ctx, conv)
}

// pinWithLimit 在用户级一把锁内完成上限校验与置顶写入，避免并发置顶超限。
func (s *ConversationService) pinWithLimit(ctx context.Context, id, userID int64) (time.Time, error) {
	lockKey := fmt.Sprintf("ai:conv:pin:%d", userID)
	if client := redis.GetClient(); client != nil {
		token := strconv.FormatInt(time.Now().UnixNano(), 10)
		ok, err := client.SetNX(ctx, lockKey, token, 10*time.Second).Result()
		if err != nil || !ok {
			return time.Time{}, common.NewBizError(common.BUSINESS_ERROR, "置顶操作并发冲突，请稍后再试")
		}
		defer func() {
			// 仅释放自己持有的锁，避免误删他人锁
			if current, getErr := client.Get(ctx, lockKey).Result(); getErr == nil && current == token {
				_ = client.Del(ctx, lockKey).Err()
			}
		}()
	}
	count, err := s.conversations.CountActivePinned(ctx, userID)
	if err != nil {
		return time.Time{}, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询置顶会话失败", err)
	}
	if count >= pinnedConversationLimit {
		return time.Time{}, common.NewBizError(common.DATA_EXISTS, "置顶会话已达上限")
	}
	now := time.Now()
	if err := s.conversations.SetPinned(ctx, id, 1, &now); err != nil {
		return time.Time{}, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "置顶会话失败", err)
	}
	return now, nil
}

func (s *ConversationService) toVO(conv *model.SysAiConversation) *ConversationVO {
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

func (s *ConversationService) toVOWithUnread(ctx context.Context, conv *model.SysAiConversation) (*ConversationVO, error) {
	result := s.toVO(conv)
	if conv.LastReadMessageID != nil {
		unread, err := s.messages.CountMessagesAfter(ctx, conv.ID, *conv.LastReadMessageID)
		if err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "统计未读消息失败", err)
		}
		result.UnreadCount = unread
	} else {
		result.UnreadCount = int64(conv.MessageCount)
	}
	return result, nil
}

// attachAuditFields 补充审计视角字段（用户名/消耗汇总/异常标注），批量查询避免 N+1。
func (s *ConversationService) attachAuditFields(ctx context.Context, items []ConversationVO) error {
	if len(items) == 0 {
		return nil
	}
	convIDs := make([]int64, 0, len(items))
	userIDs := make([]int64, 0, len(items))
	for _, item := range items {
		convIDs = append(convIDs, item.ID)
		userIDs = append(userIDs, item.UserID)
	}
	names, err := s.conversations.ListUserNames(ctx, userIDs)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询用户名失败", err)
	}
	consumption, err := s.conversations.SumConsumptionByConversation(ctx, convIDs)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "会话消耗聚合失败", err)
	}
	anomalyStatus, err := s.messages.ListAnomalyStatusByConversations(ctx, convIDs)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "会话异常聚合失败", err)
	}
	quotaConvIDs, err := s.conversations.ListQuotaAnomalyConversationIDs(ctx, convIDs)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "配额异常聚合失败", err)
	}
	riskyConvIDs, err := s.conversations.ListRiskyToolConversationIDs(ctx, convIDs)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "工具异常聚合失败", err)
	}

	for i := range items {
		items[i].UserName = names[items[i].UserID]
		stat := consumption[items[i].ID]
		token, credits := stat.Token, stat.Credits
		items[i].TokenConsumed = &token
		items[i].CreditsConsumed = &credits
		statuses := anomalyStatus[items[i].ID]
		switch {
		case hasStatus(statuses, 3):
			items[i].AnomalyType, items[i].AnomalyLabel = "failed", "存在失败消息"
		case hasKey(quotaConvIDs, items[i].ID):
			items[i].AnomalyType, items[i].AnomalyLabel = "quota", "配额不足中断"
		case hasKey(riskyConvIDs, items[i].ID):
			items[i].AnomalyType, items[i].AnomalyLabel = "risky_tool", "存在高风险工具调用"
		case hasStatus(statuses, 4):
			items[i].AnomalyType, items[i].AnomalyLabel = "canceled", "存在已取消消息"
		}
	}
	return nil
}

func (s *ConversationService) attachMatchedMessages(ctx context.Context, items []ConversationVO, keyword string) error {
	if len(items) == 0 || keyword == "" {
		return nil
	}
	convIDs := make([]int64, 0, len(items))
	for _, item := range items {
		if !strings.Contains(item.Title, keyword) {
			convIDs = append(convIDs, item.ID)
		}
	}
	if len(convIDs) == 0 {
		return nil
	}
	matched, err := s.conversations.ListMatchedMessageIDs(ctx, convIDs, keyword)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "搜索命中消息失败", err)
	}
	for i := range items {
		if id, ok := matched[items[i].ID]; ok {
			value := id
			items[i].MatchedMessageID = &value
		}
	}
	return nil
}

func hasStatus(statuses map[int]struct{}, status int) bool {
	_, ok := statuses[status]
	return ok
}

func hasKey(set map[int64]struct{}, key int64) bool {
	_, ok := set[key]
	return ok
}
