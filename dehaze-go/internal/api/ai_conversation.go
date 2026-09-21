package api

import (
	"strconv"

	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiConversationApi AI 会话/消息/反馈/产物（A 类非推理端点）。
type AiConversationApi struct {
	conversations *aidomain.ConversationService
	messages      *aidomain.MessageService
	feedbacks     *aidomain.FeedbackService
	artifacts     *aidomain.ArtifactService
}

// NewAiConversationApi 构造 AiConversationApi。
func NewAiConversationApi(
	conversations *aidomain.ConversationService,
	messages *aidomain.MessageService,
	feedbacks *aidomain.FeedbackService,
	artifacts *aidomain.ArtifactService,
) *AiConversationApi {
	return &AiConversationApi{
		conversations: conversations,
		messages:      messages,
		feedbacks:     feedbacks,
		artifacts:     artifacts,
	}
}

// requireConversationAudit 管理端会话审计权限（ROOT 放行，否则需 ai:conversation:audit）。
func requireConversationAudit(c *gin.Context) bool {
	if security.IsRoot(c) {
		return true
	}
	has, err := security.HasAnyPermission(c, "ai:conversation:audit")
	if err != nil || !has {
		_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "访问未授权"))
		return false
	}
	return true
}

// CreateConversation 创建会话。
func (a *AiConversationApi) CreateConversation(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form aidomain.ConversationCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.conversations.Create(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListConversations 会话列表（view=admin 为管理端审计视角）。
func (a *AiConversationApi) ListConversations(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	admin := c.Query("view") == "admin"
	if admin && !requireConversationAudit(c) {
		return
	}
	status, ok := parseOptionalInt(c.Query("status"))
	if !ok {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "status 取值非法"))
		return
	}
	result, err := a.conversations.List(c.Request.Context(), userID, pageNum, pageSize, c.Query("keyword"), status, admin)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListTrashConversations 回收站列表。
func (a *AiConversationApi) ListTrashConversations(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	result, err := a.conversations.ListTrash(c.Request.Context(), userID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// BatchConversations 批量操作会话。
func (a *AiConversationApi) BatchConversations(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form aidomain.ConversationBatchForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	count, err := a.conversations.BatchOperate(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(count, c)
}

// GetConversation 会话详情。
func (a *AiConversationApi) GetConversation(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	admin := c.Query("view") == "admin"
	if admin && !requireConversationAudit(c) {
		return
	}
	result, err := a.conversations.Get(c.Request.Context(), id, userID, admin)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateConversation 更新会话。
func (a *AiConversationApi) UpdateConversation(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.ConversationUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.conversations.Update(c.Request.Context(), id, userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteConversation 删除会话（软删）。
func (a *AiConversationApi) DeleteConversation(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	if err := a.conversations.Delete(c.Request.Context(), id, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// RestoreConversation 恢复软删会话。
func (a *AiConversationApi) RestoreConversation(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	result, err := a.conversations.Restore(c.Request.Context(), id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// PinConversation 置顶会话。
func (a *AiConversationApi) PinConversation(c *gin.Context) {
	a.updatePinState(c, true)
}

// UnpinConversation 取消置顶。
func (a *AiConversationApi) UnpinConversation(c *gin.Context) {
	a.updatePinState(c, false)
}

func (a *AiConversationApi) updatePinState(c *gin.Context, pin bool) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var result any
	if pin {
		result, err = a.conversations.Pin(c.Request.Context(), id, userID)
	} else {
		result, err = a.conversations.Unpin(c.Request.Context(), id, userID)
	}
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ReadConversation 标记会话已读。
func (a *AiConversationApi) ReadConversation(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	result, err := a.conversations.MarkRead(c.Request.Context(), id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ExportConversation 导出会话。
func (a *AiConversationApi) ExportConversation(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	format := c.DefaultQuery("format", "markdown")
	contentType, filename, payload, err := a.conversations.Export(c.Request.Context(), id, userID, format)
	if err != nil {
		_ = c.Error(err)
		return
	}
	c.Header("Content-Disposition", "attachment; filename=\""+filename+"\"")
	c.Data(200, contentType, []byte(payload))
}

// ListMessages 会话消息列表。
func (a *AiConversationApi) ListMessages(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	admin := c.Query("view") == "admin"
	if admin && !requireConversationAudit(c) {
		return
	}
	before, limit, ok := parseMessageCursor(c)
	if !ok {
		return
	}
	result, err := a.messages.List(c.Request.Context(), id, userID, before, limit, admin)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetMessage 消息详情（含推理步骤）。
func (a *AiConversationApi) GetMessage(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	admin := c.Query("view") == "admin"
	if admin && !requireConversationAudit(c) {
		return
	}
	result, err := a.messages.Get(c.Request.Context(), id, userID, admin)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteMessage 删除助手消息（软删）。
func (a *AiConversationApi) DeleteMessage(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	if err := a.messages.Delete(c.Request.Context(), id, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// GetBranches 查询消息的分支列表。
func (a *AiConversationApi) GetBranches(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	convID, ok := parseID(c, "id")
	if !ok {
		return
	}
	msgID, ok := parseID(c, "messageId")
	if !ok {
		return
	}
	result, err := a.messages.GetBranches(c.Request.Context(), convID, msgID, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// SwitchBranch 切换当前分支。
func (a *AiConversationApi) SwitchBranch(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	convID, ok := parseID(c, "id")
	if !ok {
		return
	}
	msgID, ok := parseID(c, "messageId")
	if !ok {
		return
	}
	result, err := a.messages.SwitchBranch(c.Request.Context(), convID, msgID, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// SubmitFeedback 提交/更新消息反馈。
func (a *AiConversationApi) SubmitFeedback(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.FeedbackCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if form.Rating != 1 && form.Rating != -1 {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "评分取值非法"))
		return
	}
	result, err := a.feedbacks.Submit(c.Request.Context(), id, userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetFeedback 查询消息反馈状态。
func (a *AiConversationApi) GetFeedback(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	result, err := a.feedbacks.Get(c.Request.Context(), id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// RevokeFeedback 撤销消息反馈。
func (a *AiConversationApi) RevokeFeedback(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	if err := a.feedbacks.Revoke(c.Request.Context(), id, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// ListConversationArtifacts 会话产物分页列表。
func (a *AiConversationApi) ListConversationArtifacts(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	result, err := a.artifacts.ListByConversation(c.Request.Context(), id, userID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListMessageArtifacts 消息关联产物列表。
func (a *AiConversationApi) ListMessageArtifacts(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	result, err := a.artifacts.ListByMessage(c.Request.Context(), id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListArtifactsByRef 按业务引用反查产物列表。
func (a *AiConversationApi) ListArtifactsByRef(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	refType := c.Query("refType")
	refID, err := strconv.ParseInt(c.Query("refId"), 10, 64)
	if refType == "" || err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "refType/refId 参数非法"))
		return
	}
	result, err := a.artifacts.ListByRef(c.Request.Context(), refType, refID, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetArtifactDetail 产物详情（含运行时图片URL）。
func (a *AiConversationApi) GetArtifactDetail(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	result, err := a.artifacts.GetDetail(c.Request.Context(), id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}
