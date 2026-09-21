package api

import (
	"strconv"

	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiMemoryApi AI 长期记忆（A 类）。
type AiMemoryApi struct {
	memories *aidomain.MemoryService
}

// NewAiMemoryApi 构造 AiMemoryApi。
func NewAiMemoryApi(memories *aidomain.MemoryService) *AiMemoryApi {
	return &AiMemoryApi{memories: memories}
}

// ListMemories 记忆分页列表。
func (a *AiMemoryApi) ListMemories(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	result, err := a.memories.List(c.Request.Context(), userID, pageNum, pageSize, c.Query("memoryType"), c.Query("source"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListArchived 归档记忆分页列表。
func (a *AiMemoryApi) ListArchived(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	result, err := a.memories.ListArchived(c.Request.Context(), userID, pageNum, pageSize, c.Query("memoryType"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreateMemory 创建记忆。
func (a *AiMemoryApi) CreateMemory(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form aidomain.MemoryCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.memories.Create(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateMemory 更新记忆。
func (a *AiMemoryApi) UpdateMemory(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.MemoryUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.memories.Update(c.Request.Context(), id, userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteMemory 删除记忆。
func (a *AiMemoryApi) DeleteMemory(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	if err := a.memories.Delete(c.Request.Context(), id, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// UnarchiveMemory 取消归档记忆。
func (a *AiMemoryApi) UnarchiveMemory(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	result, err := a.memories.Unarchive(c.Request.Context(), id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// SearchMemories 关键词搜索记忆。
func (a *AiMemoryApi) SearchMemories(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	keyword := c.Query("keyword")
	if keyword == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "keyword 不能为空"))
		return
	}
	// python 该参数为裸 int（无 ge/le），此处只做类型校验 + 下界保护：
	// GORM 对负 Limit 视为"取消 LIMIT"（会返回全量），而 python 的 SQLAlchemy 会直接报错，
	// 故负值必须拒绝，不能透传。
	limit := 5
	if raw := c.Query("limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "limit 必须为正整数"))
			return
		}
		limit = parsed
	}
	result, err := a.memories.Search(c.Request.Context(), userID, keyword, limit)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ClearMemories 批量清空记忆。
func (a *AiMemoryApi) ClearMemories(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	confirm := c.Query("confirm") == "true" || c.Query("confirm") == "1"
	start, end, ok := queryTimeRange(c, "start", "end")
	if !ok {
		return
	}
	count, err := a.memories.Clear(c.Request.Context(), userID, confirm, c.Query("memoryType"), start, end)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(count, "已清空 "+strconv.FormatInt(count, 10)+" 条记忆（30 天内可恢复）", c)
}

// RestoreMemories 恢复软删记忆。
func (a *AiMemoryApi) RestoreMemories(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	confirm := c.Query("confirm") == "true" || c.Query("confirm") == "1"
	start, end, ok := queryTimeRange(c, "start", "end")
	if !ok {
		return
	}
	count, err := a.memories.Restore(c.Request.Context(), userID, confirm, c.Query("memoryType"), start, end)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(count, "已恢复 "+strconv.FormatInt(count, 10)+" 条记忆", c)
}

// ExportMemories 导出全部记忆（JSON/Markdown）。
func (a *AiMemoryApi) ExportMemories(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	format := c.DefaultQuery("fmt", "json")
	contentType, filename, content, err := a.memories.Export(c.Request.Context(), userID, format)
	if err != nil {
		_ = c.Error(err)
		return
	}
	c.Header("Content-Disposition", "attachment; filename=\""+filename+"\"")
	c.Data(200, contentType, []byte(content))
}
