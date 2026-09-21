package aidomain

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// MemoryVO 长期记忆响应。
type MemoryVO struct {
	ID             int64           `json:"id"`
	UserID         int64           `json:"userId"`
	MemoryType     string          `json:"memoryType"`
	Content        string          `json:"content"`
	Metadata       json.RawMessage `json:"metadata,omitempty"`
	Importance     int             `json:"importance"`
	AccessCount    int             `json:"accessCount"`
	LastAccessedAt string          `json:"lastAccessedAt,omitempty"`
	Source         string          `json:"source"`
	Status         int             `json:"status"`
	Archived       int             `json:"archived"`
	CreateTime     string          `json:"createTime,omitempty"`
	UpdateTime     string          `json:"updateTime,omitempty"`
}

// MemoryCreateForm 创建记忆表单。
type MemoryCreateForm struct {
	MemoryType string         `json:"memoryType" binding:"required"`
	Content    string         `json:"content" binding:"required,max=2000"`
	Metadata   map[string]any `json:"metadata"`
	Importance int            `json:"importance" binding:"gte=0,lte=100"`
	Source     string         `json:"source"`
}

// MemoryUpdateForm 更新记忆表单。
type MemoryUpdateForm struct {
	Content    *string `json:"content" binding:"omitempty,max=2000"`
	Importance *int    `json:"importance" binding:"omitempty,gte=0,lte=100"`
	Status     *int    `json:"status" binding:"omitempty,oneof=0 1"`
}

// MemoryService 长期记忆业务逻辑。
type MemoryService struct {
	memories *repo.MemoryRepository
	auditLog *auditlogservice.AuditLogService
}

func NewMemoryService(memories *repo.MemoryRepository, auditLog *auditlogservice.AuditLogService) *MemoryService {
	return &MemoryService{memories: memories, auditLog: auditLog}
}

// List 活跃记忆分页。
func (s *MemoryService) List(ctx context.Context, userID int64, page, size int, memoryType, source string) (*vo.PageResult[MemoryVO], error) {
	items, total, err := s.memories.ListActive(ctx, userID, memoryType, source, page, size)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询记忆列表失败", err)
	}
	return &vo.PageResult[MemoryVO]{List: toMemoryVOs(items), Total: total}, nil
}

// ListArchived 归档记忆分页。
func (s *MemoryService) ListArchived(ctx context.Context, userID int64, page, size int, memoryType string) (*vo.PageResult[MemoryVO], error) {
	items, total, err := s.memories.ListArchived(ctx, userID, memoryType, page, size)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询归档记忆失败", err)
	}
	return &vo.PageResult[MemoryVO]{List: toMemoryVOs(items), Total: total}, nil
}

// Create 创建记忆。
func (s *MemoryService) Create(ctx context.Context, userID int64, form *MemoryCreateForm) (*MemoryVO, error) {
	if strings.TrimSpace(form.MemoryType) == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "记忆类型不能为空")
	}
	if form.Content == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "记忆内容不能为空")
	}
	source := form.Source
	if source == "" {
		source = "manual"
	}
	importance := form.Importance
	if importance == 0 {
		importance = 50
	}
	if importance < 0 || importance > 100 {
		return nil, common.NewBizError(common.PARAM_ERROR, "重要性评分需在 0-100 之间")
	}
	memory := &model.SysAiMemory{
		UserID:     userID,
		MemoryType: form.MemoryType,
		Content:    form.Content,
		Metadata:   marshalJSON(form.Metadata),
		Importance: importance,
		Source:     source,
		Status:     1,
	}
	if err := s.memories.Create(ctx, memory); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建记忆失败", err)
	}
	result := toMemoryVO(memory)
	return result, nil
}

// Update 更新记忆（内容/重要性/启停）。
func (s *MemoryService) Update(ctx context.Context, id, userID int64, form *MemoryUpdateForm) (*MemoryVO, error) {
	memory, err := s.memories.GetByIDAndUser(ctx, id, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询记忆失败", err)
	}
	if memory == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "记忆不存在")
	}
	fields := map[string]any{"update_by": userID}
	if form.Content != nil {
		fields["content"] = *form.Content
		memory.Content = *form.Content
	}
	if form.Importance != nil {
		if *form.Importance < 0 || *form.Importance > 100 {
			return nil, common.NewBizError(common.PARAM_ERROR, "重要性评分需在 0-100 之间")
		}
		fields["importance"] = *form.Importance
		memory.Importance = *form.Importance
	}
	if form.Status != nil {
		fields["status"] = *form.Status
		memory.Status = *form.Status
	}
	if err := s.memories.UpdateFields(ctx, id, fields); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新记忆失败", err)
	}
	return toMemoryVO(memory), nil
}

// esRequestTimeout ES 请求超时（对齐 java AiKbIndexClient 的 5s）
const esRequestTimeout = 5 * time.Second

// memoryEsIndex 记忆向量索引名，文档 _id 为记忆 id（与 python ai_memory_index 一致）
const memoryEsIndex = "ai_memory"

// deleteMemoryDocFromES 删除记忆的 ES 向量文档（对齐 python delete_memory_doc）。
//
// 不清理则已删记忆仍会被向量检索召回；ES 不可达只告警——DB 已软删，不因检索侧清理
// 失败回滚删除主流程。
func deleteMemoryDocFromES(ctx context.Context, memoryID int64) {
	esCfg := config.GetConfig().ES
	if esCfg.URL == "" {
		return
	}
	reqCtx, cancel := context.WithTimeout(ctx, esRequestTimeout)
	defer cancel()
	url := strings.TrimRight(esCfg.URL, "/") + "/" + memoryEsIndex + "/_doc/" +
		strconv.FormatInt(memoryID, 10) + "?refresh=true"
	req, err := http.NewRequestWithContext(reqCtx, http.MethodDelete, url, nil)
	if err != nil {
		logger.Warn("构建 ES 记忆文档删除请求失败", zap.Int64("memoryId", memoryID), zap.Error(err))
		return
	}
	if esCfg.Username != "" {
		req.SetBasicAuth(esCfg.Username, esCfg.Password)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		logger.Warn("ES 记忆向量文档清理失败", zap.Int64("memoryId", memoryID), zap.Error(err))
		return
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.Copy(io.Discard, resp.Body)
	// 文档不存在（404）视为已清理
	if resp.StatusCode >= 300 && resp.StatusCode != http.StatusNotFound {
		logger.Warn("ES 记忆向量文档清理返回异常状态",
			zap.Int64("memoryId", memoryID), zap.Int("status", resp.StatusCode))
	}
}

// Delete 删除记忆（软删，30 天内可恢复）。
func (s *MemoryService) Delete(ctx context.Context, id, userID int64) error {
	memory, err := s.memories.GetByIDAndUser(ctx, id, userID)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询记忆失败", err)
	}
	if memory == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "记忆不存在")
	}
	if err := s.memories.SoftDeleteWithTime(ctx, id); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除记忆失败", err)
	}
	deleteMemoryDocFromES(ctx, id)
	return nil
}

// Unarchive 取消归档并刷新衰减计时器。
func (s *MemoryService) Unarchive(ctx context.Context, id, userID int64) (*MemoryVO, error) {
	memory, err := s.memories.GetByIDAndUser(ctx, id, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询记忆失败", err)
	}
	if memory == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "记忆不存在")
	}
	if memory.Archived != 1 {
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "该记忆未处于归档状态")
	}
	now := time.Now()
	if err := s.memories.UpdateFields(ctx, id, map[string]any{
		"archived":         0,
		"last_accessed_at": now,
	}); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "取消归档失败", err)
	}
	memory.Archived = 0
	memory.LastAccessedAt = &now
	return toMemoryVO(memory), nil
}

// Search 关键词搜索记忆（命中后重激活）。
func (s *MemoryService) Search(ctx context.Context, userID int64, keyword string, limit int) ([]MemoryVO, error) {
	if keyword == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "搜索关键词不能为空")
	}
	items, err := s.memories.SearchByKeyword(ctx, userID, keyword, limit)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "搜索记忆失败", err)
	}
	result := toMemoryVOs(items)
	for i := range items {
		if err := s.memories.Touch(ctx, items[i].ID); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新记忆访问记录失败", err)
		}
	}
	return result, nil
}

// Clear 批量清空记忆（需二次确认，留审计）。
func (s *MemoryService) Clear(ctx context.Context, userID int64, confirm bool, memoryType string, start, end *time.Time) (int64, error) {
	if !confirm {
		return 0, common.NewBizError(common.PARAM_ERROR, "批量清空记忆为不可逆操作，需二次确认")
	}
	count, err := s.memories.BatchClear(ctx, userID, memoryType, start, end)
	if err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "清空记忆失败", err)
	}
	s.recordAudit(ctx, userID, "clear", map[string]any{
		"count":       count,
		"memory_type": memoryType,
		"start":       formatTimeValue(start),
		"end":         formatTimeValue(end),
	})
	return count, nil
}

// Restore 恢复窗口内的软删记忆。
func (s *MemoryService) Restore(ctx context.Context, userID int64, confirm bool, memoryType string, start, end *time.Time) (int64, error) {
	if !confirm {
		return 0, common.NewBizError(common.PARAM_ERROR, "恢复记忆操作需二次确认")
	}
	items, err := s.memories.ListDeletedForRestore(ctx, userID, memoryType, start, end)
	if err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询可恢复记忆失败", err)
	}
	if len(items) == 0 {
		return 0, nil
	}
	ids := make([]int64, 0, len(items))
	for _, item := range items {
		ids = append(ids, item.ID)
	}
	count, err := s.memories.RestoreDeleted(ctx, ids)
	if err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "恢复记忆失败", err)
	}
	return count, nil
}

// Export 导出全部活跃记忆（json/markdown），批量导出留审计。
func (s *MemoryService) Export(ctx context.Context, userID int64, format string) (contentType, filename, content string, err error) {
	items, err := s.memories.ListForExport(ctx, userID, 10000)
	if err != nil {
		return "", "", "", common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询记忆失败", err)
	}
	s.recordAudit(ctx, userID, "export", map[string]any{"format": format, "count": len(items)})

	if format == "markdown" {
		lines := []string{"# 长期记忆导出", ""}
		for _, m := range items {
			createdAt := "-"
			if !m.CreateTime.IsZero() {
				createdAt = formatTime(m.CreateTime)
			}
			lines = append(lines,
				"## "+m.MemoryType+"（来源："+m.Source+"）",
				"- 内容："+m.Content,
				"- 重要性："+strconv.Itoa(m.Importance),
				"- 创建时间："+createdAt,
				"",
			)
		}
		return "text/markdown; charset=utf-8", "memories.md", strings.Join(lines, "\n"), nil
	}

	records := make([]map[string]any, 0, len(items))
	for _, m := range items {
		records = append(records, map[string]any{
			"id":           m.ID,
			"memory_type":  m.MemoryType,
			"content":      m.Content,
			"metadata":     rawJSONOrNil(m.Metadata),
			"source":       m.Source,
			"importance":   m.Importance,
			"access_count": m.AccessCount,
			"created_at":   formatTimeValue(ptrTime(m.CreateTime)),
		})
	}
	body := map[string]any{
		"user_id":     userID,
		"exported_at": time.Now().Format(time.RFC3339),
		"memories":    records,
	}
	return "application/json; charset=utf-8", "memories.json", marshalIndent(body), nil
}

func (s *MemoryService) recordAudit(ctx context.Context, userID int64, action string, after map[string]any) {
	if s.auditLog == nil {
		return
	}
	s.auditLog.RecordAuditAsync(ctx, userID, "ai_memory", userID, action, "ai_memory", nil, after, "", "")
}

// maxMemoryResults 导出/搜索上限（python 导出上限 10000）。
const maxMemoryResults = 10000

func toMemoryVO(memory *model.SysAiMemory) *MemoryVO {
	return &MemoryVO{
		ID:             memory.ID,
		UserID:         memory.UserID,
		MemoryType:     memory.MemoryType,
		Content:        memory.Content,
		Metadata:       rawJSON(memory.Metadata),
		Importance:     memory.Importance,
		AccessCount:    memory.AccessCount,
		LastAccessedAt: formatTimePtr(memory.LastAccessedAt),
		Source:         memory.Source,
		Status:         memory.Status,
		Archived:       memory.Archived,
		CreateTime:     formatTime(memory.CreateTime),
		UpdateTime:     formatTimePtr(memory.UpdateTime),
	}
}

func toMemoryVOs(items []model.SysAiMemory) []MemoryVO {
	result := make([]MemoryVO, 0, len(items))
	for i := range items {
		result = append(result, *toMemoryVO(&items[i]))
	}
	return result
}

func rawJSONOrNil(s string) any {
	if s == "" {
		return nil
	}
	return json.RawMessage(s)
}

func marshalIndent(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(b)
}

func formatTimeValue(t *time.Time) any {
	if t == nil {
		return nil
	}
	return formatTime(*t)
}

func ptrTime(t time.Time) *time.Time {
	if t.IsZero() {
		return nil
	}
	return &t
}
