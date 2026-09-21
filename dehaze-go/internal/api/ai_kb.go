package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	kbservice "github.com/earthyzinc/dehaze-go/internal/service/kb"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// AiKbApi AI 知识库 A 类查询/编辑接口（库 CRUD 中依赖 ES 索引的创建/删除与 index-stats 由 go-proxy 转发）
type AiKbApi struct {
	service *kbservice.Service
}

func NewAiKbApi(service *kbservice.Service) *AiKbApi {
	return &AiKbApi{service: service}
}

// ListKnowledgeBases 知识库列表（view=admin 需 kb:audit）
func (a *AiKbApi) ListKnowledgeBases(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	var query bo.KnowledgeBasePageQuery
	if bindErr := c.ShouldBindQuery(&query); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	// 分页口径（python `Query(default=1, ge=1)` / `Query(default=10, ge=1, le=100)`）：缺省取 1/10，
	// 越界与非数字（pageNum=abc）一律 A0400。AiPageQuery 的分页字段已 `form:"-"` 不参与 gin 绑定，
	// 故本次解析即唯一来源，越界/非法只有这一处实现。
	page, size, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	adminView := query.View == "admin"
	if adminView {
		if permErr := middleware.CheckPermission(c, "kb:audit"); permErr != nil {
			_ = c.Error(permErr)
			return
		}
	}
	result, err := a.service.ListKnowledgeBases(c.Request.Context(), userID, &query, adminView)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetKnowledgeBase 知识库详情
func (a *AiKbApi) GetKnowledgeBase(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	kbID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.GetKnowledgeBase(c.Request.Context(), kbID, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateKnowledgeBase 编辑知识库（返回更新后的完整 VO）
func (a *AiKbApi) UpdateKnowledgeBase(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	kbID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.KnowledgeBaseUpdateForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	if err := a.service.UpdateKnowledgeBase(c.Request.Context(), kbID, userID, security.IsAdmin(c), &form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.GetKnowledgeBase(c.Request.Context(), kbID, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListDocuments 知识库文档列表
func (a *AiKbApi) ListDocuments(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	kbID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var query bo.KnowledgeDocumentQuery
	if bindErr := c.ShouldBindQuery(&query); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	// 同 ListKnowledgeBases：缺省/越界/非数字均由 parsePaginationWithSize 裁决（service 不再兜底归一化）
	page, size, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	result, err := a.service.ListDocuments(c.Request.Context(), kbID, userID, &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetDocument 文档详情（含解析后 content）
func (a *AiKbApi) GetDocument(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	documentID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.GetDocument(c.Request.Context(), documentID, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListDocumentChunks 文档分块列表（chunk_index 正序分页）
func (a *AiKbApi) ListDocumentChunks(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	documentID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	// 分页口径对齐 python `Query(default=1, ge=1)` / `Query(default=10, ge=1, le=100)`：缺省 1/10，
	// 显式传非数字、<1 或 >100 一律 A0400。原 parseAiQueryInt 调用忽略 ok 标志，会把 pageSize=500
	// 原样透传成 500 行、pageNum=0 透传成负偏移，与 python 报 A0400 形成客户端可触发的分叉。
	page, size, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	result, err := a.service.ListChunks(c.Request.Context(), documentID, userID, page, size)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreateTestSet 创建召回测试集
func (a *AiKbApi) CreateTestSet(c *gin.Context) {
	kbID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.TestSetCreateForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.CreateTestSet(c.Request.Context(), kbID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListTestSets 召回测试集分页列表
func (a *AiKbApi) ListTestSets(c *gin.Context) {
	kbID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	page, size, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	result, err := a.service.ListTestSets(c.Request.Context(), kbID, page, size)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListLowQuality 低质量片段列表（被点踩片段）
func (a *AiKbApi) ListLowQuality(c *gin.Context) {
	kbID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	page, size, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	result, err := a.service.ListLowQuality(c.Request.Context(), kbID, page, size)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// parseQueryInt 解析整型 query 参数，缺省/非法时回退默认值
func parseAiQueryInt(c *gin.Context, name string, fallback int) (int, bool) {
	raw := c.Query(name)
	if raw == "" {
		return fallback, false
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback, false
	}
	return value, true
}
