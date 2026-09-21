package api

import (
	"mime"
	"path/filepath"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

const skillManagePermission = "ai:skill:manage"

// AiSkillApi Skills 管理接口（F-M08-006）
type AiSkillApi struct {
	service *aiservice.SkillService
}

func NewAiSkillApi(service *aiservice.SkillService) *AiSkillApi {
	return &AiSkillApi{service: service}
}

// isSkillManager 管理员判定：ROOT 或命中 ai:skill:manage（普通用户列表/详情仅见启用项）
func isSkillManager(c *gin.Context) bool {
	if security.IsRoot(c) {
		return true
	}
	hasPerm, err := security.HasPermission(c, skillManagePermission)
	return err == nil && hasPerm
}

// ListSkills Skills 列表
func (a *AiSkillApi) ListSkills(c *gin.Context) {
	// 分页字段（AiPageQuery）已 form:"-"，分页只由本 helper 解析：非数字/越界一律 A0400，
	// 对齐 python SkillPageQuery(BasePageQuery)。
	pageNum, pageSize, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	var query bo.SkillQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	query.PageNum, query.PageSize = pageNum, pageSize
	result, err := a.service.ListSkills(c.Request.Context(), &query, !isSkillManager(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListMarket Skill 市场目录
func (a *AiSkillApi) ListMarket(c *gin.Context) {
	result, err := a.service.ListMarket(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetSkill Skill 详情
func (a *AiSkillApi) GetSkill(c *gin.Context) {
	skillID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.GetSkill(c.Request.Context(), skillID, !isSkillManager(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetSkillFile 读取 SKILL 资源文件（返回原始文件内容，非 JSON 信封）
func (a *AiSkillApi) GetSkillFile(c *gin.Context) {
	skillID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	path := c.Query("path")
	data, err := a.service.GetSkillFile(c.Request.Context(), skillID, path, !isSkillManager(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	contentType := mime.TypeByExtension(filepath.Ext(path))
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	c.Data(200, contentType, data)
}

// CreateSkill 创建 Skill
func (a *AiSkillApi) CreateSkill(c *gin.Context) {
	var form bo.SkillCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.CreateSkill(c.Request.Context(), &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateSkill 更新 Skill
func (a *AiSkillApi) UpdateSkill(c *gin.Context) {
	skillID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.SkillUpdateForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.UpdateSkill(c.Request.Context(), skillID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// SetSkillStatus 启停 Skill
func (a *AiSkillApi) SetSkillStatus(c *gin.Context) {
	skillID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.SkillStatusForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.SetStatus(c.Request.Context(), skillID, *form.Status, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ShareToMarket 共享 Skill 至市场
func (a *AiSkillApi) ShareToMarket(c *gin.Context) {
	var form bo.SkillShareForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.ShareToMarket(c.Request.Context(), form.SkillID, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteSkill 删除 Skill
func (a *AiSkillApi) DeleteSkill(c *gin.Context) {
	skillID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.service.DeleteSkill(c.Request.Context(), skillID, security.GetUserID(c)); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}
