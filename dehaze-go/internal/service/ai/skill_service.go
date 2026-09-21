package ai

import (
	"context"
	"io"
	"regexp"
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
)

// contentMaxBytes Skill 指令内容上限（100KB）
const contentMaxBytes = 100 * 1024

// dangerousPattern 危险操作正则（与 dehaze-python ai_skill.DANGEROUS_PATTERN 同源）
var dangerousPattern = regexp.MustCompile(`(?i)(rm\s+-rf\s*/|mkfs\.?(ext\d?|xfs|vfat|ntfs)?\b|curl[^\n]*\|\s*(ba)?sh\b|wget[^\n]*\|\s*(ba)?sh\b|sudo\s+(rm|shutdown|reboot|mkfs|dd)|dd\s+if=.*of=/dev/)`)

// skillObjectPrefix SKILL 资源对象前缀：skills/{skill_id}/{path}（与 name 解耦，改名不影响定位）
const skillObjectPrefix = "skills"

// SkillService Skills 管理（F-M08-006）
type SkillService struct {
	repo    *airepo.SkillRepository
	storage *storage.Registry
}

func NewSkillService(repo *airepo.SkillRepository, registry *storage.Registry) *SkillService {
	return &SkillService{repo: repo, storage: registry}
}

// ListSkills 列表：管理员全量分页（含禁用，支持状态筛选）；普通用户仅启用项（不分页全量）
func (s *SkillService) ListSkills(ctx context.Context, q *bo.SkillQuery, enabledOnly bool) (*vo.PageResult[vo.SkillVO], error) {
	page, size := q.PageNum, q.PageSize
	var status *int
	if q.Status != nil {
		value := int(*q.Status)
		status = &value
	}
	if enabledOnly {
		items, err := s.repo.ListEnabled(ctx)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 Skill 列表失败", err)
		}
		rows, err := s.toListItems(ctx, items)
		if err != nil {
			return nil, err
		}
		return &vo.PageResult[vo.SkillVO]{List: rows, Total: int64(len(rows))}, nil
	}
	items, total, err := s.repo.Paginate(ctx, page, size, q.Keyword, status)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 Skill 列表失败", err)
	}
	rows, err := s.toListItems(ctx, items)
	if err != nil {
		return nil, err
	}
	return &vo.PageResult[vo.SkillVO]{List: rows, Total: total}, nil
}

// ListMarket Skill 市场目录：已共享且启用的 Skill
func (s *SkillService) ListMarket(ctx context.Context) ([]vo.SkillMarketVO, error) {
	skills, err := s.repo.ListMarket(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 Skill 市场失败", err)
	}
	names := make([]string, 0, len(skills))
	for i := range skills {
		names = append(names, skills[i].Name)
	}
	refs, err := s.repo.CountByNames(ctx, names)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "统计 Skill 关联数失败", err)
	}
	items := make([]vo.SkillMarketVO, 0, len(skills))
	for i := range skills {
		items = append(items, vo.SkillMarketVO{
			SkillID:     skills[i].ID,
			Name:        skills[i].Name,
			Description: skills[i].Description,
			Scene:       skills[i].Scene,
			Enabled:     skills[i].Status == 1,
			AgentCount:  refs[skills[i].Name],
		})
	}
	return items, nil
}

// GetSkill 详情（含指令全文）；enabledOnly=true 时禁用项按不存在处理
func (s *SkillService) GetSkill(ctx context.Context, skillID int64, enabledOnly bool) (*vo.SkillVO, error) {
	skill, err := s.requireSkill(ctx, skillID, enabledOnly)
	if err != nil {
		return nil, err
	}
	result, err := s.toDetail(ctx, skill)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

// GetSkillFile 读取 SKILL 资源文件（path 须命中该 Skill 文件清单，防任意对象读取）
func (s *SkillService) GetSkillFile(ctx context.Context, skillID int64, path string, enabledOnly bool) ([]byte, error) {
	skill, err := s.requireSkill(ctx, skillID, enabledOnly)
	if err != nil {
		return nil, err
	}
	files, err := s.repo.ListFiles(ctx, skill.ID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 SKILL 文件清单失败", err)
	}
	found := false
	for i := range files {
		if files[i].Path == path {
			found = true
			break
		}
	}
	if !found {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "SKILL 文件不存在")
	}
	svc := s.defaultStorage()
	if svc == nil {
		return nil, common.NewBizError(common.SYSTEM_EXECUTION_ERROR, "对象存储不可用")
	}
	reader, err := svc.Download(ctx, skillObjectKey(skill.ID, path))
	if err != nil {
		return nil, common.WrapBizError(common.RESOURCE_NOT_FOUND, "SKILL 文件读取失败", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, common.WrapBizError(common.RESOURCE_NOT_FOUND, "SKILL 文件读取失败", err)
	}
	return data, nil
}

// CreateSkill 创建 Skill：唯一性校验 + 指令内容校验，创建后为禁用态待管理员启用
func (s *SkillService) CreateSkill(ctx context.Context, form *bo.SkillCreateForm, operatorID int64) (*vo.SkillVO, error) {
	if err := validateSkillContent(form.Instruction); err != nil {
		return nil, err
	}
	existing, err := s.repo.GetByName(ctx, form.Name, false)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "校验 Skill 名称失败", err)
	}
	if existing != nil {
		return nil, common.NewBizError(common.DATA_EXISTS, "Skill 名称已存在")
	}
	instruction := form.Instruction
	skill := &model.SysAiSkill{
		Name:        form.Name,
		Description: form.Description,
		Scene:       form.Scene,
		Instruction: &instruction,
		Status:      0,
		Source:      "admin",
	}
	skill.CreateBy = operatorID
	skill.UpdateBy = operatorID
	if err := s.repo.Create(ctx, skill); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建 Skill 失败", err)
	}
	result, err := s.toDetail(ctx, skill)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *SkillService) UpdateSkill(ctx context.Context, skillID int64, form *bo.SkillUpdateForm, operatorID int64) (*vo.SkillVO, error) {
	skill, err := s.requireSkill(ctx, skillID, false)
	if err != nil {
		return nil, err
	}
	if form.Name != nil && *form.Name != skill.Name {
		duplicate, findErr := s.repo.GetByName(ctx, *form.Name, false)
		if findErr != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "校验 Skill 名称失败", findErr)
		}
		if duplicate != nil && duplicate.ID != skillID {
			return nil, common.NewBizError(common.DATA_EXISTS, "Skill 名称已存在")
		}
	}
	if form.Instruction != nil {
		if err := validateSkillContent(*form.Instruction); err != nil {
			return nil, err
		}
	}

	updates := map[string]interface{}{"update_by": operatorID}
	if form.Name != nil {
		updates["name"] = *form.Name
	}
	if form.Description != nil {
		updates["description"] = *form.Description
	}
	if form.Scene != nil {
		updates["scene"] = *form.Scene
	}
	if form.Instruction != nil {
		updates["instruction"] = *form.Instruction
	}
	if err := s.repo.Update(ctx, skillID, updates); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新 Skill 失败", err)
	}
	updated, err := s.requireSkill(ctx, skillID, false)
	if err != nil {
		return nil, err
	}
	result, err := s.toDetail(ctx, updated)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *SkillService) SetStatus(ctx context.Context, skillID int64, status int8, operatorID int64) (*vo.SkillVO, error) {
	skill, err := s.requireSkill(ctx, skillID, false)
	if err != nil {
		return nil, err
	}
	if skill.Status != status {
		if err := s.repo.Update(ctx, skillID, map[string]interface{}{"status": status, "update_by": operatorID}); err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "更新 Skill 状态失败", err)
		}
	}
	updated, err := s.requireSkill(ctx, skillID, false)
	if err != nil {
		return nil, err
	}
	result, err := s.toDetail(ctx, updated)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *SkillService) ShareToMarket(ctx context.Context, skillID, operatorID int64) (*vo.SkillVO, error) {
	skill, err := s.requireSkill(ctx, skillID, false)
	if err != nil {
		return nil, err
	}
	if skill.Status != 1 {
		return nil, common.NewBizError(common.PARAM_ERROR, "Skill 需先启用才能共享至市场")
	}
	if skill.MarketShared != 1 {
		if err := s.repo.Update(ctx, skillID, map[string]interface{}{"market_shared": 1, "update_by": operatorID}); err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "共享 Skill 失败", err)
		}
	}
	updated, err := s.requireSkill(ctx, skillID, false)
	if err != nil {
		return nil, err
	}
	result, err := s.toDetail(ctx, updated)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

// DeleteSkill 软删 Skill（被 Agent 关联时拒绝），并清理对象存储资源与文件清单
func (s *SkillService) DeleteSkill(ctx context.Context, skillID, operatorID int64) error {
	skill, err := s.requireSkill(ctx, skillID, false)
	if err != nil {
		return err
	}
	refs, err := s.repo.CountByNames(ctx, []string{skill.Name})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "统计 Skill 关联数失败", err)
	}
	if refs[skill.Name] > 0 {
		return common.NewBizError(common.DATA_BIND_EXISTS,
			"Skill ["+skill.Name+"] 已被 "+strconv.FormatInt(refs[skill.Name], 10)+" 个 Agent 关联，请先解绑再删除")
	}
	if err := s.repo.SoftDelete(ctx, skillID, operatorID); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除 Skill 失败", err)
	}
	s.cleanupSkillFiles(ctx, skill.ID)
	return nil
}

// cleanupSkillFiles 清理对象存储资源与文件清单（失败不阻断删除主流程）
func (s *SkillService) cleanupSkillFiles(ctx context.Context, skillID int64) {
	files, err := s.repo.ListFiles(ctx, skillID)
	if err == nil {
		if svc := s.defaultStorage(); svc != nil {
			for i := range files {
				_ = svc.Delete(ctx, skillObjectKey(skillID, files[i].Path))
			}
		}
	}
	_ = s.repo.DeleteFiles(ctx, skillID)
}

func (s *SkillService) requireSkill(ctx context.Context, skillID int64, enabledOnly bool) (*model.SysAiSkill, error) {
	skill, err := s.repo.GetByID(ctx, skillID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 Skill 失败", err)
	}
	if skill == nil || (enabledOnly && skill.Status != 1) {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "Skill 不存在")
	}
	return skill, nil
}

func (s *SkillService) toListItems(ctx context.Context, skills []model.SysAiSkill) ([]vo.SkillVO, error) {
	names := make([]string, 0, len(skills))
	for i := range skills {
		names = append(names, skills[i].Name)
	}
	refs, err := s.repo.CountByNames(ctx, names)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "统计 Skill 关联数失败", err)
	}
	items := make([]vo.SkillVO, 0, len(skills))
	for i := range skills {
		items = append(items, vo.SkillVO{
			ID:           skills[i].ID,
			Name:         skills[i].Name,
			Description:  skills[i].Description,
			Scene:        skills[i].Scene,
			Status:       skills[i].Status,
			Source:       skills[i].Source,
			AgentCount:   refs[skills[i].Name],
			MarketShared: skills[i].MarketShared,
			CreateTime:   skills[i].CreatedAt,
			UpdateTime:   skills[i].UpdatedAt,
		})
	}
	return items, nil
}

func (s *SkillService) toDetail(ctx context.Context, skill *model.SysAiSkill) (vo.SkillVO, error) {
	refs, err := s.repo.CountByNames(ctx, []string{skill.Name})
	if err != nil {
		return vo.SkillVO{}, common.WrapBizError(common.DATABASE_ERROR, "统计 Skill 关联数失败", err)
	}
	files, err := s.repo.ListFiles(ctx, skill.ID)
	if err != nil {
		return vo.SkillVO{}, common.WrapBizError(common.DATABASE_ERROR, "查询 SKILL 文件清单失败", err)
	}
	fileVOs := make([]vo.SkillFileVO, 0, len(files))
	for i := range files {
		fileVOs = append(fileVOs, vo.SkillFileVO{
			Path:     files[i].Path,
			FileSize: files[i].FileSize,
			FileType: files[i].FileType,
		})
	}
	return vo.SkillVO{
		ID:            skill.ID,
		Name:          skill.Name,
		Description:   skill.Description,
		Scene:         skill.Scene,
		Instruction:   skill.Instruction,
		License:       skill.License,
		Compatibility: skill.Compatibility,
		Metadata:      skill.Metadata,
		AllowedTools:  skill.AllowedTools,
		Files:         fileVOs,
		Status:        skill.Status,
		Source:        skill.Source,
		AgentCount:    refs[skill.Name],
		MarketShared:  skill.MarketShared,
		CreateTime:    skill.CreatedAt,
		UpdateTime:    skill.UpdatedAt,
	}, nil
}

func (s *SkillService) defaultStorage() storage.StorageService {
	if s.storage == nil {
		return nil
	}
	svc, err := s.storage.Default()
	if err != nil {
		return nil
	}
	return svc
}

func skillObjectKey(skillID int64, relPath string) string {
	return skillObjectPrefix + "/" + strconv.FormatInt(skillID, 10) + "/" + relPath
}

func validateSkillContent(content string) error {
	if len(content) > contentMaxBytes {
		return common.NewBizError(common.PARAM_ERROR, "Skill 指令内容超过 100KB 上限")
	}
	if dangerousPattern.MatchString(content) {
		return common.NewBizError(common.PARAM_ERROR, "指令含危险操作")
	}
	return nil
}
