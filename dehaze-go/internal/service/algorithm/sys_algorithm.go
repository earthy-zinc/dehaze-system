package algorithm

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/mapper"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	predlog "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
)

// AlgorithmService 算法服务
type AlgorithmService struct {
	algorithmRepo algorepo.IAlgorithmRepository
	predLogRepo   predlog.IPredLogRepository
}

// NewAlgorithmService 创建算法服务实例
func NewAlgorithmService(algorithmRepo algorepo.IAlgorithmRepository, predLogRepo predlog.IPredLogRepository) *AlgorithmService {
	return &AlgorithmService{algorithmRepo: algorithmRepo, predLogRepo: predLogRepo}
}

// ====================
// IAlgorithmService 接口实现
// ====================

// GetPage 算法分页列表
func (s *AlgorithmService) GetPage(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error) {
	readResult, err := s.algorithmRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法分页列表失败", err)
	}
	if readResult == nil {
		return &vo.PageResult[vo.AlgorithmVO]{List: []vo.AlgorithmVO{}, Total: 0}, nil
	}

	voList := make([]vo.AlgorithmVO, 0, len(readResult.List))
	for _, item := range readResult.List {
		voList = append(voList, vo.AlgorithmVO{
			ID:          item.ID,
			ParentID:    item.ParentID,
			Name:        item.Name,
			Type:        item.Type,
			Img:         item.Img,
			Description: item.Description,
			Path:        item.Path,
			Flops:       item.Flops,
			Params:      item.Params,
			ImportPath:  item.ImportPath,
			Status:      item.Status,
			Size:        item.Size,
			Children:    mapAlgorithmReadChildren(item.Children),
		})
	}

	return &vo.PageResult[vo.AlgorithmVO]{
		List:  voList,
		Total: readResult.Total,
	}, nil
}

// GetTree 获取算法树形列表（对齐 Java 树形表格格式）
func (s *AlgorithmService) GetTree(ctx context.Context, q *query.AlgorithmQuery) ([]vo.AlgorithmVO, error) {
	algorithms, err := s.algorithmRepo.FindAll(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法列表失败", err)
	}

	// 构建树形结构（parent_id == 0 为根节点）
	tree := make([]vo.AlgorithmVO, 0)
	for _, algo := range algorithms {
		if algo.ParentID == 0 {
			tree = append(tree, mapAlgorithmToVO(algo, algorithms))
		}
	}
	return tree, nil
}

func mapAlgorithmToVO(algo read.Algorithm, all []read.Algorithm) vo.AlgorithmVO {
	voItem := vo.AlgorithmVO{
		ID:          algo.ID,
		ParentID:    algo.ParentID,
		Name:        algo.Name,
		Type:        algo.Type,
		Img:         algo.Img,
		Description: algo.Description,
		Path:        algo.Path,
		Flops:       algo.Flops,
		Params:      algo.Params,
		ImportPath:  algo.ImportPath,
		Status:      algo.Status,
		Size:        algo.Size,
	}
	for _, child := range all {
		if child.ParentID == algo.ID {
			voItem.Children = append(voItem.Children, mapAlgorithmToVO(child, all))
		}
	}
	return voItem
}

// GetOptions 获取算法下拉选项
func (s *AlgorithmService) GetOptions(ctx context.Context) ([]vo.Option, error) {
	readOptions, err := s.algorithmRepo.FindOptions(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法选项失败", err)
	}

	return mapper.OptionsFromRead(readOptions), nil
}

// GetFormData 获取算法表单数据
func (s *AlgorithmService) GetFormData(ctx context.Context, id int64) (*bo.AlgorithmFormBO, error) {
	algorithm, err := s.algorithmRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
	}

	form := &bo.AlgorithmFormBO{
		ID:          algorithm.ID,
		ParentID:    algorithm.ParentID,
		Type:        algorithm.Type,
		Name:        algorithm.Name,
		Path:        algorithm.Path,
		ImportPath:  algorithm.ImportPath,
		Description: algorithm.Description,
		Status:      algorithm.Status,
	}

	return form, nil
}

// Create 创建算法
func (s *AlgorithmService) Create(ctx context.Context, form *bo.AlgorithmFormBO) (int64, error) {
	// 如果父节点ID不为0，检查父节点是否存在
	if form.ParentID != 0 {
		parentAlgorithm, err := s.algorithmRepo.FindByID(ctx, form.ParentID)
		if err != nil {
			return 0, common.WrapBizError(common.DATABASE_ERROR, "查询父算法失败", err)
		}
		if parentAlgorithm == nil {
			return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "父算法不存在")
		}
	}

	algorithm := &model.SysAlgorithm{
		ParentID:    form.ParentID,
		Type:        form.Type,
		Name:        form.Name,
		Path:        form.Path,
		ImportPath:  form.ImportPath,
		Description: form.Description,
		Status:      int8(form.Status),
	}

	if err := s.algorithmRepo.Create(ctx, algorithm); err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "创建算法失败", err)
	}
	return algorithm.ID, nil
}

// Update 更新算法
func (s *AlgorithmService) Update(ctx context.Context, id int64, form *bo.AlgorithmFormBO) error {
	// 校验算法是否存在
	oldAlgorithm, err := s.algorithmRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if oldAlgorithm == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
	}

	// 如果父节点ID不为0，检查父节点是否存在
	if form.ParentID != 0 && form.ParentID != id {
		parentAlgorithm, err := s.algorithmRepo.FindByID(ctx, form.ParentID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询父算法失败", err)
		}
		if parentAlgorithm == nil {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "父算法不存在")
		}
	}

	// 更新算法信息
	oldAlgorithm.ParentID = form.ParentID
	oldAlgorithm.Type = form.Type
	oldAlgorithm.Name = form.Name
	oldAlgorithm.Path = form.Path
	oldAlgorithm.ImportPath = form.ImportPath
	oldAlgorithm.Description = form.Description
	oldAlgorithm.Status = int8(form.Status)

	if err := s.algorithmRepo.Update(ctx, oldAlgorithm); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新算法失败", err)
	}
	return nil
}

// Delete 删除算法
func (s *AlgorithmService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "请选择要删除的算法")
	}

	// 校验算法是否存在
	for _, id := range ids {
		algorithm, err := s.algorithmRepo.FindByID(ctx, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
		}
		if algorithm == nil {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
		}
	}

	// 检查是否有子算法
	hasChildren, err := s.algorithmRepo.HasChildrenByParentIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查子算法失败", err)
	}

	if hasChildren {
		return common.NewBizError(common.DATA_BIND_EXISTS, "存在子算法，无法删除")
	}

	if err := s.algorithmRepo.Delete(ctx, ids); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除算法失败", err)
	}
	return nil
}

// UpdateStatus 更新算法状态（含状态流转校验）
func (s *AlgorithmService) UpdateStatus(ctx context.Context, id int64, status int8) error {
	// 1. 查询当前算法
	algorithm, err := s.algorithmRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
	}

	// 2. 校验状态流转合法性
	if !bo.CanTransitionTo(algorithm.Status, status) {
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW,
			fmt.Sprintf("不允许将算法状态从 %d 变更为 %d", algorithm.Status, status))
	}

	// 3. 执行状态更新
	if err := s.algorithmRepo.UpdateStatus(ctx, id, status); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新算法状态失败", err)
	}
	return nil
}

// Compare 批量查询算法用于对比
func (s *AlgorithmService) Compare(ctx context.Context, ids []int64) ([]model.SysAlgorithm, error) {
	if len(ids) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "算法ID列表不能为空")
	}
	var algorithms []model.SysAlgorithm
	for _, id := range ids {
		a, err := s.algorithmRepo.FindByID(ctx, id)
		if err != nil {
			continue
		}
		algorithms = append(algorithms, *a)
	}
	return algorithms, nil
}

// GetVersionHistory 获取算法版本历史
// 查询 sys_algorithm_version 表，按 create_time 降序排序
// 表不存在或查询失败时返回空数组（兼容性处理）
func (s *AlgorithmService) GetVersionHistory(ctx context.Context, algorithmID int64) ([]vo.AlgorithmVersionVO, error) {
	versions, err := s.algorithmRepo.FindVersionsByAlgorithmID(ctx, algorithmID)
	if err != nil {
		return []vo.AlgorithmVersionVO{}, nil
	}

	result := make([]vo.AlgorithmVersionVO, 0, len(versions))
	for _, v := range versions {
		var isActive *bool
		if v.IsActive != nil {
			b := *v.IsActive != 0
			isActive = &b
		}
		result = append(result, vo.AlgorithmVersionVO{
			ID:          v.ID,
			AlgorithmID: v.AlgorithmID,
			Version:     v.Version,
			ChangeLog:   v.ChangeLog,
			Status:      v.Status,
			IsActive:    isActive,
			ModelFileID: v.ModelFileID,
			CreateTime:  v.CreatedAt,
		})
	}
	return result, nil
}

// GetMonitorData 获取算法监控数据
// 查询 sys_pred_log 表统计；表不存在或查询失败时返回零值（successRate=100）
func (s *AlgorithmService) GetMonitorData(ctx context.Context, algorithmID int64) (*vo.AlgorithmMonitorVO, error) {
	monitor := &vo.AlgorithmMonitorVO{
		CallCount:      0,
		AvgTime:        0,
		SuccessRate:    100.0,
		TodayCallCount: 0,
	}

	stats, err := s.predLogRepo.GetMonitorStats(ctx, algorithmID)
	if err != nil {
		return monitor, nil
	}

	monitor.CallCount = stats.CallCount
	monitor.TodayCallCount = stats.TodayCallCount
	monitor.AvgTime = math.Round(stats.AvgTime*100) / 100
	if stats.CallCount > 0 {
		rate := float64(stats.SuccessCount) / float64(stats.CallCount) * 100
		monitor.SuccessRate = math.Round(rate*100) / 100
	}
	return monitor, nil
}

// algorithmExportData 算法导出数据结构（字段顺序对齐 Java 导出格式）
type algorithmExportData struct {
	FormatVersion string  `json:"formatVersion"`
	Name          string  `json:"name"`
	Type          string  `json:"type"`
	ParentName    string  `json:"parentName"`
	Version       *string `json:"version"`
	Description   string  `json:"description"`
	ImportPath    string  `json:"importPath"`
	Flops         string  `json:"flops"`
	Params        string  `json:"params"`
	Status        int8    `json:"status"`
	StatusLabel   string  `json:"statusLabel"`
	ExportTime    string  `json:"exportTime"`
}

// algorithmStatusLabel 返回算法状态对应的中文标签（对齐 Java AlgorithmStatusEnum）
func algorithmStatusLabel(status int8) string {
	switch status {
	case 0:
		return "草稿"
	case 1:
		return "测试中"
	case 2:
		return "待审核"
	case 3:
		return "已发布"
	case 4:
		return "已停用"
	case 5:
		return "已归档"
	default:
		return ""
	}
}

// ExportAlgorithmJson 导出算法为 JSON 字符串
func (s *AlgorithmService) ExportAlgorithmJson(ctx context.Context, id int64) (string, error) {
	algorithm, err := s.algorithmRepo.FindByID(ctx, id)
	if err != nil {
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil {
		return "", common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
	}

	// 获取父算法名称用于导入参考
	parentName := ""
	if algorithm.ParentID > 0 {
		parent, err := s.algorithmRepo.FindByID(ctx, algorithm.ParentID)
		if err == nil && parent != nil {
			parentName = parent.Name
		}
	}

	data := algorithmExportData{
		FormatVersion: "1.0",
		Name:          algorithm.Name,
		Type:          algorithm.Type,
		ParentName:    parentName,
		Version:       algorithm.Version,
		Description:   algorithm.Description,
		ImportPath:    algorithm.ImportPath,
		Flops:         algorithm.Flops,
		Params:        algorithm.Params,
		Status:        algorithm.Status,
		StatusLabel:   algorithmStatusLabel(algorithm.Status),
		ExportTime:    time.Now().Format("2006-01-02T15:04:05"),
	}

	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return "", common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "序列化算法数据失败", err)
	}
	return string(bytes), nil
}

// ValidateImport 校验导入文件
//   - 空文件返回业务错误
//   - 非 .json 后缀返回业务错误
//   - 解析 JSON，校验 name 和 type 字段非空
//   - 成功返回 "校验通过: 算法名称=xxx, 类型=xxx"
func (s *AlgorithmService) ValidateImport(ctx context.Context, filename string, content []byte) (string, error) {
	if len(content) == 0 {
		return "", common.NewBizError(common.PARAM_ERROR, "导入文件不能为空")
	}

	if !strings.HasSuffix(strings.ToLower(filename), ".json") {
		return "", common.NewBizError(common.PARAM_ERROR, "仅支持 .json 格式的算法导出文件")
	}

	var data map[string]interface{}
	if err := json.Unmarshal(content, &data); err != nil {
		return "", common.NewBizError(common.PARAM_ERROR, "导入文件解析失败: "+err.Error())
	}

	name, _ := data["name"].(string)
	if strings.TrimSpace(name) == "" {
		return "", common.NewBizError(common.PARAM_ERROR, "导入文件缺少必填字段: name")
	}

	typ, _ := data["type"].(string)
	if strings.TrimSpace(typ) == "" {
		return "", common.NewBizError(common.PARAM_ERROR, "导入文件缺少必填字段: type")
	}

	return "校验通过: 算法名称=" + name + ", 类型=" + typ, nil
}

func mapAlgorithmReadChildren(children []read.Algorithm) []vo.AlgorithmVO {
	if len(children) == 0 {
		return []vo.AlgorithmVO{}
	}

	result := make([]vo.AlgorithmVO, 0, len(children))
	for _, child := range children {
		result = append(result, vo.AlgorithmVO{
			ID:          child.ID,
			ParentID:    child.ParentID,
			Name:        child.Name,
			Type:        child.Type,
			Img:         child.Img,
			Description: child.Description,
			Path:        child.Path,
			Flops:       child.Flops,
			Params:      child.Params,
			ImportPath:  child.ImportPath,
			Status:      child.Status,
			Size:        child.Size,
			Children:    mapAlgorithmReadChildren(child.Children),
		})
	}

	return result
}

