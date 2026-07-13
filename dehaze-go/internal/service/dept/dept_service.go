package dept

import (
	"context"
	"encoding/json"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	"github.com/earthyzinc/dehaze-go/internal/service/mapper"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

const (
	// DEPT_TREE_KEY 部门树缓存key
	DEPT_TREE_KEY = "dept:tree"
	// DEPT_OPTIONS_KEY 部门选项缓存key
	DEPT_OPTIONS_KEY = "dept:options"
	// DEPT_CACHE_TTL 部门缓存过期时间（1小时）
	DEPT_CACHE_TTL = time.Hour
	// ROOT_DEPT_ID 根部门ID，不可删除或修改上级
	ROOT_DEPT_ID = 1
	// MAX_DEPT_DEPTH 部门最大层级深度
	MAX_DEPT_DEPTH = 5
)

// DeptService 部门服务
type DeptService struct {
	deptRepo deptrepo.IDeptRepository
	cache    types.ICache
}

// NewDeptService 创建部门服务实例
func NewDeptService(cache types.ICache, deptRepo deptrepo.IDeptRepository) *DeptService {
	return &DeptService{
		cache:    cache,
		deptRepo: deptRepo,
	}
}

// ====================
// IDeptService 接口实现
// ====================

// GetList 获取部门列表
func (s *DeptService) GetList(ctx context.Context, q *query.DeptQuery) ([]vo.DeptVO, error) {
	// 只有未过滤的查询才使用缓存
	isUnfiltered := q == nil || (q.Keywords == "" && q.Status == nil)

	// 尝试从缓存获取部门树（仅未过滤查询）
	if isUnfiltered && s.cache != nil {
		cached, err := s.cache.Get(ctx, DEPT_TREE_KEY)
		if err == nil && cached != "" {
			var deptVOs []vo.DeptVO
			if jsonErr := json.Unmarshal([]byte(cached), &deptVOs); jsonErr == nil {
				return deptVOs, nil
			}
		}
	}

	deptList, err := s.deptRepo.FindAll(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询部门列表失败", err)
	}

	if len(deptList) == 0 {
		return []vo.DeptVO{}, nil
	}

	deptIds := make(map[int64]bool)
	for _, dept := range deptList {
		deptIds[dept.ID] = true
	}

	parentIds := make(map[int64]bool)
	for _, dept := range deptList {
		parentIds[dept.ParentID] = true
	}

	var rootIds []int64
	for parentId := range parentIds {
		if _, exists := deptIds[parentId]; !exists {
			rootIds = append(rootIds, parentId)
		}
	}

	var deptVOs []vo.DeptVO
	for _, rootId := range rootIds {
		children := s.recurDeptList(rootId, deptList)
		deptVOs = append(deptVOs, children...)
	}

	// 只有未过滤的查询才写入缓存
	if isUnfiltered && s.cache != nil {
		if data, jsonErr := json.Marshal(deptVOs); jsonErr == nil {
			_ = s.cache.Set(ctx, DEPT_TREE_KEY, string(data), DEPT_CACHE_TTL)
		}
	}

	return deptVOs, nil
}

// 递归生成部门树形列表
func (s *DeptService) recurDeptList(parentId int64, deptList []model.SysDept) []vo.DeptVO {
	var result []vo.DeptVO
	for _, dept := range deptList {
		if dept.ParentID == parentId {
			deptVO := vo.DeptVO{
				ID:         dept.ID,
				ParentID:   dept.ParentID,
				Name:       dept.Name,
				Sort:       dept.Sort,
				Status:     dept.Status,
				CreateTime: dept.CreatedAt,
				UpdateTime: dept.UpdatedAt,
			}
			children := s.recurDeptList(dept.ID, deptList)
			deptVO.Children = children
			result = append(result, deptVO)
		}
	}
	return result
}

// GetOptions 获取部门下拉选项
func (s *DeptService) GetOptions(ctx context.Context) ([]vo.Option, error) {
	// 尝试从缓存获取
	if s.cache != nil {
		cached, err := s.cache.Get(ctx, DEPT_OPTIONS_KEY)
		if err == nil && cached != "" {
			var options []vo.Option
			if jsonErr := json.Unmarshal([]byte(cached), &options); jsonErr == nil {
				return options, nil
			}
		}
	}

	readOptions, err := s.deptRepo.GetOptions(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询部门选项失败", err)
	}

	options := mapper.OptionsFromRead(readOptions)

	// 写入缓存
	if s.cache != nil {
		if data, jsonErr := json.Marshal(options); jsonErr == nil {
			_ = s.cache.Set(ctx, DEPT_OPTIONS_KEY, string(data), DEPT_CACHE_TTL)
		}
	}

	return options, nil
}

// GetFormData 获取部门表单数据
func (s *DeptService) GetFormData(ctx context.Context, id int64) (*bo.DeptFormBO, error) {
	form, err := s.deptRepo.GetFormData(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询部门表单数据失败", err)
	}
	return form, nil
}

// Create 创建部门
func (s *DeptService) Create(ctx context.Context, form *bo.DeptFormBO) (int64, error) {
	// 校验同一父部门下名称是否唯一
	depts, err := s.deptRepo.FindAll(ctx, &query.DeptQuery{})
	if err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "查询部门列表失败", err)
	}
	for _, dept := range depts {
		if dept.Name == form.Name && dept.ParentID == form.ParentID {
			return 0, common.NewBizError(common.DATA_EXISTS, "同一层级下部门名称已存在")
		}
	}

	// 校验层级深度限制
	if form.ParentID != 0 {
		depth, err := s.calculateDepth(ctx, form.ParentID, depts)
		if err != nil {
			return 0, err // 直接返回，calculateDepth已经返回BizError
		}
		if depth+1 > MAX_DEPT_DEPTH {
			return 0, common.NewBizError(common.BUSINESS_ERROR, "部门层级超过5级限制")
		}
	}

	// 生成部门路径
	treePath, err := s.generateDeptTreePath(ctx, form.ParentID)
	if err != nil {
		return 0, err // 直接返回
	}

	// 创建部门实体
	dept := &model.SysDept{
		Name:     form.Name,
		ParentID: form.ParentID,
		Status:   form.Status,
		Sort:     form.Sort,
		TreePath: treePath,
		Deleted:  0,
	}
	dept.CreatedAt = time.Now()
	dept.UpdatedAt = time.Now()

	if err := s.deptRepo.Create(ctx, dept); err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "创建部门失败", err)
	}

	// 清除缓存
	s.clearCache(ctx)

	return dept.ID, nil
}

// Update 更新部门
func (s *DeptService) Update(ctx context.Context, id int64, form *bo.DeptFormBO) error {
	// 根部门保护：禁止修改上级部门
	if id == ROOT_DEPT_ID && form.ParentID != 0 {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "根部门不能修改上级部门")
	}

	// 校验同一父部门下名称是否唯一（排除当前部门）
	depts, err := s.deptRepo.FindAll(ctx, &query.DeptQuery{})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询部门列表失败", err)
	}
	for _, dept := range depts {
		if dept.Name == form.Name && dept.ParentID == form.ParentID && dept.ID != id {
			return common.NewBizError(common.DATA_EXISTS, "同一层级下部门名称已存在")
		}
	}

	// 检测循环引用：不能将部门移动到自身或其子部门下
	if form.ParentID != 0 { // 根部门不需要检测
		isChild, err := s.isChildOrSelf(ctx, id, form.ParentID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检测循环引用失败", err)
		}
		if isChild {
			return common.NewBizError(common.OPERATION_NOT_ALLOW, "不能选择自身或其子部门作为上级部门")
		}
	}

	// 生成部门路径
	treePath, err := s.generateDeptTreePath(ctx, form.ParentID)
	if err != nil {
		return err
	}

	// 查询原部门信息
	oldDept, err := s.deptRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询部门失败", err)
	}
	if oldDept == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "部门不存在")
	}

	// 更新部门
	oldDept.Name = form.Name
	oldDept.ParentID = form.ParentID
	oldDept.Status = form.Status
	oldDept.Sort = form.Sort
	oldDept.TreePath = treePath
	oldDept.UpdatedAt = time.Now()

	if err := s.deptRepo.Update(ctx, oldDept); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新部门失败", err)
	}

	// 清除缓存
	s.clearCache(ctx)

	return nil
}

// Delete 删除部门（支持批量，级联删除子部门）
func (s *DeptService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "删除的部门ID不能为空")
	}

	// 根部门保护
	for _, id := range ids {
		if id == ROOT_DEPT_ID {
			return common.NewBizError(common.OPERATION_NOT_ALLOW, "根部门不能删除")
		}
	}

	// 获取所有部门用于查找子部门和验证存在性
	depts, err := s.deptRepo.FindAll(ctx, &query.DeptQuery{})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询部门列表失败", err)
	}

	// 检查部门是否存在
	existingMap := make(map[int64]bool)
	for _, dept := range depts {
		existingMap[dept.ID] = true
	}
	for _, id := range ids {
		if !existingMap[id] {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "部门不存在")
		}
	}

	// 构建父子关系映射，收集所有子部门ID（级联删除）
	childrenMap := make(map[int64][]int64)
	for _, dept := range depts {
		childrenMap[dept.ParentID] = append(childrenMap[dept.ParentID], dept.ID)
	}

	allIDs := make(map[int64]bool)
	var collectChildren func(id int64)
	collectChildren = func(id int64) {
		if allIDs[id] {
			return
		}
		allIDs[id] = true
		for _, childID := range childrenMap[id] {
			collectChildren(childID)
		}
	}
	for _, id := range ids {
		collectChildren(id)
	}

	// 检查是否有关联用户
	for id := range allIDs {
		hasUsers, err := s.deptRepo.HasUsers(ctx, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查关联用户失败", err)
		}
		if hasUsers {
			return common.NewBizError(common.DATA_BIND_EXISTS, "部门存在关联用户，不能删除")
		}
	}

	// 转换为切片
	idList := make([]int64, 0, len(allIDs))
	for id := range allIDs {
		idList = append(idList, id)
	}

	// 批量删除
	if err := s.deptRepo.Delete(ctx, idList); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除部门失败", err)
	}

	// 清除缓存
	s.clearCache(ctx)

	return nil
}

// generateDeptTreePath 生成部门树形路径
func (s *DeptService) generateDeptTreePath(ctx context.Context, parentID int64) (string, error) {
	const rootNodeID int64 = 0

	if parentID == rootNodeID {
		return strconv.FormatInt(parentID, 10), nil
	}

	parent, err := s.deptRepo.FindByID(ctx, parentID)
	if err != nil {
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询父部门失败", err)
	}
	if parent == nil {
		return "", common.NewBizError(common.RESOURCE_NOT_FOUND, "父部门不存在")
	}

	return parent.TreePath + "," + strconv.FormatInt(parent.ID, 10), nil
}

// clearCache 清除部门相关缓存
func (s *DeptService) clearCache(ctx context.Context) {
	if s.cache != nil {
		_ = s.cache.Delete(ctx, DEPT_TREE_KEY)
		_ = s.cache.Delete(ctx, DEPT_OPTIONS_KEY)
	}
}

// isChildOrSelf 检测 targetID 是否是 deptID 或其子部门
// 用于循环引用检测：如果 targetID 是 deptID 或其子部门，则不能将 deptID 移动到 targetID 下
func (s *DeptService) isChildOrSelf(ctx context.Context, deptID, targetID int64) (bool, error) {
	// 自身检测
	if deptID == targetID {
		return true, nil
	}

	// 获取所有部门
	depts, err := s.deptRepo.FindAll(ctx, &query.DeptQuery{})
	if err != nil {
		return false, err
	}

	// 构建父子关系映射
	childrenMap := make(map[int64][]int64)
	for _, dept := range depts {
		childrenMap[dept.ParentID] = append(childrenMap[dept.ParentID], dept.ID)
	}

	// 递归检测 targetID 是否是 deptID 的子部门
	var checkChildren func(parentID int64) bool
	checkChildren = func(parentID int64) bool {
		children, exists := childrenMap[parentID]
		if !exists {
			return false
		}
		for _, childID := range children {
			if childID == targetID {
				return true
			}
			if checkChildren(childID) {
				return true
			}
		}
		return false
	}

	return checkChildren(deptID), nil
}

// calculateDepth 计算指定部门的层级深度
// deptID=0 时返回 0（根级别），deptID=1 时返回 1，依此类推
func (s *DeptService) calculateDepth(ctx context.Context, deptID int64, depts []model.SysDept) (int, error) {
	if deptID == 0 {
		return 0, nil
	}

	// 构建部门ID到父ID的映射
	parentMap := make(map[int64]int64)
	for _, dept := range depts {
		parentMap[dept.ID] = dept.ParentID
	}

	depth := 0
	currentID := deptID

	// 沿着父级链向上遍历，计算深度
	for currentID != 0 {
		depth++
		parentID, exists := parentMap[currentID]
		if !exists {
			// 部门不存在，可能是数据异常
			return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "部门不存在")
		}
		currentID = parentID
	}

	return depth, nil
}
