package service

import (
	"context"
	"errors"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// DeptService 部门服务
type DeptService struct {
	deptRepo repository.IDeptRepository
}

// NewDeptService 创建部门服务实例
func NewDeptService(deptRepo repository.IDeptRepository) *DeptService {
	return &DeptService{deptRepo: deptRepo}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *DeptService) getRepo() repository.IDeptRepository {
	if s.deptRepo != nil {
		return s.deptRepo
	}
	return repository.NewDeptRepository(global.DB)
}

// SetDeptRepo 设置 Repository（测试用）
func (s *DeptService) SetDeptRepo(repo repository.IDeptRepository) {
	s.deptRepo = repo
}

// ====================
// IDeptService 接口实现
// ====================

// GetList 获取部门列表
func (s *DeptService) GetList(ctx context.Context, q *query.DeptQuery) ([]vo.DeptVO, error) {
	repo := s.getRepo()

	deptList, err := repo.FindAll(ctx, q)
	if err != nil {
		return nil, err
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
	repo := s.getRepo()
	return repo.GetOptions(ctx)
}

// GetFormData 获取部门表单数据
func (s *DeptService) GetFormData(ctx context.Context, id int64) (*bo.DeptFormBO, error) {
	repo := s.getRepo()
	return repo.GetFormData(ctx, id)
}

// Create 创建部门
func (s *DeptService) Create(ctx context.Context, form *bo.DeptFormBO) error {
	repo := s.getRepo()

	// 校验部门名称是否存在
	depts, err := repo.FindAll(ctx, &query.DeptQuery{})
	if err != nil {
		return err
	}
	for _, dept := range depts {
		if dept.Name == form.Name {
			return errors.New("部门名称已存在")
		}
	}

	// 生成部门路径
	treePath, err := s.generateDeptTreePath(ctx, form.ParentID)
	if err != nil {
		return err
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

	return repo.Create(ctx, dept)
}

// Update 更新部门
func (s *DeptService) Update(ctx context.Context, id int64, form *bo.DeptFormBO) error {
	repo := s.getRepo()

	// 校验部门名称是否存在（排除当前部门）
	depts, err := repo.FindAll(ctx, &query.DeptQuery{})
	if err != nil {
		return err
	}
	for _, dept := range depts {
		if dept.Name == form.Name && dept.ID != id {
			return errors.New("部门名称已存在")
		}
	}

	// 生成部门路径
	treePath, err := s.generateDeptTreePath(ctx, form.ParentID)
	if err != nil {
		return err
	}

	// 查询原部门信息
	oldDept, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if oldDept == nil {
		return errors.New("部门不存在")
	}

	// 更新部门
	oldDept.Name = form.Name
	oldDept.ParentID = form.ParentID
	oldDept.Status = form.Status
	oldDept.Sort = form.Sort
	oldDept.TreePath = treePath
	oldDept.UpdatedAt = time.Now()

	return repo.Update(ctx, oldDept)
}

// Delete 删除部门
func (s *DeptService) Delete(ctx context.Context, id int64) error {
	repo := s.getRepo()

	// 检查是否有子部门
	hasChildren, err := repo.HasChildren(ctx, id)
	if err != nil {
		return err
	}
	if hasChildren {
		return errors.New("该部门存在子部门，不能删除")
	}

	// 检查是否有关联用户
	hasUsers, err := repo.HasUsers(ctx, id)
	if err != nil {
		return err
	}
	if hasUsers {
		return errors.New("该部门存在关联用户，不能删除")
	}

	return repo.Delete(ctx, id)
}

// generateDeptTreePath 生成部门树形路径
func (s *DeptService) generateDeptTreePath(ctx context.Context, parentID int64) (string, error) {
	repo := s.getRepo()
	const rootNodeID int64 = 0

	if parentID == rootNodeID {
		return strconv.FormatInt(parentID, 10), nil
	}

	parent, err := repo.FindByID(ctx, parentID)
	if err != nil {
		return "", err
	}
	if parent == nil {
		return "", errors.New("父部门不存在")
	}

	return parent.TreePath + "," + strconv.FormatInt(parent.ID, 10), nil
}
