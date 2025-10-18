package service

import (
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"gorm.io/gorm"
)

type DeptService struct{}

// ListDepartments 获取部门列表
func (deptService *DeptService) ListDepartments(queryParams query.DeptQuery) (deptVOs []vo.DeptVO, err error) {
	// 构建查询
	db := global.DB.Model(&model.SysDept{}).
		Where("deleted = ?", 0)

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}
	if queryParams.Status != nil {
		db = db.Where("status = ?", *queryParams.Status)
	}

	// 按排序字段升序排列
	db = db.Order("sort ASC")

	// 查询数据
	var deptList []model.SysDept
	err = db.Find(&deptList).Error
	if err != nil {
		return deptVOs, err
	}

	if len(deptList) == 0 {
		return deptVOs, nil
	}

	// 获取所有部门ID
	deptIds := make(map[int64]bool)
	for _, dept := range deptList {
		deptIds[dept.ID] = true
	}

	// 获取父节点ID
	parentIds := make(map[int64]bool)
	for _, dept := range deptList {
		parentIds[dept.ParentID] = true
	}

	// 获取根节点ID（递归的起点），即父节点ID中不包含在部门ID中的节点
	var rootIds []int64
	for parentId := range parentIds {
		if _, exists := deptIds[parentId]; !exists {
			rootIds = append(rootIds, parentId)
		}
	}

	// 递归生成部门树形列表
	for _, rootId := range rootIds {
		children := deptService.recurDeptList(rootId, deptList)
		deptVOs = append(deptVOs, children...)
	}

	return deptVOs, nil
}

// 递归生成部门树形列表
func (deptService *DeptService) recurDeptList(parentId int64, deptList []model.SysDept) []vo.DeptVO {
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
			// 递归获取子部门
			children := deptService.recurDeptList(dept.ID, deptList)
			deptVO.Children = children
			result = append(result, deptVO)
		}
	}
	return result
}

// ListDeptOptions 部门下拉选项
func (deptService *DeptService) ListDeptOptions() (options []vo.Option, err error) {
	// 查询启用状态的部门数据
	var deptList []model.SysDept
	err = global.DB.Model(&model.SysDept{}).
		Where("status = ? AND deleted = ?", 1, 0).
		Select("id, parent_id, name").
		Order("sort ASC").
		Find(&deptList).Error

	if err != nil {
		return options, err
	}

	if len(deptList) == 0 {
		return options, nil
	}

	// 获取所有部门ID
	deptIds := make(map[int64]bool)
	for _, dept := range deptList {
		deptIds[dept.ID] = true
	}

	// 获取父节点ID
	parentIds := make(map[int64]bool)
	for _, dept := range deptList {
		parentIds[dept.ParentID] = true
	}

	// 获取根节点ID
	var rootIds []int64
	for parentId := range parentIds {
		if _, exists := deptIds[parentId]; !exists {
			rootIds = append(rootIds, parentId)
		}
	}

	// 递归生成部门树形下拉选项
	for _, rootId := range rootIds {
		children := deptService.recurDeptTreeOptions(rootId, deptList)
		options = append(options, children...)
	}

	return options, nil
}

// 递归生成部门树形下拉选项
func (deptService *DeptService) recurDeptTreeOptions(parentId int64, deptList []model.SysDept) []vo.Option {
	var result []vo.Option
	for _, dept := range deptList {
		if dept.ParentID == parentId {
			option := vo.Option{
				Value: dept.ID,
				Label: dept.Name,
			}
			// 递归获取子部门选项
			children := deptService.recurDeptTreeOptions(dept.ID, deptList)
			if len(children) > 0 {
				option.Children = children
			}
			result = append(result, option)
		}
	}
	return result
}

// SaveDept 新增部门
func (deptService *DeptService) SaveDept(deptFormBO bo.DeptFormBO) (id int64, err error) {
	// 校验部门名称是否存在
	var count int64
	err = global.DB.Model(&model.SysDept{}).
		Where("name = ? AND deleted = ?", deptFormBO.Name, 0).
		Count(&count).Error

	if err != nil {
		return 0, err
	}

	if count > 0 {
		return 0, errors.New("部门名称已存在")
	}

	// 创建部门实体
	dept := model.SysDept{
		Name:     deptFormBO.Name,
		ParentID: deptFormBO.ParentID,
		Status:   deptFormBO.Status,
		Sort:     deptFormBO.Sort,
		Deleted:  0,
	}

	// 生成部门路径(tree_path)
	treePath, err := deptService.generateDeptTreePath(deptFormBO.ParentID)
	if err != nil {
		return 0, err
	}
	dept.TreePath = treePath

	// 设置创建和更新时间
	dept.CreatedAt = time.Now()
	dept.UpdatedAt = time.Now()

	// 保存部门
	err = global.DB.Create(&dept).Error
	if err != nil {
		return 0, errors.New("部门保存失败")
	}

	return dept.ID, nil
}

// UpdateDept 修改部门
func (deptService *DeptService) UpdateDept(deptId int64, deptFormBO bo.DeptFormBO) (id int64, err error) {
	// 校验部门名称是否存在（排除当前部门）
	var count int64
	err = global.DB.Model(&model.SysDept{}).
		Where("name = ? AND id != ? AND deleted = ?", deptFormBO.Name, deptId, 0).
		Count(&count).Error

	if err != nil {
		return 0, err
	}

	if count > 0 {
		return 0, errors.New("部门名称已存在")
	}

	// 生成部门路径(tree_path)
	treePath, err := deptService.generateDeptTreePath(deptFormBO.ParentID)
	if err != nil {
		return 0, err
	}

	// 更新部门信息
	updates := map[string]interface{}{
		"name":       deptFormBO.Name,
		"parent_id":  deptFormBO.ParentID,
		"status":     deptFormBO.Status,
		"sort":       deptFormBO.Sort,
		"tree_path":  treePath,
		"updated_at": time.Now(),
	}

	err = global.DB.Model(&model.SysDept{}).
		Where("id = ? AND deleted = ?", deptId, 0).
		Updates(updates).Error

	if err != nil {
		return 0, errors.New("部门更新失败")
	}

	return deptId, nil
}

// DeleteByIds 删除部门
func (deptService *DeptService) DeleteByIds(ids string) (err error) {
	// 删除部门及子部门
	if ids != "" {
		idStrings := strings.Split(ids, ",")
		for _, idStr := range idStrings {
			deptId, err := strconv.ParseInt(idStr, 10, 64)
			if err != nil {
				return errors.New("部门ID格式不正确")
			}

			// 删除部门及子部门
			err = global.DB.Where("id = ? OR CONCAT(',',tree_path,',') LIKE CONCAT('%,',?,',%')", deptId, deptId).
				Delete(&model.SysDept{}).Error

			if err != nil {
				return errors.New("部门删除失败")
			}
		}
	}
	return nil
}

// GetDeptForm 获取部门表单数据
func (deptService *DeptService) GetDeptForm(deptId int64) (deptFormBO bo.DeptFormBO, err error) {
	var dept model.SysDept
	err = global.DB.Model(&model.SysDept{}).
		Where("id = ? AND deleted = ?", deptId, 0).
		Select("id, name, parent_id, status, sort").
		First(&dept).Error

	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return deptFormBO, errors.New("部门不存在")
		}
		return deptFormBO, err
	}

	id := dept.ID
	deptFormBO = bo.DeptFormBO{
		ID:       &id,
		Name:     dept.Name,
		ParentID: dept.ParentID,
		Status:   dept.Status,
		Sort:     dept.Sort,
	}

	return deptFormBO, nil
}

// 部门路径生成
func (deptService *DeptService) generateDeptTreePath(parentId int64) (treePath string, err error) {
	const rootNodeId int64 = 0
	if parentId == rootNodeId {
		treePath = strconv.FormatInt(parentId, 10)
	} else {
		var parent model.SysDept
		err = global.DB.Where("id = ?", parentId).First(&parent).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return "", errors.New("父部门不存在")
			}
			return "", err
		}
		treePath = parent.TreePath + "," + strconv.FormatInt(parent.ID, 10)
	}
	return treePath, nil
}
