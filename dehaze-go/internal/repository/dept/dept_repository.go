package dept

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"

	"gorm.io/gorm"
)

// DeptRepository 部门仓储实现
type DeptRepository struct {
	db *gorm.DB
}

// NewDeptRepository 创建部门仓储实例
func NewDeptRepository(db *gorm.DB) *DeptRepository {
	return &DeptRepository{db: db}
}

// FindByID 根据 ID 查询部门
func (r *DeptRepository) FindByID(ctx context.Context, id int64) (*model.SysDept, error) {
	var dept model.SysDept
	err := r.db.WithContext(ctx).
		Where("id = ?", id).
		First(&dept).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &dept, err
}

// FindAll 查询所有部门
func (r *DeptRepository) FindAll(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
	var depts []model.SysDept
	db := r.db.WithContext(ctx).Model(&model.SysDept{})

	if q != nil {
		if q.Keywords != "" {
			keyword := "%" + q.Keywords + "%"
			db = db.Where("name LIKE ?", keyword)
		}
		if q.Status != nil {
			db = db.Where("status = ?", *q.Status)
		}
	}

	err := db.Order("sort ASC").Find(&depts).Error
	return depts, err
}

// FindByParentID 根据父 ID 查询子部门
func (r *DeptRepository) FindByParentID(ctx context.Context, parentID int64) ([]model.SysDept, error) {
	var depts []model.SysDept
	err := r.db.WithContext(ctx).
		Where("parent_id = ?", parentID).
		Order("sort ASC").
		Find(&depts).Error
	return depts, err
}

// FindIDByName 根据部门名称查询 ID
func (r *DeptRepository) FindIDByName(ctx context.Context, name string) (int64, error) {
	var id int64
	err := r.db.WithContext(ctx).
		Model(&model.SysDept{}).
		Where("name = ?", name).
		Pluck("id", &id).Error
	return id, err
}

// Create 创建部门
func (r *DeptRepository) Create(ctx context.Context, dept *model.SysDept) error {
	return r.db.WithContext(ctx).Create(dept).Error
}

// Update 更新部门
func (r *DeptRepository) Update(ctx context.Context, dept *model.SysDept) error {
	return r.db.WithContext(ctx).Model(dept).
		Select("name", "parent_id", "tree_path", "sort", "status", "update_by").
		Updates(dept).Error
}

// Delete 删除部门（支持批量软删除）
func (r *DeptRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysDept{}).
		Where("id IN ?", ids).
		Updates(map[string]interface{}{"deleted": 1}).Error
}

// HasChildren 检查部门是否有子部门
func (r *DeptRepository) HasChildren(ctx context.Context, id int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysDept{}).
		Where("parent_id = ?", id).
		Count(&count).Error
	return count > 0, err
}

// HasUsers 检查部门是否关联用户
func (r *DeptRepository) HasUsers(ctx context.Context, deptID int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUser{}).
		Where("dept_id = ?", deptID).
		Count(&count).Error
	return count > 0, err
}

// HasUsersInBatch 批量检查部门是否关联用户（避免 N+1 查询）
func (r *DeptRepository) HasUsersInBatch(ctx context.Context, deptIDs []int64) (map[int64]bool, error) {
	result := make(map[int64]bool, len(deptIDs))
	if len(deptIDs) == 0 {
		return result, nil
	}

	type deptCount struct {
		DeptID int64 `gorm:"column:dept_id"`
		Count  int64 `gorm:"column:cnt"`
	}
	var counts []deptCount
	err := r.db.WithContext(ctx).
		Model(&model.SysUser{}).
		Select("dept_id, COUNT(*) as cnt").
		Where("dept_id IN ?", deptIDs).
		Group("dept_id").
		Scan(&counts).Error
	if err != nil {
		return nil, err
	}

	for _, c := range counts {
		if c.Count > 0 {
			result[c.DeptID] = true
		}
	}
	return result, nil
}

// FindIDsByNames 根据部门名称批量查询 ID（返回 name→id 映射，避免 N+1 查询）
func (r *DeptRepository) FindIDsByNames(ctx context.Context, names []string) (map[string]int64, error) {
	result := make(map[string]int64, len(names))
	if len(names) == 0 {
		return result, nil
	}

	type nameID struct {
		Name string `gorm:"column:name"`
		ID   int64  `gorm:"column:id"`
	}
	var pairs []nameID
	err := r.db.WithContext(ctx).
		Model(&model.SysDept{}).
		Select("name, id").
		Where("name IN ?", names).
		Scan(&pairs).Error
	if err != nil {
		return nil, err
	}

	for _, p := range pairs {
		result[p.Name] = p.ID
	}
	return result, nil
}

// DeptOptionRead 部门选项读模型（含 parent_id 用于构建树）
type DeptOptionRead struct {
	Value    int64  `json:"value"`
	Label    string `json:"label"`
	ParentID int64  `json:"parentId"`
}

// GetOptions 获取部门下拉选项
func (r *DeptRepository) GetOptions(ctx context.Context) ([]read.Option, error) {
	var rawOptions []DeptOptionRead
	err := r.db.WithContext(ctx).
		Model(&model.SysDept{}).
		Select("id as value, name as label, parent_id as parent_id").
		Where("status = 1").
		Order("sort ASC").
		Scan(&rawOptions).Error
	if err != nil {
		return nil, err
	}

	// 构建树形结构
	return buildDeptOptionTree(0, rawOptions), nil
}

// buildDeptOptionTree 递归构建部门选项树
func buildDeptOptionTree(parentID int64, all []DeptOptionRead) []read.Option {
	var result []read.Option
	for _, item := range all {
		if item.ParentID == parentID {
			option := read.Option{
				Value:    item.Value,
				Label:    item.Label,
				Children: buildDeptOptionTree(item.Value, all),
			}
			if len(option.Children) == 0 {
				option.Children = nil
			}
			result = append(result, option)
		}
	}
	return result
}

// GetFormData 获取部门表单数据
func (r *DeptRepository) GetFormData(ctx context.Context, deptID int64) (*bo.DeptFormBO, error) {
	var form bo.DeptFormBO
	err := r.db.WithContext(ctx).
		Model(&model.SysDept{}).
		Select("id, name, parent_id, tree_path, sort, status").
		Where("id = ?", deptID).
		Scan(&form).Error
	if err != nil {
		return nil, err
	}
	if form.ID == nil {
		return nil, nil
	}
	return &form, nil
}

// GetSubDeptIDs 获取部门及所有子部门 ID
// tree_path 格式：逗号分隔的祖先 ID 路径，如 "0,1,2"
// 查询策略：匹配所有 tree_path 以 "当前部门tree_path," 开头的记录（即后代），并包含自身
func (r *DeptRepository) GetSubDeptIDs(ctx context.Context, deptID int64) ([]int64, error) {
	dept, err := r.FindByID(ctx, deptID)
	if err != nil {
		return nil, err
	}
	if dept == nil {
		return nil, nil
	}

	var ids []int64
	prefix := dept.TreePath + ","
	err = r.db.WithContext(ctx).
		Model(&model.SysDept{}).
		Select("id").
		Where("tree_path LIKE ? OR id = ?", prefix+"%", deptID).
		Scan(&ids).Error
	return ids, err
}

// Ensure DeptRepository implements IDeptRepository
var _ IDeptRepository = (*DeptRepository)(nil)
