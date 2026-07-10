package dict

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"

	"gorm.io/gorm"
)

// DictRepository 字典数据仓储实现
type DictRepository struct {
	db *gorm.DB
}

// NewDictRepository 创建字典数据仓储实例
func NewDictRepository(db *gorm.DB) *DictRepository {
	return &DictRepository{db: db}
}

// FindByID 根据 ID 查询字典
func (r *DictRepository) FindByID(ctx context.Context, id int64) (*model.SysDict, error) {
	var dict model.SysDict
	err := r.db.WithContext(ctx).
		Where("id = ?", id).
		First(&dict).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &dict, err
}

// FindByIDs 根据 ID 列表批量查询字典
func (r *DictRepository) FindByIDs(ctx context.Context, ids []int64) ([]model.SysDict, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var dictList []model.SysDict
	err := r.db.WithContext(ctx).
		Model(&model.SysDict{}).
		Where("id IN ?", ids).
		Find(&dictList).Error
	return dictList, err
}

// FindByTypeCode 根据类型编码查询字典列表
func (r *DictRepository) FindByTypeCode(ctx context.Context, typeCode string) ([]model.SysDict, error) {
	var dictList []model.SysDict
	err := r.db.WithContext(ctx).
		Model(&model.SysDict{}).
		Where("type_code = ?", typeCode).
		Order("sort ASC, create_time DESC").
		Find(&dictList).Error
	return dictList, err
}

// FindPage 分页查询字典
func (r *DictRepository) FindPage(ctx context.Context, q *query.DictPageQuery) (*read.PageResult[read.DictPage], error) {
	pageNum := q.PageNum
	pageSize := q.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysDict{})

	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}
	if q.TypeCode != "" {
		db = db.Where("type_code = ?", q.TypeCode)
	}

	db = db.Order("sort ASC, create_time DESC")

	var total int64
	err := db.Count(&total).Error
	if err != nil {
		return nil, err
	}

	var dictList []model.SysDict
	err = db.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&dictList).Error
	if err != nil {
		return nil, err
	}

	var dictPages []read.DictPage
	for _, dict := range dictList {
		dictPage := read.DictPage{
			ID:         dict.ID,
			Name:       dict.Name,
			Value:      dict.Value,
			TypeCode:   dict.TypeCode,
			Defaulted:  dict.Defaulted,
			Sort:       dict.Sort,
			Status:     dict.Status,
			Remark:     dict.Remark,
			CreateTime: dict.CreatedAt,
		}
		dictPages = append(dictPages, dictPage)
	}

	return &read.PageResult[read.DictPage]{
		List:     dictPages,
		Total:    total,
		PageNum:  pageNum,
		PageSize: pageSize,
	}, nil
}

// Create 创建字典
func (r *DictRepository) Create(ctx context.Context, dict *model.SysDict) error {
	return r.db.WithContext(ctx).Create(dict).Error
}

// Update 更新字典
func (r *DictRepository) Update(ctx context.Context, dict *model.SysDict) error {
	return r.db.WithContext(ctx).Model(dict).
		Select("type_code", "name", "value", "status", "sort", "defaulted", "remark", "update_by").
		Updates(dict).Error
}

// Delete 删除字典
func (r *DictRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}
	return r.db.WithContext(ctx).Delete(&model.SysDict{}, ids).Error
}

// FindByTypeCodes 根据类型编码列表查询字典列表
func (r *DictRepository) FindByTypeCodes(ctx context.Context, typeCodes []string) ([]model.SysDict, error) {
	if len(typeCodes) == 0 {
		return nil, nil
	}
	var dictList []model.SysDict
	err := r.db.WithContext(ctx).
		Model(&model.SysDict{}).
		Where("type_code IN ?", typeCodes).
		Find(&dictList).Error
	return dictList, err
}

// UpdateTypeCode 批量更新字典的类型编码
func (r *DictRepository) UpdateTypeCode(ctx context.Context, oldCode, newCode string) error {
	return r.db.WithContext(ctx).
		Model(&model.SysDict{}).
		Where("type_code = ?", oldCode).
		Update("type_code", newCode).Error
}

// DeleteByTypeCodes 根据类型编码列表删除字典
func (r *DictRepository) DeleteByTypeCodes(ctx context.Context, typeCodes []string) error {
	if len(typeCodes) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Where("type_code IN ?", typeCodes).
		Delete(&model.SysDict{}).Error
}

// CountByTypeCodes 根据类型编码列表统计字典数量
func (r *DictRepository) CountByTypeCodes(ctx context.Context, typeCodes []string) (int64, error) {
	if len(typeCodes) == 0 {
		return 0, nil
	}
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysDict{}).
		Where("type_code IN ?", typeCodes).
		Count(&count).Error
	return count, err
}

// ExistsByTypeCodeAndValue 检查同一类型下字典值是否存在
func (r *DictRepository) ExistsByTypeCodeAndValue(ctx context.Context, typeCode, value string, excludeID ...int64) (bool, error) {
	var count int64
	db := r.db.WithContext(ctx).Model(&model.SysDict{}).
		Where("type_code = ? AND value = ?", typeCode, value)

	if len(excludeID) > 0 && excludeID[0] > 0 {
		db = db.Where("id != ?", excludeID[0])
	}

	err := db.Count(&count).Error
	return count > 0, err
}

// Transaction 执行事务
func (r *DictRepository) Transaction(ctx context.Context, fn func(repo IDictRepository) error) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		return fn(&DictRepository{db: tx})
	})
}

// WithDB 返回使用指定 DB 的新实例（用于跨 Repository 事务）
func (r *DictRepository) WithDB(db *gorm.DB) IDictRepository {
	return &DictRepository{db: db}
}

// Ensure DictRepository implements IDictRepository
var _ IDictRepository = (*DictRepository)(nil)
