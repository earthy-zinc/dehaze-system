package repository

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"

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

// FindByTypeCode 根据类型编码查询字典列表
func (r *DictRepository) FindByTypeCode(ctx context.Context, typeCode string) ([]model.SysDict, error) {
	var dictList []model.SysDict
	err := r.db.WithContext(ctx).
		Model(&model.SysDict{}).
		Where("type_code = ?", typeCode).
		Order("sort ASC").
		Find(&dictList).Error
	return dictList, err
}

// FindPage 分页查询字典
func (r *DictRepository) FindPage(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error) {
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

	var dictPageVOs []vo.DictPageVO
	for _, dict := range dictList {
		dictPageVO := vo.DictPageVO{
			ID:     dict.ID,
			Name:   dict.Name,
			Value:  dict.Value,
			Status: dict.Status,
		}
		dictPageVOs = append(dictPageVOs, dictPageVO)
	}

	return &vo.PageResult[vo.DictPageVO]{
		List:     dictPageVOs,
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
		Select("type_code", "name", "value", "status", "sort", "remark", "update_by").
		Updates(dict).Error
}

// Delete 删除字典
func (r *DictRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}
	return r.db.WithContext(ctx).Delete(&model.SysDict{}, ids).Error
}

// Ensure DictRepository implements IDictRepository
var _ IDictRepository = (*DictRepository)(nil)
