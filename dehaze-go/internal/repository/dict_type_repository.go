package repository

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"

	"gorm.io/gorm"
)

// DictTypeRepository 字典类型仓储实现
type DictTypeRepository struct {
	db *gorm.DB
}

// NewDictTypeRepository 创建字典类型仓储实例
func NewDictTypeRepository(db *gorm.DB) *DictTypeRepository {
	return &DictTypeRepository{db: db}
}

// FindByID 根据 ID 查询字典类型
func (r *DictTypeRepository) FindByID(ctx context.Context, id int64) (*model.SysDictType, error) {
	var dictType model.SysDictType
	err := r.db.WithContext(ctx).
		Where("id = ?", id).
		First(&dictType).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &dictType, err
}

// FindByCode 根据编码查询字典类型
func (r *DictTypeRepository) FindByCode(ctx context.Context, code string) (*model.SysDictType, error) {
	var dictType model.SysDictType
	err := r.db.WithContext(ctx).
		Where("code = ?", code).
		First(&dictType).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &dictType, err
}

// ExistsByCode 检查字典类型编码是否存在
func (r *DictTypeRepository) ExistsByCode(ctx context.Context, code string, excludeID ...int64) (bool, error) {
	var count int64
	db := r.db.WithContext(ctx).Model(&model.SysDictType{}).
		Where("code = ?", code)

	if len(excludeID) > 0 && excludeID[0] > 0 {
		db = db.Where("id != ?", excludeID[0])
	}

	err := db.Count(&count).Error
	return count > 0, err
}

// FindPage 分页查询字典类型
func (r *DictTypeRepository) FindPage(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error) {
	pageNum := q.PageNum
	pageSize := q.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysDictType{})

	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("name LIKE ? OR code LIKE ?", keyword, keyword)
	}

	var total int64
	err := db.Count(&total).Error
	if err != nil {
		return nil, err
	}

	var dictTypeList []model.SysDictType
	err = db.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&dictTypeList).Error
	if err != nil {
		return nil, err
	}

	var dictTypePageVOs []vo.DictTypePageVO
	for _, dictType := range dictTypeList {
		dictTypePageVO := vo.DictTypePageVO{
			ID:     dictType.ID,
			Name:   dictType.Name,
			Code:   dictType.Code,
			Status: dictType.Status,
		}
		dictTypePageVOs = append(dictTypePageVOs, dictTypePageVO)
	}

	return &vo.PageResult[vo.DictTypePageVO]{
		List:     dictTypePageVOs,
		Total:    total,
		PageNum:  pageNum,
		PageSize: pageSize,
	}, nil
}

// Create 创建字典类型
func (r *DictTypeRepository) Create(ctx context.Context, dictType *model.SysDictType) error {
	return r.db.WithContext(ctx).Create(dictType).Error
}

// Update 更新字典类型
func (r *DictTypeRepository) Update(ctx context.Context, dictType *model.SysDictType) error {
	return r.db.WithContext(ctx).Model(dictType).
		Select("name", "code", "status", "remark", "update_by").
		Updates(dictType).Error
}

// Delete 删除字典类型
func (r *DictTypeRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}
	return r.db.WithContext(ctx).Delete(&model.SysDictType{}, ids).Error
}

// Ensure DictTypeRepository implements IDictTypeRepository
var _ IDictTypeRepository = (*DictTypeRepository)(nil)
