package repository

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"gorm.io/gorm"
)

// AlgorithmRepository 算法仓储实现
type AlgorithmRepository struct {
	db *gorm.DB
}

// NewAlgorithmRepository 创建算法仓储实例
func NewAlgorithmRepository(db *gorm.DB) *AlgorithmRepository {
	return &AlgorithmRepository{db: db}
}

// FindByID 根据 ID 查询算法
func (r *AlgorithmRepository) FindByID(ctx context.Context, id int64) (*model.SysAlgorithm, error) {
	var algorithm model.SysAlgorithm
	err := r.db.WithContext(ctx).
		Where("id = ?", id).
		First(&algorithm).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &algorithm, err
}

// FindPage 分页查询算法
func (r *AlgorithmRepository) FindPage(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error) {
	pageNum := q.PageNum
	pageSize := q.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysAlgorithm{})

	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}

	var total int64
	err := db.Count(&total).Error
	if err != nil {
		return nil, err
	}

	var algorithmList []model.SysAlgorithm
	err = db.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&algorithmList).Error
	if err != nil {
		return nil, err
	}

	// 转换为 VO
	var algorithmVOs []vo.AlgorithmVO
	for _, algorithm := range algorithmList {
		voItem := vo.AlgorithmVO{
			ID:          algorithm.ID,
			Name:        algorithm.Name,
			Type:        algorithm.Type,
			Img:         algorithm.Img,
			Description: algorithm.Description,
			Path:        algorithm.Path,
			Flops:       algorithm.Flops,
			Params:      algorithm.Params,
			ImportPath:  algorithm.ImportPath,
			Status:      int(algorithm.Status),
			Size:        algorithm.Size,
		}
		algorithmVOs = append(algorithmVOs, voItem)
	}

	return &vo.PageResult[vo.AlgorithmVO]{
		List:     algorithmVOs,
		Total:    total,
		PageNum:  pageNum,
		PageSize: pageSize,
	}, nil
}

// FindOptions 获取算法下拉选项
func (r *AlgorithmRepository) FindOptions(ctx context.Context) ([]vo.Option, error) {
	var algorithms []model.SysAlgorithm
	err := r.db.WithContext(ctx).
		Model(&model.SysAlgorithm{}).
		Where("status = ?", 1).
		Select("id, name").
		Find(&algorithms).Error
	if err != nil {
		return nil, err
	}

	options := make([]vo.Option, len(algorithms))
	for i, algorithm := range algorithms {
		options[i] = vo.Option{
			Value: algorithm.ID,
			Label: algorithm.Name,
		}
	}
	return options, nil
}

// Create 创建算法
func (r *AlgorithmRepository) Create(ctx context.Context, algorithm *model.SysAlgorithm) error {
	return r.db.WithContext(ctx).Create(algorithm).Error
}

// Update 更新算法
func (r *AlgorithmRepository) Update(ctx context.Context, algorithm *model.SysAlgorithm) error {
	return r.db.WithContext(ctx).Model(algorithm).
		Select("parent_id", "type", "name", "path", "import_path", "description", "status", "update_by").
		Updates(algorithm).Error
}

// Delete 删除算法
func (r *AlgorithmRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}
	return r.db.WithContext(ctx).Delete(&model.SysAlgorithm{}, ids).Error
}

// UpdateStatus 更新算法状态
func (r *AlgorithmRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	return r.db.WithContext(ctx).
		Model(&model.SysAlgorithm{}).
		Where("id = ?", id).
		Update("status", status).Error
}

// Ensure AlgorithmRepository implements IAlgorithmRepository
var _ IAlgorithmRepository = (*AlgorithmRepository)(nil)
