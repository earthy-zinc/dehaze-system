package algorithm

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
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
	if err != nil {
		return nil, err
	}
	return &algorithm, nil
}

// FindPage 分页查询算法
func (r *AlgorithmRepository) FindPage(ctx context.Context, q *query.AlgorithmQuery) (*read.PageResult[read.Algorithm], error) {
	pageNum := q.PageNum
	pageSize := q.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysAlgorithm{}).Where("deleted = 0")

	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("name LIKE ? OR type LIKE ?", keyword, keyword)
	}
	if q.Type != "" {
		db = db.Where("type = ?", q.Type)
	}
	if q.Status != nil {
		db = db.Where("status = ?", *q.Status)
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

	// 转换为读模型
	var algorithmReads []read.Algorithm
	for _, algorithm := range algorithmList {
		item := read.Algorithm{
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
		algorithmReads = append(algorithmReads, item)
	}

	return &read.PageResult[read.Algorithm]{
		List:     algorithmReads,
		Total:    total,
		PageNum:  pageNum,
		PageSize: pageSize,
	}, nil
}

// FindAll 查询所有算法（用于树形列表，过滤软删行）
func (r *AlgorithmRepository) FindAll(ctx context.Context, q *query.AlgorithmQuery) ([]read.Algorithm, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAlgorithm{}).Where("deleted = 0")

	// 选择树以 nil query 调用（全量不过滤），需判空防 panic
	if q != nil && q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}

	var algorithmList []model.SysAlgorithm
	err := db.Find(&algorithmList).Error
	if err != nil {
		return nil, err
	}

	algorithmReads := make([]read.Algorithm, 0, len(algorithmList))
	for _, algorithm := range algorithmList {
		item := read.Algorithm{
			ID:          algorithm.ID,
			ParentID:    algorithm.ParentID,
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
		algorithmReads = append(algorithmReads, item)
	}
	return algorithmReads, nil
}

// FindOptions 获取算法下拉选项
func (r *AlgorithmRepository) FindOptions(ctx context.Context) ([]read.Option, error) {
	var algorithms []model.SysAlgorithm
	err := r.db.WithContext(ctx).
		Model(&model.SysAlgorithm{}).
		Where("status = ? AND deleted = 0", 4).
		Select("id, name").
		Find(&algorithms).Error
	if err != nil {
		return nil, err
	}

	options := make([]read.Option, len(algorithms))
	for i, algorithm := range algorithms {
		options[i] = read.Option{
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

// Delete 删除算法（软删 deleted=行 id，对齐 Java @TableLogic delval=id，释放唯一键位）
func (r *AlgorithmRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}
	return r.db.WithContext(ctx).
		Model(&model.SysAlgorithm{}).
		Where("id IN ? AND deleted = 0", ids).
		Update("deleted", gorm.Expr("id")).Error
}

// UpdateStatus 更新算法状态
func (r *AlgorithmRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	return r.db.WithContext(ctx).
		Model(&model.SysAlgorithm{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{"status": status}).Error
}

// FindVersionsByAlgorithmID 查询算法版本历史（按创建时间降序，过滤软删行）
func (r *AlgorithmRepository) FindVersionsByAlgorithmID(ctx context.Context, algorithmID int64) ([]model.SysAlgorithmVersion, error) {
	var versions []model.SysAlgorithmVersion
	err := r.db.WithContext(ctx).
		Where("algorithm_id = ? AND deleted = 0", algorithmID).
		Order("create_time DESC").
		Find(&versions).Error
	if err != nil {
		return nil, err
	}
	return versions, nil
}

// ExistsByVersion 检查算法版本是否存在（仅活跃行，软删行不占唯一键位可重建）
func (r *AlgorithmRepository) ExistsByVersion(ctx context.Context, algorithmID int64, version string, excludeID ...int64) (bool, error) {
	var count int64
	query := r.db.WithContext(ctx).Model(&model.SysAlgorithmVersion{}).
		Where("algorithm_id = ? AND version = ?", algorithmID, version)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

// SearchPublished 搜索已发布算法（status=4），支持关键词模糊匹配
// SearchPublished 按关键词搜索已发布算法（python list_published 同口径：仅匹配名称/描述，不分页）
func (r *AlgorithmRepository) SearchPublished(ctx context.Context, keyword string) ([]model.SysAlgorithm, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAlgorithm{}).
		Where("status = 4 AND deleted = 0")
	if keyword != "" {
		like := "%" + keyword + "%"
		db = db.Where("name LIKE ? OR description LIKE ?", like, like)
	}
	var list []model.SysAlgorithm
	if err := db.Order("id").Find(&list).Error; err != nil {
		return nil, err
	}
	return list, nil
}

// FindAllPublished 查询所有已发布算法（status=4）
func (r *AlgorithmRepository) FindAllPublished(ctx context.Context) ([]model.SysAlgorithm, error) {
	var list []model.SysAlgorithm
	err := r.db.WithContext(ctx).
		Where("status = 4 AND deleted = 0").
		Find(&list).Error
	return list, err
}

// CountPublished 统计已发布算法数量
func (r *AlgorithmRepository) CountPublished(ctx context.Context) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysAlgorithm{}).
		Where("status = 4 AND deleted = 0").
		Count(&count).Error
	return count, err
}

// FindNameByID 查询算法名称
func (r *AlgorithmRepository) FindNameByID(ctx context.Context, id int64) (string, error) {
	var name string
	err := r.db.WithContext(ctx).
		Table("sys_algorithm").
		Where("id = ? AND deleted = 0", id).
		Select("name").
		Scan(&name).Error
	return name, err
}

// ExistsByID 检查算法是否存在
func (r *AlgorithmRepository) ExistsByID(ctx context.Context, id int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysAlgorithm{}).
		Where("id = ? AND deleted = 0", id).
		Count(&count).Error
	return count > 0, err
}

// Audit 写入审核人/时间/备注并流转状态（目标状态由 service 按业务规则给定）
func (r *AlgorithmRepository) Audit(ctx context.Context, id int64, auditBy int64, status int8, remark *string) error {
	return r.db.WithContext(ctx).Model(&model.SysAlgorithm{}).Where("id = ?", id).
		Updates(map[string]interface{}{
			"audit_by":     auditBy,
			"audit_time":   time.Now(),
			"audit_remark": remark,
			"status":       status,
		}).Error
}

// FindVersionByID 按 ID 查询版本行（过滤软删行，未找到返回 nil, nil）
func (r *AlgorithmRepository) FindVersionByID(ctx context.Context, versionID int64) (*model.SysAlgorithmVersion, error) {
	var version model.SysAlgorithmVersion
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", versionID).First(&version).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &version, nil
}

// CreateVersion 新增版本行
func (r *AlgorithmRepository) CreateVersion(ctx context.Context, version *model.SysAlgorithmVersion) error {
	return r.db.WithContext(ctx).Create(version).Error
}

// UpdateVersion 更新版本行
func (r *AlgorithmRepository) UpdateVersion(ctx context.Context, version *model.SysAlgorithmVersion) error {
	return r.db.WithContext(ctx).Save(version).Error
}

// DeactivateActiveVersions 将算法下所有活跃版本置为非活跃
func (r *AlgorithmRepository) DeactivateActiveVersions(ctx context.Context, algorithmID int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAlgorithmVersion{}).
		Where("algorithm_id = ? AND is_active = 1", algorithmID).
		Update("is_active", int8(0)).Error
}

// Ensure AlgorithmRepository implements IAlgorithmRepository
var _ IAlgorithmRepository = (*AlgorithmRepository)(nil)
