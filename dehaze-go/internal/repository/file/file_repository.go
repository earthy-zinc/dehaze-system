package file

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type fileRepository struct {
	db *gorm.DB
}

// NewFileRepository 创建文件仓储实例
func NewFileRepository(db *gorm.DB) IFileRepository {
	return &fileRepository{db: db}
}

var _ IFileRepository = (*fileRepository)(nil)

func (r *fileRepository) FindByID(ctx context.Context, id int64) (*model.SysFile, error) {
	var file model.SysFile
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&file).Error
	if err != nil {
		return nil, err
	}
	return &file, nil
}

func (r *fileRepository) FindByIDs(ctx context.Context, ids []int64) ([]model.SysFile, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var files []model.SysFile
	err := r.db.WithContext(ctx).Where("id IN ?", ids).Find(&files).Error
	return files, err
}

func (r *fileRepository) FindByMD5(ctx context.Context, md5 string) (*model.SysFile, error) {
	var file model.SysFile
	err := r.db.WithContext(ctx).Where("md5 = ? AND deleted = 0", md5).First(&file).Error
	if err != nil {
		return nil, err
	}
	return &file, nil
}

func (r *fileRepository) Upsert(ctx context.Context, f *model.SysFile) error {
	return r.db.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "md5"}},
		DoUpdates: clause.AssignmentColumns([]string{"type", "name", "object_name", "storage", "size", "size_bytes", "deleted", "update_time"}),
	}).Create(f).Error
}

func (r *fileRepository) FindByObjectName(ctx context.Context, objectName string) (*model.SysFile, error) {
	var file model.SysFile
	err := r.db.WithContext(ctx).Where("object_name = ?", objectName).First(&file).Error
	if err != nil {
		return nil, err
	}
	return &file, nil
}

func (r *fileRepository) FindByPath(ctx context.Context, path string) (*model.SysFile, error) {
	var file model.SysFile
	err := r.db.WithContext(ctx).Where("path = ?", path).First(&file).Error
	if err != nil {
		return nil, err
	}
	return &file, nil
}

func (r *fileRepository) FindPage(ctx context.Context, pageNum, pageSize int, keywords string) ([]model.SysFile, int64, error) {
	var files []model.SysFile
	var total int64

	query := r.db.WithContext(ctx).Model(&model.SysFile{})
	if keywords != "" {
		like := "%" + keywords + "%"
		query = query.Where("name LIKE ? OR type LIKE ?", like, like)
	}

	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	if pageNum < 1 {
		pageNum = 1
	}
	if pageSize < 1 {
		pageSize = 10
	}
	offset := (pageNum - 1) * pageSize
	if err := query.Order("id DESC").Offset(offset).Limit(pageSize).Find(&files).Error; err != nil {
		return nil, 0, err
	}
	return files, total, nil
}

func (r *fileRepository) Create(ctx context.Context, file *model.SysFile) (*model.SysFile, error) {
	err := r.db.WithContext(ctx).Create(file).Error
	if err != nil {
		return nil, err
	}
	return file, nil
}

func (r *fileRepository) Update(ctx context.Context, file *model.SysFile) error {
	return r.db.WithContext(ctx).Unscoped().Save(file).Error
}

func (r *fileRepository) Delete(ctx context.Context, ids []int64) error {
	return r.db.WithContext(ctx).Model(&model.SysFile{}).Where("id IN ?", ids).Update("deleted", 1).Error
}
