package repository

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// itemFileRepository 项文件仓储实现
type itemFileRepository struct {
	db *gorm.DB
}

// NewItemFileRepository 创建项文件仓储实例
func NewItemFileRepository(db *gorm.DB) IItemFileRepository {
	return &itemFileRepository{db: db}
}

// FindByID 根据 ID 查询项文件
func (r *itemFileRepository) FindByID(ctx context.Context, id int64) (*model.SysItemFile, error) {
	var itemFile model.SysItemFile
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&itemFile).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &itemFile, nil
}

// FindByItemID 根据数据项 ID 查询所有项文件
func (r *itemFileRepository) FindByItemID(ctx context.Context, itemID int64) ([]model.SysItemFile, error) {
	var itemFiles []model.SysItemFile
	err := r.db.WithContext(ctx).Where("item_id = ?", itemID).Find(&itemFiles).Error
	if err != nil {
		return nil, err
	}
	return itemFiles, nil
}

// Create 创建项文件
func (r *itemFileRepository) Create(ctx context.Context, itemFile *model.SysItemFile) error {
	return r.db.WithContext(ctx).Create(itemFile).Error
}

// Update 更新项文件
func (r *itemFileRepository) Update(ctx context.Context, itemFile *model.SysItemFile) error {
	return r.db.WithContext(ctx).Save(itemFile).Error
}

// Delete 删除项文件
func (r *itemFileRepository) Delete(ctx context.Context, id int64) error {
	result := r.db.WithContext(ctx).Delete(&model.SysItemFile{}, id)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}
	return nil
}

// DeleteByItemID 根据数据项 ID 删除所有项文件
func (r *itemFileRepository) DeleteByItemID(ctx context.Context, itemID int64) error {
	return r.db.WithContext(ctx).Where("item_id = ?", itemID).Delete(&model.SysItemFile{}).Error
}

// UpdateThumbnail 更新缩略图
func (r *itemFileRepository) UpdateThumbnail(ctx context.Context, itemFileID, thumbnailFileID int64) error {
	result := r.db.WithContext(ctx).Model(&model.SysItemFile{}).
		Where("id = ?", itemFileID).
		Update("thumbnail_file_id", thumbnailFileID)

	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}
	return nil
}
