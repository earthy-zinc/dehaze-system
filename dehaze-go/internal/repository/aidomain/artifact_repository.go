package aidomain

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// ArtifactRepository AI 中间产物数据访问。
type ArtifactRepository struct {
	db *gorm.DB
}

func NewArtifactRepository(db *gorm.DB) *ArtifactRepository {
	return &ArtifactRepository{db: db}
}

func (r *ArtifactRepository) GetByID(ctx context.Context, id int64) (*model.SysAiArtifact, error) {
	var artifact model.SysAiArtifact
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&artifact).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &artifact, err
}

func (r *ArtifactRepository) ListByConversation(ctx context.Context, convID int64, page, size int) ([]model.SysAiArtifact, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiArtifact{}).
		Where("conversation_id = ?", convID).
		Order("id DESC")
	return countAndFind[model.SysAiArtifact](db, page, size)
}

func (r *ArtifactRepository) ListByMessage(ctx context.Context, messageID int64) ([]model.SysAiArtifact, error) {
	var items []model.SysAiArtifact
	err := r.db.WithContext(ctx).
		Where("message_id = ?", messageID).
		Order("id DESC").Find(&items).Error
	return items, err
}

func (r *ArtifactRepository) ListByRef(ctx context.Context, refType string, refID int64) ([]model.SysAiArtifact, error) {
	var items []model.SysAiArtifact
	err := r.db.WithContext(ctx).
		Where("ref_type = ? AND ref_id = ?", refType, refID).
		Order("id DESC").Find(&items).Error
	return items, err
}

// FileRow sys_file 关联文件（产物图片 URL 运行时拼接用）。
type FileRow struct {
	ID         int64  `gorm:"column:id"`
	Storage    string `gorm:"column:storage"`
	ObjectName string `gorm:"column:object_name"`
	Deleted    int64  `gorm:"column:deleted"`
}

func (r *ArtifactRepository) GetFileByID(ctx context.Context, id int64) (*FileRow, error) {
	var file FileRow
	err := r.db.WithContext(ctx).Table("sys_file").
		Select("id, storage, object_name, deleted").
		Where("id = ? AND deleted = 0", id).First(&file).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &file, err
}

// GetRefFileID 经预测/评估日志解析产物引用的文件 ID。
func (r *ArtifactRepository) GetRefFileID(ctx context.Context, table string, id int64) (*int64, error) {
	if table != "sys_pred_log" && table != "sys_eval_log" {
		return nil, nil
	}
	var row struct {
		FileID *int64 `gorm:"column:file_id"`
	}
	err := r.db.WithContext(ctx).Table(table).
		Select("pred_file_id AS file_id").
		Where("id = ?", id).First(&row).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return row.FileID, err
}
