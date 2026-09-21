package kb

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"gorm.io/gorm"
)

type Repository struct {
	db *gorm.DB
}

func NewRepository(db *gorm.DB) *Repository {
	return &Repository{db: db}
}

// ==================== 知识库 ====================

// PaginateVisible 当前用户可见的知识库（公共库 + 本人私有库）
func (r *Repository) PaginateVisible(ctx context.Context, userID int64, keyword string, page, size int) ([]model.SysKnowledgeBase, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysKnowledgeBase{}).
		Where("deleted = 0").
		Where("(visibility = ? OR (visibility = ? AND create_by = ?))", "public", "private", userID)
	return r.paginateKnowledgeBases(db, keyword, page, size)
}

// PaginateAll 全部知识库（管理端 view=admin 只读监控，不过滤可见性）
func (r *Repository) PaginateAll(ctx context.Context, keyword string, page, size int) ([]model.SysKnowledgeBase, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysKnowledgeBase{}).Where("deleted = 0")
	return r.paginateKnowledgeBases(db, keyword, page, size)
}

func (r *Repository) paginateKnowledgeBases(db *gorm.DB, keyword string, page, size int) ([]model.SysKnowledgeBase, int64, error) {
	if keyword != "" {
		db = db.Where("name LIKE ? ESCAPE '\\\\'", "%"+escapeLike(keyword)+"%")
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysKnowledgeBase
	err := db.Order("create_time DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

func (r *Repository) GetKnowledgeBase(ctx context.Context, id int64) (*model.SysKnowledgeBase, error) {
	var kb model.SysKnowledgeBase
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&kb).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &kb, err
}

// GetByNameAndOwner 同 owner 下名称查重（未删除）
func (r *Repository) GetByNameAndOwner(ctx context.Context, name string, ownerID int64) (*model.SysKnowledgeBase, error) {
	var kb model.SysKnowledgeBase
	err := r.db.WithContext(ctx).
		Where("name = ? AND create_by = ? AND deleted = 0", name, ownerID).
		First(&kb).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &kb, err
}

func (r *Repository) CountPrivateByOwner(ctx context.Context, ownerID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysKnowledgeBase{}).
		Where("visibility = ? AND create_by = ? AND deleted = 0", "private", ownerID).
		Count(&count).Error
	return count, err
}

func (r *Repository) UpdateKnowledgeBase(ctx context.Context, id int64, updates map[string]interface{}) error {
	updates["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysKnowledgeBase{}).Where("id = ?", id).Updates(updates).Error
}

// ==================== 文档 ====================

func (r *Repository) PaginateDocuments(ctx context.Context, kbID int64, processingStatus string, page, size int) ([]model.SysKnowledgeDocument, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysKnowledgeDocument{}).
		Where("knowledge_base_id = ? AND deleted = 0", kbID)
	if processingStatus != "" {
		db = db.Where("processing_status = ?", processingStatus)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysKnowledgeDocument
	err := db.Order("id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

func (r *Repository) CreateDocument(ctx context.Context, doc *model.SysKnowledgeDocument) error {
	return r.db.WithContext(ctx).Create(doc).Error
}

func (r *Repository) GetDocument(ctx context.Context, id int64) (*model.SysKnowledgeDocument, error) {
	var doc model.SysKnowledgeDocument
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&doc).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &doc, err
}

// ==================== 召回测试集 ====================

func (r *Repository) CreateTestSet(ctx context.Context, testSet *model.SysKnowledgeTestSet) error {
	return r.db.WithContext(ctx).Create(testSet).Error
}

func (r *Repository) PaginateTestSets(ctx context.Context, kbID int64, page, size int) ([]model.SysKnowledgeTestSet, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysKnowledgeTestSet{}).
		Where("knowledge_base_id = ? AND deleted = 0", kbID)
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysKnowledgeTestSet
	err := db.Order("id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// ==================== 文档分块 ====================

// PaginateChunks 文档分块分页（chunk_index 正序，与 python list_chunks 一致）
func (r *Repository) PaginateChunks(ctx context.Context, documentID int64, page, size int) ([]model.SysKnowledgeChunk, int64, error) {
	var total int64
	if err := r.db.WithContext(ctx).Model(&model.SysKnowledgeChunk{}).
		Where("document_id = ?", documentID).Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysKnowledgeChunk
	err := r.db.WithContext(ctx).Where("document_id = ?", documentID).
		Order("chunk_index ASC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// ==================== 低质量片段（被点踩片段） ====================

// ListLowQuality 按知识库查被点踩片段（rating=-1，thumbs_down_count 降序）
func (r *Repository) ListLowQuality(ctx context.Context, kbID int64, page, size int) ([]vo.LowQualityChunkVO, int64, error) {
	countDB := r.db.WithContext(ctx).Table("sys_knowledge_chunk_feedback AS f").
		Joins("JOIN sys_knowledge_chunk AS c ON c.id = f.chunk_id").
		Where("c.knowledge_base_id = ? AND f.rating = ?", kbID, -1)
	var total int64
	if err := countDB.Distinct("c.id").Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var rows []vo.LowQualityChunkVO
	err := r.db.WithContext(ctx).Table("sys_knowledge_chunk_feedback AS f").
		Select("c.id AS chunk_id, c.content AS content, c.document_id AS document_id, COUNT(f.id) AS thumbs_down_count").
		Joins("JOIN sys_knowledge_chunk AS c ON c.id = f.chunk_id").
		Where("c.knowledge_base_id = ? AND f.rating = ?", kbID, -1).
		Group("c.id, c.content, c.document_id").
		Order("thumbs_down_count DESC, c.id ASC").
		Offset((page - 1) * size).Limit(size).
		Scan(&rows).Error
	return rows, total, err
}

func escapeLike(s string) string {
	return strings.NewReplacer("\\", "\\\\", "%", "\\%", "_", "\\_").Replace(s)
}
