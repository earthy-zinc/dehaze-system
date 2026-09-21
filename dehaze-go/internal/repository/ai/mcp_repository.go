package ai

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type McpRepository struct {
	db *gorm.DB
}

func NewMcpRepository(db *gorm.DB) *McpRepository {
	return &McpRepository{db: db}
}

// ==================== Server 注册表 ====================

func (r *McpRepository) PaginateServers(ctx context.Context, page, size int, keyword string, status *int) ([]model.SysAiMcpServer, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiMcpServer{}).Where("deleted = 0")
	if keyword != "" {
		pattern := "%" + escapeLike(keyword) + "%"
		db = db.Where("(name LIKE ? ESCAPE '\\\\' OR description LIKE ? ESCAPE '\\\\')", pattern, pattern)
	}
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiMcpServer
	err := db.Order("id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

func (r *McpRepository) GetServer(ctx context.Context, id int64) (*model.SysAiMcpServer, error) {
	var s model.SysAiMcpServer
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &s, err
}

// GetServerByName 按名称查询；includeDeleted=true 时含软删行（name 唯一，历史不可复用）
func (r *McpRepository) GetServerByName(ctx context.Context, name string, includeDeleted bool) (*model.SysAiMcpServer, error) {
	db := r.db.WithContext(ctx).Where("name = ?", name)
	if includeDeleted {
		db = db.Unscoped()
	} else {
		db = db.Where("deleted = 0")
	}
	var s model.SysAiMcpServer
	err := db.First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &s, err
}

func (r *McpRepository) CreateServer(ctx context.Context, s *model.SysAiMcpServer) error {
	return r.db.WithContext(ctx).Create(s).Error
}

func (r *McpRepository) UpdateServer(ctx context.Context, id int64, updates map[string]interface{}) error {
	updates["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiMcpServer{}).Where("id = ?", id).Updates(updates).Error
}

func (r *McpRepository) SoftDeleteServer(ctx context.Context, id, updateBy int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiMcpServer{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// ==================== 命名空间 ====================

func (r *McpRepository) ListNamespaces(ctx context.Context, serverID int64) ([]model.SysAiMcpNamespace, error) {
	var items []model.SysAiMcpNamespace
	err := r.db.WithContext(ctx).
		Where("server_id = ?", serverID).
		Order("id ASC").Find(&items).Error
	return items, err
}

// ReplaceNamespaces 覆盖式更新：整组删旧插新
func (r *McpRepository) ReplaceNamespaces(ctx context.Context, serverID int64, items []model.SysAiMcpNamespace) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("server_id = ?", serverID).Delete(&model.SysAiMcpNamespace{}).Error; err != nil {
			return err
		}
		if len(items) == 0 {
			return nil
		}
		return tx.Create(&items).Error
	})
}

// CountNamespaceRefs 统计引用指定命名空间的 Server 数（校验命名空间归属）
func (r *McpRepository) CountNamespaceRefs(ctx context.Context, serverID int64, namespace string) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiMcpNamespace{}).
		Where("server_id = ? AND namespace = ?", serverID, namespace).
		Count(&count).Error
	return count, err
}

// ==================== 调用审计 ====================

func (r *McpRepository) PaginateCalls(ctx context.Context, page, size int, serverID *int64, toolName string) ([]model.SysAiMcpCall, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiMcpCall{})
	if serverID != nil {
		db = db.Where("server_id = ?", *serverID)
	}
	if toolName != "" {
		db = db.Where("tool_name = ?", toolName)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiMcpCall
	err := db.Order("id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

func (r *McpRepository) CreateCall(ctx context.Context, c *model.SysAiMcpCall) error {
	return r.db.WithContext(ctx).Create(c).Error
}
