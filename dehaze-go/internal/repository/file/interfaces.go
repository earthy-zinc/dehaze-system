package file

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
)

// ====================
// 文件管理 Repository
// ====================

// IFileRepository 文件仓储接口
type IFileRepository interface {
	// FindByID 根据 ID 查询文件
	FindByID(ctx context.Context, id int64) (*model.SysFile, error)

	// FindByIDs 根据 ID 列表查询文件
	FindByIDs(ctx context.Context, ids []int64) ([]model.SysFile, error)

	// FindByMD5 根据 MD5 查询文件
	FindByMD5(ctx context.Context, md5 string) (*model.SysFile, error)

	// FindByObjectName 根据对象名称查询文件
	FindByObjectName(ctx context.Context, objectName string) (*model.SysFile, error)

	// Create 创建文件记录
	Create(ctx context.Context, file *model.SysFile) (*model.SysFile, error)

	// Delete 删除文件记录
	Delete(ctx context.Context, ids []int64) error

	// FindByPath 根据路径查询文件
	FindByPath(ctx context.Context, path string) (*model.SysFile, error)
}

// IItemFileRepository 项文件仓储接口
type IItemFileRepository interface {
	// FindByID 根据 ID 查询项文件
	FindByID(ctx context.Context, id int64) (*model.SysItemFile, error)

	// FindByItemID 根据数据项 ID 查询所有项文件
	FindByItemID(ctx context.Context, itemID int64) ([]model.SysItemFile, error)

	// FindByItemIDs 根据数据项 ID 列表查询所有项文件
	FindByItemIDs(ctx context.Context, itemIDs []int64) ([]model.SysItemFile, error)

	// Create 创建项文件
	Create(ctx context.Context, itemFile *model.SysItemFile) error

	// Update 更新项文件
	Update(ctx context.Context, itemFile *model.SysItemFile) error

	// Delete 删除项文件
	Delete(ctx context.Context, id int64) error

	// DeleteByItemID 根据数据项 ID 删除所有项文件
	DeleteByItemID(ctx context.Context, itemID int64) error

	// DeleteByItemIDs 根据数据项 ID 列表删除所有项文件
	DeleteByItemIDs(ctx context.Context, itemIDs []int64) error

	// UpdateThumbnail 更新缩略图
	UpdateThumbnail(ctx context.Context, itemFileID, thumbnailFileID int64) error
}
