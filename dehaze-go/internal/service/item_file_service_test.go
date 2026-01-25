package service

import (
	"context"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
)

func TestItemFileFindByID_Success(t *testing.T) {
	mockItemFileRepo := new(mock_repository.MockItemFileRepository)
	itemFileService := NewItemFileService(mockItemFileRepo)

	mockItemFileRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysItemFile, error) {
		return &model.SysItemFile{
			ID:     1,
			ItemID: 100,
			FileID: 200,
			Type:   "image",
		}, nil
	}

	itemFile, err := itemFileService.GetItemFileById(1)

	assert.NoError(t, err)
	assert.Equal(t, int64(1), itemFile.ID)
	assert.Equal(t, int64(100), itemFile.ItemID)
	assert.Equal(t, int64(200), itemFile.FileID)
	assert.Equal(t, "image", itemFile.Type)
}

func TestItemFileFindByID_NotFound(t *testing.T) {
	mockItemFileRepo := new(mock_repository.MockItemFileRepository)
	itemFileService := NewItemFileService(mockItemFileRepo)

	mockItemFileRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysItemFile, error) {
		return nil, nil
	}

	_, err := itemFileService.GetItemFileById(999)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
}

func TestItemFileFindByID_Error(t *testing.T) {
	mockItemFileRepo := new(mock_repository.MockItemFileRepository)
	itemFileService := NewItemFileService(mockItemFileRepo)

	mockItemFileRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysItemFile, error) {
		return nil, errors.New("database error")
	}

	_, err := itemFileService.GetItemFileById(1)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

// TestItemFileFindByItemID_Success 跳过：GetImageUrlVOs 依赖 global.DB，尚未完全改造
func TestItemFileFindByItemID_Success(t *testing.T) {
	t.Skip("跳过：GetImageUrlVOs 依赖 global.DB，尚未完全改造为依赖注入")
}

// TestItemFileFindByItemID_Empty 跳过：GetImageUrlVOs 依赖 global.DB，尚未完全改造
func TestItemFileFindByItemID_Empty(t *testing.T) {
	t.Skip("跳过：GetImageUrlVOs 依赖 global.DB，尚未完全改造为依赖注入")
}

// TestItemFileFindByItemID_Error 跳过：GetImageUrlVOs 依赖 global.DB，尚未完全改造
func TestItemFileFindByItemID_Error(t *testing.T) {
	t.Skip("跳过：GetImageUrlVOs 依赖 global.DB，尚未完全改造为依赖注入")
}

// TestItemFileDelete_Success 跳过：DeleteItemFile 依赖 global.DB 查询数据项
func TestItemFileDelete_Success(t *testing.T) {
	t.Skip("跳过：DeleteItemFile 依赖 global.DB 查询数据项")
}

// TestItemFileDelete_NotFound 测试删除不存在的项文件
func TestItemFileDelete_NotFound(t *testing.T) {
	mockItemFileRepo := new(mock_repository.MockItemFileRepository)
	itemFileService := NewItemFileService(mockItemFileRepo)

	mockItemFileRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysItemFile, error) {
		return nil, nil
	}

	err := itemFileService.DeleteItemFile(999)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
}

// TestItemFileDeleteByItemID_Success 跳过：DeleteItemFileByItemId 依赖 global.DB
func TestItemFileDeleteByItemID_Success(t *testing.T) {
	t.Skip("跳过：DeleteItemFileByItemId 依赖 global.DB")
}

func TestItemFileDeleteByItemID_Error(t *testing.T) {
	mockItemFileRepo := new(mock_repository.MockItemFileRepository)
	itemFileService := NewItemFileService(mockItemFileRepo)

	mockItemFileRepo.FindByItemIDFunc = func(ctx context.Context, itemID int64) ([]model.SysItemFile, error) {
		return nil, errors.New("query error")
	}

	err := itemFileService.DeleteItemFileByItemId(100)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "查询项文件失败")
}

func TestItemFileUpdateThumbnail_Success(t *testing.T) {
	mockItemFileRepo := new(mock_repository.MockItemFileRepository)
	itemFileService := NewItemFileService(mockItemFileRepo)

	mockItemFileRepo.UpdateThumbnailFunc = func(ctx context.Context, itemFileID, thumbnailFileID int64) error {
		return nil
	}

	err := itemFileService.UpdateThumbnail(1, 300)

	assert.NoError(t, err)
}

func TestItemFileUpdateThumbnail_Error(t *testing.T) {
	mockItemFileRepo := new(mock_repository.MockItemFileRepository)
	itemFileService := NewItemFileService(mockItemFileRepo)

	mockItemFileRepo.UpdateThumbnailFunc = func(ctx context.Context, itemFileID, thumbnailFileID int64) error {
		return errors.New("update error")
	}

	err := itemFileService.UpdateThumbnail(1, 300)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "更新缩略图失败")
}

func TestItemFile_RepositoryInterface(t *testing.T) {
	mockItemFileRepo := new(mock_repository.MockItemFileRepository)
	itemFileService := NewItemFileService(mockItemFileRepo)

	var _ dto.ImageFileInfo = dto.ImageFileInfo{}
	var _ *ItemFileService = itemFileService

	assert.NotNil(t, itemFileService)
	assert.NotNil(t, mockItemFileRepo)
}
