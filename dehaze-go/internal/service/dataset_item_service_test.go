package service

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
)

func TestDatasetItemCreate_Success(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	var createdItem *model.SysDatasetItem
	mockItemRepo.CreateFunc = func(ctx context.Context, item *model.SysDatasetItem) error {
		item.ID = 1
		createdItem = item
		return nil
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		dataset := &model.SysDataset{}
		dataset.ID = id
		return dataset, nil
	}

	item, err := itemService.CreateDatasetItem(1)

	assert.NoError(t, err)
	assert.NotNil(t, item)
	assert.Equal(t, int64(1), item.ID)
	assert.Equal(t, int64(1), item.DatasetID)
	assert.Equal(t, "", item.Name)
	assert.NotNil(t, createdItem)
	assert.Equal(t, int64(1), createdItem.DatasetID)
}

func TestDatasetItemCreateWithName_Success(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	var createdItem *model.SysDatasetItem
	mockItemRepo.CreateFunc = func(ctx context.Context, item *model.SysDatasetItem) error {
		item.ID = 1
		createdItem = item
		return nil
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	item, err := itemService.CreateDatasetItemWithName(1, "测试数据项")

	assert.NoError(t, err)
	assert.NotNil(t, item)
	assert.Equal(t, int64(1), item.ID)
	assert.Equal(t, int64(1), item.DatasetID)
	assert.Equal(t, "测试数据项", item.Name)
	assert.NotNil(t, createdItem)
	assert.Equal(t, "测试数据项", createdItem.Name)
}

func TestDatasetItemFindByID_Success(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	now := time.Now()
	mockItemRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
		return &model.SysDatasetItem{
			ID:        1,
			DatasetID: 1,
			Name:      "测试数据项",
			CreatedAt: now,
			UpdatedAt: now,
		}, nil
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	item, err := itemService.GetDatasetItemById(1)

	assert.NoError(t, err)
	assert.Equal(t, int64(1), item.ID)
	assert.Equal(t, int64(1), item.DatasetID)
	assert.Equal(t, "测试数据项", item.Name)
}

func TestDatasetItemFindByID_NotFound(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
		return nil, nil
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	_, err := itemService.GetDatasetItemById(999)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
}

func TestDatasetItemFindByDatasetID_Success(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	now := time.Now()
	mockItemRepo.FindByDatasetIDFunc = func(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error) {
		return []model.SysDatasetItem{
			{ID: 1, DatasetID: 1, Name: "数据项1", CreatedAt: now, UpdatedAt: now},
			{ID: 2, DatasetID: 1, Name: "数据项2", CreatedAt: now, UpdatedAt: now},
		}, nil
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	items, err := itemService.GetDatasetItemsByDatasetID(1)

	assert.NoError(t, err)
	assert.Len(t, items, 2)
	assert.Equal(t, int64(1), items[0].ID)
	assert.Equal(t, "数据项1", items[0].Name)
	assert.Equal(t, int64(2), items[1].ID)
	assert.Equal(t, "数据项2", items[1].Name)
}

func TestDatasetItemFindByDatasetID_Empty(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.FindByDatasetIDFunc = func(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error) {
		return []model.SysDatasetItem{}, nil
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	items, err := itemService.GetDatasetItemsByDatasetID(999)

	assert.NoError(t, err)
	assert.NotNil(t, items)
	assert.Len(t, items, 0)
}

func TestDatasetItemUpdate_Success(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
		return &model.SysDatasetItem{ID: id, DatasetID: 1, Name: "原始名称"}, nil
	}

	mockItemRepo.UpdateFunc = func(ctx context.Context, item *model.SysDatasetItem) error {
		item.Name = "更新后的名称"
		return nil
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	err := itemService.UpdateDatasetItem(1, "更新后的名称")

	assert.NoError(t, err)
}

func TestDatasetItemUpdate_NotFound(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
		return nil, nil
	}

	mockItemRepo.UpdateFunc = func(ctx context.Context, item *model.SysDatasetItem) error {
		return errors.New("record not found")
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	err := itemService.UpdateDatasetItem(999, "新名称")

	assert.Error(t, err)
}

func TestDatasetItemDelete_Success(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	var deletedIDs []int64
	mockItemRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
		return &model.SysDatasetItem{ID: id, DatasetID: 1, Name: "测试项"}, nil
	}
	mockItemRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		deletedIDs = ids
		return nil
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	err := itemService.DeleteDatasetItem(1)

	assert.NoError(t, err)
	assert.Equal(t, []int64{1}, deletedIDs)
}

func TestDatasetItemDelete_NotFound(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
		return &model.SysDatasetItem{ID: id, DatasetID: 1, Name: "测试项"}, nil
	}
	mockItemRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		return errors.New("record not found")
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	err := itemService.DeleteDatasetItem(999)

	assert.Error(t, err)
}

func TestDatasetItemCreate_RepositoryError(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.CreateFunc = func(ctx context.Context, item *model.SysDatasetItem) error {
		return errors.New("database error")
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	_, err := itemService.CreateDatasetItem(1)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestDatasetItemUpdate_RepositoryError(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
		return &model.SysDatasetItem{ID: id, DatasetID: 1, Name: "原始名称"}, nil
	}

	mockItemRepo.UpdateFunc = func(ctx context.Context, item *model.SysDatasetItem) error {
		return errors.New("update failed")
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	err := itemService.UpdateDatasetItem(1, "新名称")

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "update failed")
}

func TestDatasetItemDelete_RepositoryError(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
		return &model.SysDatasetItem{ID: id, DatasetID: 1, Name: "测试项"}, nil
	}
	mockItemRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		return errors.New("delete failed")
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	err := itemService.DeleteDatasetItem(1)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "delete failed")
}

// TestDatasetItemFindPage_Success 测试分页查询数据项
// 注意：GetDatasetItemsByPage 方法依赖 ItemFileService，该服务尚未改造为依赖注入模式
// 因此此测试暂时跳过，等 ItemFileService 改造完成后可以启用
func TestDatasetItemFindPage_Success(t *testing.T) {
	t.Skip("跳过：ItemFileService 尚未改造为依赖注入模式")
}

// TestDatasetItemFindPage_Empty 测试分页查询空数据集
// 注意：GetDatasetItemsByPage 方法依赖 ItemFileService，该服务尚未改造为依赖注入模式
// 因此此测试暂时跳过，等 ItemFileService 改造完成后可以启用
func TestDatasetItemFindPage_Empty(t *testing.T) {
	t.Skip("跳过：ItemFileService 尚未改造为依赖注入模式")
}

func TestDatasetItemFindByDatasetID_Error(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	mockItemRepo.FindByDatasetIDFunc = func(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error) {
		return nil, errors.New("query error")
	}

	mockDatasetRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return func() *model.SysDataset { d := &model.SysDataset{}; d.ID = id; return d }(), nil
	}

	_, err := itemService.GetDatasetItemsByDatasetID(1)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "查询数据项失败")
}

// TestDatasetItemFindPage_Error 测试分页查询错误
// 注意：GetDatasetItemsByPage 方法依赖 ItemFileService，该服务尚未改造为依赖注入模式
// 因此此测试暂时跳过，等 ItemFileService 改造完成后可以启用
func TestDatasetItemFindPage_Error(t *testing.T) {
	t.Skip("跳过：ItemFileService 尚未改造为依赖注入模式")
}

func TestDatasetItemCreate_RepositoryInterface(t *testing.T) {
	mockItemRepo := new(mock_repository.MockDatasetItemRepository)
	mockDatasetRepo := new(mock_repository.MockDatasetRepository)
	itemService := NewDatasetItemServiceWithDatasetRepo(mockItemRepo, mockDatasetRepo)

	var _ repository.IDatasetItemRepository = mockItemRepo
	var _ *DatasetItemService = itemService

	assert.NotNil(t, itemService)
	assert.NotNil(t, mockItemRepo)
}
