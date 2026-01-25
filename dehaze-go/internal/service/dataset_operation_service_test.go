package service

import (
	"testing"

	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
)

// TODO: DatasetOperationService 已完成依赖注入改造，但方法内部仍依赖 global.DB、global.REDIS 等全局变量进行事务操作
// 需要进一步改造以支持完整的 Mock Repository 测试

// createTestDatasetOperationService 创建测试用的 DatasetOperationService 实例
func createTestDatasetOperationService(
	datasetRepo *mock_repository.MockDatasetRepository,
	datasetItemRepo *mock_repository.MockDatasetItemRepository,
	itemFileRepo *mock_repository.MockItemFileRepository,
) *DatasetOperationService {
	return NewDatasetOperationService(datasetRepo, datasetItemRepo, itemFileRepo)
}

// ========== CreateDatasetItemWithImages 测试 ==========

func TestCreateDatasetItemWithImages_Success(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := CreateDatasetItemWithImagesRequest{
			DatasetID: 1,
			ItemName:  "test-item",
			ClearImage: ImageUploadInfo{
				Type: "clear",
				Name: "clear.jpg",
				Path: "/data/clear.jpg",
				URL:  "http://example.com/clear.jpg",
				Size: 1024000,
				MD5:  "abc123",
			},
			HazyImages: []service.ImageUploadInfo{
				{
					Type:      "hazy",
					Name:      "hazy1.jpg",
					Path:      "/data/hazy1.jpg",
					URL:       "http://example.com/hazy1.jpg",
					Size:      1024000,
					MD5:       "def456",
					HazeLevel: "medium",
				},
			},
			Options: CreateItemOptions{
				ValidateResolution: false,
				SkipThumbnail:      true,
			},
		}

		result, err := svc.CreateDatasetItemWithImages(ctx, req)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, int64(1), result.DatasetID)
		assert.Greater(t, result.ImageCount, 0)
	*/
}

func TestCreateDatasetItemWithImages_DatasetNotFound(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := CreateDatasetItemWithImagesRequest{
			DatasetID: 999,
			ClearImage: ImageUploadInfo{
				Path: "/data/clear.jpg",
			},
		}

		result, err := svc.CreateDatasetItemWithImages(ctx, req)

		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "数据集不存在")
	*/
}

func TestCreateDatasetItemWithImages_InvalidClearImage(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := CreateDatasetItemWithImagesRequest{
			DatasetID:  1,
			ClearImage: ImageUploadInfo{Path: ""},
		}

		result, err := svc.CreateDatasetItemWithImages(ctx, req)

		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "清晰图路径不能为空")
	*/
}

// ========== BatchCreateDatasetItemsWithImages 测试 ==========

func TestBatchCreateDatasetItems_Success(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := BatchCreateDatasetItemsWithImagesRequest{
			DatasetID: 1,
			Items: []service.BatchItemRequest{
				{
					Prefix: "batch1",
					ClearImage: ImageUploadInfo{
						Name: "clear1.jpg",
						Path: "/data/clear1.jpg",
					},
					HazyImages: []service.ImageUploadInfo{
						{Name: "hazy1.jpg", Path: "/data/hazy1.jpg"},
					},
				},
				{
					Prefix: "batch1",
					ClearImage: ImageUploadInfo{
						Name: "clear2.jpg",
						Path: "/data/clear2.jpg",
					},
					HazyImages: []service.ImageUploadInfo{
						{Name: "hazy2.jpg", Path: "/data/hazy2.jpg"},
					},
				},
			},
			Options: CreateItemOptions{
				SkipThumbnail: true,
			},
		}

		result, err := svc.BatchCreateDatasetItemsWithImages(ctx, req)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, 2, result.Total)
		assert.Equal(t, 2, result.Success)
		assert.Equal(t, 0, result.Failed)
		assert.Len(t, result.ItemIDs, 2)
	*/
}

func TestBatchCreateDatasetItems_Empty(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := BatchCreateDatasetItemsWithImagesRequest{
			DatasetID: 1,
			Items:     []service.BatchItemRequest{},
		}

		result, err := svc.BatchCreateDatasetItemsWithImages(ctx, req)

		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "批量创建列表不能为空")
	*/
}

func TestBatchCreateDatasetItems_PartialFail(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := BatchCreateDatasetItemsWithImagesRequest{
			DatasetID: 1,
			Items: []service.BatchItemRequest{
				{
					Prefix: "valid",
					ClearImage: ImageUploadInfo{
						Path: "/data/clear1.jpg",
					},
					HazyImages: []service.ImageUploadInfo{
						{Path: "/data/hazy1.jpg"},
					},
				},
				{
					Prefix:     "invalid",
					ClearImage: ImageUploadInfo{Path: ""},
				},
			},
			Options: CreateItemOptions{
				SkipThumbnail: true,
			},
		}

		result, err := svc.BatchCreateDatasetItemsWithImages(ctx, req)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, 2, result.Total)
		assert.Equal(t, 1, result.Success)
		assert.Equal(t, 1, result.Failed)
		assert.Len(t, result.Errors, 1)
	*/
}

// ========== DeleteDatasetItemCascade 测试 ==========

func TestDeleteDatasetItemCascade_Success(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		itemID := int64(1)

		err := svc.DeleteDatasetItemCascade(ctx, itemID)

		assert.NoError(t, err)
	*/
}

func TestDeleteDatasetItemCascade_NotFound(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		err := svc.DeleteDatasetItemCascade(ctx, 999)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "数据项不存在")
	*/
}

// ========== BatchDeleteDatasets 测试 ==========

func TestBatchDeleteDatasets_Success(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := bo.BatchDeleteForm{
			IDs:   []int64{1, 2},
			Force: false,
		}

		result, err := svc.BatchDeleteDatasets(ctx, req)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, 2, result.Total)
		assert.Equal(t, 2, result.Success)
		assert.Equal(t, 0, result.Failed)
	*/
}

func TestBatchDeleteDatasets_Empty(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := bo.BatchDeleteForm{
			IDs:   []int64{},
			Force: false,
		}

		result, err := svc.BatchDeleteDatasets(ctx, req)

		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "删除ID列表不能为空")
	*/
}

func TestBatchDeleteDatasets_WithChildren(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := bo.BatchDeleteForm{
			IDs:   []int64{1},
			Force: false,
		}

		result, err := svc.BatchDeleteDatasets(ctx, req)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		// 应该删除父数据集和子数据集
		assert.Greater(t, result.Success, 1)
	*/
}

func TestBatchDeleteDatasets_HasItemsNoForce(t *testing.T) {
	t.Skip("等待 DatasetOperationService 完成依赖注入改造")
	/*
		ctx := context.Background()
		mockDatasetRepo := new(mock_repository.MockDatasetRepository)
		mockDatasetItemRepo := new(mock_repository.MockDatasetItemRepository)
		mockItemFileRepo := new(mock_repository.MockItemFileRepository)
		svc := createTestDatasetOperationService(mockDatasetRepo, mockDatasetItemRepo, mockItemFileRepo)

		req := bo.BatchDeleteForm{
			IDs:   []int64{1},
			Force: false,
		}

		result, err := svc.BatchDeleteDatasets(ctx, req)

		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "数据项")
	*/
}
