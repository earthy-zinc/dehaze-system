package service

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
)

func TestDatasetGetPage_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	now := time.Now()
	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error) {
		return &vo.PageResult[vo.DatasetVO]{
			List: []vo.DatasetVO{
				{ID: 1, ParentID: 0, Type: "image", Name: "训练集", Description: "用于训练", Path: "/data/train", Status: 1, CreateTime: now, UpdateTime: now},
				{ID: 2, ParentID: 0, Type: "image", Name: "测试集", Description: "用于测试", Path: "/data/test", Status: 1, CreateTime: now, UpdateTime: now},
			},
			Total:    2,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := datasetService.GetPage(ctx, &query.DatasetQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(2), result.Total)
	assert.Len(t, result.List, 2)
	assert.Equal(t, "训练集", result.List[0].Name)
	assert.Equal(t, "image", result.List[0].Type)
	assert.Equal(t, "测试集", result.List[1].Name)
}

func TestDatasetGetPage_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error) {
		return &vo.PageResult[vo.DatasetVO]{
			List:     []vo.DatasetVO{},
			Total:    0,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := datasetService.GetPage(ctx, &query.DatasetQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(0), result.Total)
	assert.Len(t, result.List, 0)
}

func TestDatasetGetPage_WithKeywords(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	now := time.Now()
	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error) {
		assert.Equal(t, "训练", q.Keywords)
		return &vo.PageResult[vo.DatasetVO]{
			List: []vo.DatasetVO{
				{ID: 1, ParentID: 0, Type: "image", Name: "训练集", Description: "用于训练", Path: "/data/train", Status: 1, CreateTime: now, UpdateTime: now},
			},
			Total:    1,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := datasetService.GetPage(ctx, &query.DatasetQuery{Keywords: "训练"})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(1), result.Total)
	assert.Len(t, result.List, 1)
	assert.Equal(t, "训练集", result.List[0].Name)
}

func TestDatasetGetFormData_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	now := time.Now()
	id := int64(1)
	mockRepo.GetFormDataFunc = func(ctx context.Context, datasetID int64) (*bo.DatasetFormBO, error) {
		assert.Equal(t, int64(1), datasetID)
		return &bo.DatasetFormBO{
			ID:          &id,
			ParentID:    0,
			Type:        "image",
			Name:        "训练集",
			Description: "用于训练",
			Path:        "/data/train",
			Status:      1,
			CreateTime:  now.Format("2006-01-02T15:04:05"),
			UpdateTime:  now.Format("2006-01-02T15:04:05"),
			Statistics: &bo.StatisticsBO{
				ItemCount:          0,
				FileCount:          0,
				TotalSize:          0,
				ClearCount:         0,
				HazyCount:          0,
				SceneDistribution:  make(map[string]int64),
				HazeDistribution:   make(map[string]int64),
				FormatDistribution: make(map[string]int64),
			},
		}, nil
	}

	result, err := datasetService.GetFormData(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(1), *result.ID)
	assert.Equal(t, "训练集", result.Name)
	assert.Equal(t, "image", result.Type)
	assert.Equal(t, "/data/train", result.Path)
	assert.Equal(t, int8(1), result.Status)
	assert.NotNil(t, result.Statistics)
}

func TestDatasetGetFormData_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	mockRepo.GetFormDataFunc = func(ctx context.Context, datasetID int64) (*bo.DatasetFormBO, error) {
		return &bo.DatasetFormBO{}, nil
	}

	result, err := datasetService.GetFormData(ctx, 999)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Nil(t, result.ID)
	assert.Equal(t, "", result.Name)
}

func TestDatasetCreate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	var createdDataset *model.SysDataset
	mockRepo.CreateFunc = func(ctx context.Context, dataset *model.SysDataset) error {
		dataset.ID = 1
		createdDataset = dataset
		return nil
	}

	form := &bo.DatasetFormBO{
		ParentID:    0,
		Type:        "image",
		Name:        "新数据集",
		Description: "新建的数据集",
		Path:        "/data/new",
		Status:      1,
	}

	err := datasetService.Create(ctx, form)

	assert.NoError(t, err)
	assert.NotNil(t, createdDataset)
	assert.Equal(t, "新数据集", createdDataset.Name)
	assert.Equal(t, "image", createdDataset.Type)
	assert.Equal(t, "/data/new", createdDataset.Path)
	assert.Equal(t, "新建的数据集", createdDataset.Description)
	assert.Equal(t, int8(1), createdDataset.Status)
	assert.Equal(t, int8(0), createdDataset.Deleted)
}

func TestDatasetCreate_WithParent(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	var createdDataset *model.SysDataset
	mockRepo.CreateFunc = func(ctx context.Context, dataset *model.SysDataset) error {
		dataset.ID = 2
		createdDataset = dataset
		return nil
	}

	form := &bo.DatasetFormBO{
		ParentID:    1,
		Type:        "image",
		Name:        "子数据集",
		Description: "父数据集的子集",
		Path:        "/data/train/sub",
		Status:      1,
	}

	err := datasetService.Create(ctx, form)

	assert.NoError(t, err)
	assert.NotNil(t, createdDataset)
	assert.Equal(t, "子数据集", createdDataset.Name)
	assert.Equal(t, int64(1), createdDataset.ParentID)
	assert.Equal(t, "/data/train/sub", createdDataset.Path)
}

func TestDatasetUpdate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	now := time.Now()

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return &model.SysDataset{
			BaseModel:   model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
			ParentID:    0,
			Type:        "image",
			Name:        "训练集",
			Description: "用于训练",
			Path:        "/data/train",
			Status:      1,
		}, nil
	}

	var updatedDataset *model.SysDataset
	mockRepo.UpdateFunc = func(ctx context.Context, dataset *model.SysDataset) error {
		updatedDataset = dataset
		return nil
	}

	form := &bo.DatasetFormBO{
		ParentID:    0,
		Type:        "image",
		Name:        "训练集-更新",
		Description: "更新后的描述",
		Path:        "/data/train_updated",
		Status:      1,
	}

	err := datasetService.Update(ctx, 1, form)

	assert.NoError(t, err)
	assert.NotNil(t, updatedDataset)
	assert.Equal(t, "训练集-更新", updatedDataset.Name)
	assert.Equal(t, "/data/train_updated", updatedDataset.Path)
	assert.Equal(t, "更新后的描述", updatedDataset.Description)
}

func TestDatasetUpdate_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return nil, nil
	}

	form := &bo.DatasetFormBO{
		Name:   "训练集",
		Status: 1,
	}

	err := datasetService.Update(ctx, 999, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
}

func TestDatasetDelete_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	deletedIDs := []int64{}
	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		deletedIDs = ids
		return nil
	}

	err := datasetService.Delete(ctx, []int64{1, 2, 3})

	assert.NoError(t, err)
	assert.Equal(t, []int64{1, 2, 3}, deletedIDs)
}

func TestDatasetDelete_Multiple(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	deletedIDs := []int64{}
	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		deletedIDs = ids
		return nil
	}

	err := datasetService.Delete(ctx, []int64{5, 10, 15, 20})

	assert.NoError(t, err)
	assert.Len(t, deletedIDs, 4)
	assert.Equal(t, []int64{5, 10, 15, 20}, deletedIDs)
}

func TestDatasetDelete_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	err := datasetService.Delete(ctx, []int64{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "删除数据为空")
}

func TestDatasetCreate_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	mockRepo.CreateFunc = func(ctx context.Context, dataset *model.SysDataset) error {
		return errors.New("database error")
	}

	form := &bo.DatasetFormBO{
		Name:   "测试集",
		Status: 1,
	}

	err := datasetService.Create(ctx, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestDatasetUpdate_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	now := time.Now()

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDataset, error) {
		return &model.SysDataset{
			BaseModel: model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
			Name:      "测试集",
			Status:    1,
		}, nil
	}

	mockRepo.UpdateFunc = func(ctx context.Context, dataset *model.SysDataset) error {
		return errors.New("update failed")
	}

	form := &bo.DatasetFormBO{
		Name:   "测试集",
		Status: 1,
	}

	err := datasetService.Update(ctx, 1, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "update failed")
}

func TestDatasetDelete_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDatasetRepository)
	datasetService := NewDatasetService(mockRepo)

	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		return errors.New("delete failed")
	}

	err := datasetService.Delete(ctx, []int64{1})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "delete failed")
}
