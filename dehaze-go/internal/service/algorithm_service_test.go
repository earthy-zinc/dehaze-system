package service

import (
	"context"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
)

func TestAlgorithmGetPage_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error) {
		return &vo.PageResult[vo.AlgorithmVO]{
			List: []vo.AlgorithmVO{
				{ID: 1, Name: "YOLOv5", Type: "detection", Status: 1},
				{ID: 2, Name: "ResNet", Type: "classification", Status: 1},
			},
			Total:    2,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := algorithmService.GetPage(ctx, &query.AlgorithmQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(2), result.Total)
	assert.Len(t, result.List, 2)
	assert.Equal(t, "YOLOv5", result.List[0].Name)
	assert.Equal(t, "detection", result.List[0].Type)
}

func TestAlgorithmGetPage_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error) {
		return &vo.PageResult[vo.AlgorithmVO]{
			List:     []vo.AlgorithmVO{},
			Total:    0,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := algorithmService.GetPage(ctx, &query.AlgorithmQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(0), result.Total)
	assert.Len(t, result.List, 0)
}

func TestAlgorithmGetOptions_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	mockRepo.FindOptionsFunc = func(ctx context.Context) ([]vo.Option, error) {
		return []vo.Option{
			{Value: int64(1), Label: "YOLOv5"},
			{Value: int64(2), Label: "ResNet"},
		}, nil
	}

	result, err := algorithmService.GetOptions(ctx)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 2)
	assert.Equal(t, int64(1), result[0].Value)
	assert.Equal(t, "YOLOv5", result[0].Label)
	assert.Equal(t, int64(2), result[1].Value)
	assert.Equal(t, "ResNet", result[1].Label)
}

func TestAlgorithmGetOptions_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	mockRepo.FindOptionsFunc = func(ctx context.Context) ([]vo.Option, error) {
		return []vo.Option{}, nil
	}

	result, err := algorithmService.GetOptions(ctx)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 0)
}

func TestAlgorithmGetFormData_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	now := time.Now()
	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysAlgorithm, error) {
		return &model.SysAlgorithm{
			BaseModel:   model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
			ParentID:    0,
			Type:        "detection",
			Name:        "YOLOv5",
			Path:        "/models/yolov5.pt",
			ImportPath:  "models.yolov5",
			Description: "目标检测算法",
			Status:      1,
		}, nil
	}

	result, err := algorithmService.GetFormData(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(1), result.ID)
	assert.Equal(t, "YOLOv5", result.Name)
	assert.Equal(t, "detection", result.Type)
	assert.Equal(t, "/models/yolov5.pt", result.Path)
	assert.Equal(t, "models.yolov5", result.ImportPath)
	assert.Equal(t, "目标检测算法", result.Description)
	assert.Equal(t, 1, result.Status)
}

func TestAlgorithmGetFormData_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysAlgorithm, error) {
		return nil, nil
	}

	result, err := algorithmService.GetFormData(ctx, 999)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
	assert.Nil(t, result)
}

func TestAlgorithmCreate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	var createdAlgorithm *model.SysAlgorithm
	mockRepo.CreateFunc = func(ctx context.Context, algorithm *model.SysAlgorithm) error {
		createdAlgorithm = algorithm
		return nil
	}

	form := &bo.AlgorithmFormBO{
		ParentID:    0,
		Type:        "detection",
		Name:        "YOLOv8",
		Path:        "/models/yolov8.pt",
		ImportPath:  "models.yolov8",
		Description: "新一代目标检测算法",
		Status:      1,
	}

	err := algorithmService.Create(ctx, form)

	assert.NoError(t, err)
	assert.NotNil(t, createdAlgorithm)
	assert.Equal(t, "YOLOv8", createdAlgorithm.Name)
	assert.Equal(t, "detection", createdAlgorithm.Type)
	assert.Equal(t, "/models/yolov8.pt", createdAlgorithm.Path)
	assert.Equal(t, "models.yolov8", createdAlgorithm.ImportPath)
	assert.Equal(t, "新一代目标检测算法", createdAlgorithm.Description)
	assert.Equal(t, int8(1), createdAlgorithm.Status)
}

func TestAlgorithmUpdate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	now := time.Now()

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysAlgorithm, error) {
		return &model.SysAlgorithm{
			BaseModel:   model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
			ParentID:    0,
			Type:        "detection",
			Name:        "YOLOv5",
			Path:        "/models/yolov5.pt",
			ImportPath:  "models.yolov5",
			Description: "目标检测算法",
			Status:      1,
		}, nil
	}

	var updatedAlgorithm *model.SysAlgorithm
	mockRepo.UpdateFunc = func(ctx context.Context, algorithm *model.SysAlgorithm) error {
		updatedAlgorithm = algorithm
		return nil
	}

	form := &bo.AlgorithmFormBO{
		ParentID:    0,
		Type:        "detection",
		Name:        "YOLOv5-Updated",
		Path:        "/models/yolov5_v2.pt",
		ImportPath:  "models.yolov5_v2",
		Description: "更新后的目标检测算法",
		Status:      1,
	}

	err := algorithmService.Update(ctx, 1, form)

	assert.NoError(t, err)
	assert.NotNil(t, updatedAlgorithm)
	assert.Equal(t, "YOLOv5-Updated", updatedAlgorithm.Name)
	assert.Equal(t, "/models/yolov5_v2.pt", updatedAlgorithm.Path)
	assert.Equal(t, "models.yolov5_v2", updatedAlgorithm.ImportPath)
	assert.Equal(t, "更新后的目标检测算法", updatedAlgorithm.Description)
}

func TestAlgorithmUpdate_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysAlgorithm, error) {
		return nil, nil
	}

	form := &bo.AlgorithmFormBO{
		Name:   "YOLOv5",
		Status: 1,
	}

	err := algorithmService.Update(ctx, 999, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
}

// TestAlgorithmDelete_Success 删除成功（待Service完全改造后启用）
func TestAlgorithmDelete_Success(t *testing.T) {
	t.Skip("AlgorithmService.Delete 方法还未完全改造为依赖注入模式，待改造后启用")

	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	deletedIDs := []int64{}
	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		deletedIDs = ids
		return nil
	}

	err := algorithmService.Delete(ctx, []int64{1, 2, 3})

	assert.NoError(t, err)
	assert.Equal(t, []int64{1, 2, 3}, deletedIDs)
}

// TestAlgorithmDelete_Empty 删除数据为空（待Service完全改造后启用）
func TestAlgorithmDelete_Empty(t *testing.T) {
	t.Skip("AlgorithmService.Delete 方法还未完全改造为依赖注入模式，待改造后启用")

	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	err := algorithmService.Delete(ctx, []int64{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "删除数据为空")
}

// TestAlgorithmUpdateStatus_Success 更新状态成功（待Service完全改造后启用）
func TestAlgorithmUpdateStatus_Success(t *testing.T) {
	t.Skip("AlgorithmService.UpdateStatus 方法还未完全改造为依赖注入模式，待改造后启用")

	ctx := context.Background()
	mockRepo := new(mock_repository.MockAlgorithmRepository)
	algorithmService := NewAlgorithmService(mockRepo)

	mockRepo.UpdateStatusFunc = func(ctx context.Context, id int64, status int8) error {
		return nil
	}

	err := algorithmService.UpdateStatus(ctx, 1, 0)

	assert.NoError(t, err)
}
