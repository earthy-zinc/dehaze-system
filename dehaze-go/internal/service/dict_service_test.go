package service

import (
	"context"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
)

func TestDictGetPage_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error) {
		return &vo.PageResult[vo.DictPageVO]{
			List: []vo.DictPageVO{
				{ID: 1, Name: "正常", Value: "1", Status: 1},
				{ID: 2, Name: "禁用", Value: "0", Status: 1},
			},
			Total:    2,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := dictService.GetPage(ctx, &query.DictPageQuery{PageNum: 1, PageSize: 10})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(2), result.Total)
	assert.Len(t, result.List, 2)
	assert.Equal(t, "正常", result.List[0].Name)
	assert.Equal(t, "1", result.List[0].Value)
}

func TestDictGetPage_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error) {
		return &vo.PageResult[vo.DictPageVO]{
			List:     []vo.DictPageVO{},
			Total:    0,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := dictService.GetPage(ctx, &query.DictPageQuery{PageNum: 1, PageSize: 10})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(0), result.Total)
	assert.Len(t, result.List, 0)
}

func TestDictGetByTypeCode_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindByTypeCodeFunc = func(ctx context.Context, typeCode string) ([]model.SysDict, error) {
		return []model.SysDict{
			{BaseModel: model.BaseModel{ID: 1}, Name: "正常", Value: "1", TypeCode: "user_status", Status: 1},
			{BaseModel: model.BaseModel{ID: 2}, Name: "禁用", Value: "0", TypeCode: "user_status", Status: 1},
		}, nil
	}

	result, err := dictService.GetByTypeCode(ctx, "user_status")

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 2)
	assert.Equal(t, "正常", result[0].Label)
	assert.Equal(t, "1", result[0].Value)
	assert.Equal(t, "禁用", result[1].Label)
	assert.Equal(t, "0", result[1].Value)
}

func TestDictGetByTypeCode_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindByTypeCodeFunc = func(ctx context.Context, typeCode string) ([]model.SysDict, error) {
		return []model.SysDict{}, nil
	}

	result, err := dictService.GetByTypeCode(ctx, "user_status")

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 0)
}

func TestDictGetFormData_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDict, error) {
		return &model.SysDict{
			BaseModel: model.BaseModel{ID: 1},
			TypeCode:  "user_status",
			Name:      "正常",
			Value:     "1",
			Status:    1,
			Sort:      1,
			Remark:    "正常状态",
		}, nil
	}

	result, err := dictService.GetFormData(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(1), *result.ID)
	assert.Equal(t, "user_status", result.TypeCode)
	assert.Equal(t, "正常", result.Name)
	assert.Equal(t, "1", result.Value)
	assert.Equal(t, int8(1), result.Status)
	assert.Equal(t, 1, result.Sort)
	assert.Equal(t, "正常状态", result.Remark)
}

func TestDictGetFormData_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDict, error) {
		return nil, nil
	}

	result, err := dictService.GetFormData(ctx, 999)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
	assert.Nil(t, result)
}

func TestDictCreate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	var createdDict *model.SysDict
	mockRepo.CreateFunc = func(ctx context.Context, dict *model.SysDict) error {
		createdDict = dict
		return nil
	}

	form := &bo.DictFormBO{
		TypeCode: "user_status",
		Name:     "正常",
		Value:    "1",
		Status:   1,
		Sort:     1,
		Remark:   "正常状态",
	}

	err := dictService.Create(ctx, form)

	assert.NoError(t, err)
	assert.NotNil(t, createdDict)
	assert.Equal(t, "user_status", createdDict.TypeCode)
	assert.Equal(t, "正常", createdDict.Name)
	assert.Equal(t, "1", createdDict.Value)
	assert.Equal(t, int8(1), createdDict.Status)
	assert.Equal(t, 1, createdDict.Sort)
}

func TestDictUpdate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDict, error) {
		return &model.SysDict{
			BaseModel: model.BaseModel{ID: 1},
			TypeCode:  "user_status",
			Name:      "正常",
			Value:     "1",
			Status:    1,
			Sort:      1,
			Remark:    "",
		}, nil
	}

	var updatedDict *model.SysDict
	mockRepo.UpdateFunc = func(ctx context.Context, dict *model.SysDict) error {
		updatedDict = dict
		return nil
	}

	form := &bo.DictFormBO{
		TypeCode: "user_status",
		Name:     "正常(修改)",
		Value:    "1",
		Status:   1,
		Sort:     2,
		Remark:   "修改备注",
	}

	err := dictService.Update(ctx, 1, form)

	assert.NoError(t, err)
	assert.NotNil(t, updatedDict)
	assert.Equal(t, "正常(修改)", updatedDict.Name)
	assert.Equal(t, 2, updatedDict.Sort)
	assert.Equal(t, "修改备注", updatedDict.Remark)
}

func TestDictUpdate_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDict, error) {
		return nil, nil
	}

	form := &bo.DictFormBO{
		TypeCode: "user_status",
		Name:     "正常",
		Value:    "1",
		Status:   1,
	}

	err := dictService.Update(ctx, 999, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
}

func TestDictDelete_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	deletedIDs := []int64{}
	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		deletedIDs = ids
		return nil
	}

	err := dictService.Delete(ctx, []int64{1, 2, 3})

	assert.NoError(t, err)
	assert.Equal(t, []int64{1, 2, 3}, deletedIDs)
}

func TestDictDelete_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	err := dictService.Delete(ctx, []int64{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "删除数据为空")
}

func TestDictDelete_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		return errors.New("database error")
	}

	err := dictService.Delete(ctx, []int64{1})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestDictGetPage_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictRepository)
	dictService := NewDictService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error) {
		return nil, errors.New("database error")
	}

	result, err := dictService.GetPage(ctx, &query.DictPageQuery{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
	assert.Nil(t, result)
}
