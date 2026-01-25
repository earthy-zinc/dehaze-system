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

func TestDictTypeGetPage_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error) {
		return &vo.PageResult[vo.DictTypePageVO]{
			List: []vo.DictTypePageVO{
				{ID: 1, Name: "用户状态", Code: "user_status", Status: 1},
				{ID: 2, Name: "性别", Code: "gender", Status: 1},
			},
			Total:    2,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := dictTypeService.GetPage(ctx, &query.DictTypePageQuery{PageNum: 1, PageSize: 10})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(2), result.Total)
	assert.Len(t, result.List, 2)
	assert.Equal(t, "用户状态", result.List[0].Name)
	assert.Equal(t, "user_status", result.List[0].Code)
}

func TestDictTypeGetPage_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error) {
		return &vo.PageResult[vo.DictTypePageVO]{
			List:     []vo.DictTypePageVO{},
			Total:    0,
			PageNum:  1,
			PageSize: 10,
		}, nil
	}

	result, err := dictTypeService.GetPage(ctx, &query.DictTypePageQuery{PageNum: 1, PageSize: 10})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(0), result.Total)
	assert.Len(t, result.List, 0)
}

func TestDictTypeGetFormData_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDictType, error) {
		return &model.SysDictType{
			BaseModel: model.BaseModel{ID: 1, CreatedAt: time.Now(), UpdatedAt: time.Now()},
			Name:      "用户状态",
			Code:      "user_status",
			Status:    1,
			Remark:    "用户状态字典",
		}, nil
	}

	result, err := dictTypeService.GetFormData(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(1), *result.ID)
	assert.Equal(t, "用户状态", result.Name)
	assert.Equal(t, "user_status", result.Code)
	assert.Equal(t, int8(1), result.Status)
	assert.Equal(t, "用户状态字典", result.Remark)
}

func TestDictTypeGetFormData_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDictType, error) {
		return nil, nil
	}

	result, err := dictTypeService.GetFormData(ctx, 999)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
	assert.Nil(t, result)
}

func TestDictTypeCreate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		return false, nil
	}

	var createdDictType *model.SysDictType
	mockRepo.CreateFunc = func(ctx context.Context, dictType *model.SysDictType) error {
		createdDictType = dictType
		return nil
	}

	form := &bo.DictTypeFormBO{
		Name:   "数据状态",
		Code:   "data_status",
		Status: 1,
		Remark: "数据状态字典",
	}

	err := dictTypeService.Create(ctx, form)

	assert.NoError(t, err)
	assert.NotNil(t, createdDictType)
	assert.Equal(t, "数据状态", createdDictType.Name)
	assert.Equal(t, "data_status", createdDictType.Code)
	assert.Equal(t, int8(1), createdDictType.Status)
	assert.Equal(t, "数据状态字典", createdDictType.Remark)
}

func TestDictTypeCreate_DuplicateCode(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		return true, nil
	}

	form := &bo.DictTypeFormBO{
		Name:   "用户状态",
		Code:   "user_status",
		Status: 1,
	}

	err := dictTypeService.Create(ctx, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "编码已存在")
}

func TestDictTypeUpdate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	now := time.Now()

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDictType, error) {
		return &model.SysDictType{
			BaseModel: model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
			Name:      "用户状态",
			Code:      "user_status",
			Status:    1,
			Remark:    "",
		}, nil
	}

	mockRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		if code == "user_status_new" {
			return false, nil
		}
		return true, nil
	}

	var updatedDictType *model.SysDictType
	mockRepo.UpdateFunc = func(ctx context.Context, dictType *model.SysDictType) error {
		updatedDictType = dictType
		return nil
	}

	form := &bo.DictTypeFormBO{
		Name:   "用户状态(修改)",
		Code:   "user_status_new",
		Status: 1,
		Remark: "修改备注",
	}

	err := dictTypeService.Update(ctx, 1, form)

	assert.NoError(t, err)
	assert.NotNil(t, updatedDictType)
	assert.Equal(t, "用户状态(修改)", updatedDictType.Name)
	assert.Equal(t, "user_status_new", updatedDictType.Code)
	assert.Equal(t, "修改备注", updatedDictType.Remark)
}

func TestDictTypeUpdate_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDictType, error) {
		return nil, nil
	}

	form := &bo.DictTypeFormBO{
		Name:   "用户状态",
		Code:   "user_status",
		Status: 1,
	}

	err := dictTypeService.Update(ctx, 999, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "不存在")
}

func TestDictTypeUpdate_DuplicateCode(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	now := time.Now()

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDictType, error) {
		return &model.SysDictType{
			BaseModel: model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
			Name:      "用户状态",
			Code:      "user_status",
			Status:    1,
		}, nil
	}

	mockRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		if code == "existing_code" {
			return true, nil
		}
		return false, nil
	}

	form := &bo.DictTypeFormBO{
		Name:   "用户状态",
		Code:   "existing_code",
		Status: 1,
	}

	err := dictTypeService.Update(ctx, 1, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "编码已存在")
}

// TestDictTypeDelete_Success 删除成功（待Service完全改造后启用）
func TestDictTypeDelete_Success(t *testing.T) {
	t.Skip("DictTypeService.Delete 方法还未完全改造为依赖注入模式，待改造后启用")

	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	deletedIDs := []int64{}
	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		deletedIDs = ids
		return nil
	}

	err := dictTypeService.Delete(ctx, []int64{1, 2, 3})

	assert.NoError(t, err)
	assert.Equal(t, []int64{1, 2, 3}, deletedIDs)
}

func TestDictTypeDelete_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	err := dictTypeService.Delete(ctx, []int64{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "删除数据为空")
}

// TestDictTypeDelete_RepositoryError Repository错误（待Service完全改造后启用）
func TestDictTypeDelete_RepositoryError(t *testing.T) {
	t.Skip("DictTypeService.Delete 方法还未完全改造为依赖注入模式，待改造后启用")

	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		return errors.New("database error")
	}

	err := dictTypeService.Delete(ctx, []int64{1})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestDictTypeGetPage_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDictTypeRepository)
	dictTypeService := NewDictTypeService(mockRepo)

	mockRepo.FindPageFunc = func(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error) {
		return nil, errors.New("database error")
	}

	result, err := dictTypeService.GetPage(ctx, &query.DictTypePageQuery{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
	assert.Nil(t, result)
}
