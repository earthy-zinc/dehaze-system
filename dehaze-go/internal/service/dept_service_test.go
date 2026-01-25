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

func TestDeptGetList_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	now := time.Now()

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{
			{
				BaseModel: model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
				Name:      "总公司",
				ParentID:  0,
				Sort:      1,
				Status:    1,
			},
			{
				BaseModel: model.BaseModel{ID: 2, CreatedAt: now, UpdatedAt: now},
				Name:      "技术部",
				ParentID:  1,
				Sort:      1,
				Status:    1,
			},
			{
				BaseModel: model.BaseModel{ID: 3, CreatedAt: now, UpdatedAt: now},
				Name:      "市场部",
				ParentID:  1,
				Sort:      2,
				Status:    1,
			},
		}, nil
	}

	result, err := deptService.GetList(ctx, &query.DeptQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 1)
	assert.Equal(t, int64(1), result[0].ID)
	assert.Equal(t, "总公司", result[0].Name)
	assert.Len(t, result[0].Children, 2)
	assert.Equal(t, "技术部", result[0].Children[0].Name)
	assert.Equal(t, "市场部", result[0].Children[1].Name)
}

func TestDeptGetList_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{}, nil
	}

	result, err := deptService.GetList(ctx, &query.DeptQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 0)
}

func TestDeptGetList_MultipleRoots(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	now := time.Now()

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{
			{
				BaseModel: model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
				Name:      "总公司A",
				ParentID:  0,
				Sort:      1,
				Status:    1,
			},
			{
				BaseModel: model.BaseModel{ID: 2, CreatedAt: now, UpdatedAt: now},
				Name:      "技术部",
				ParentID:  1,
				Sort:      1,
				Status:    1,
			},
			{
				BaseModel: model.BaseModel{ID: 3, CreatedAt: now, UpdatedAt: now},
				Name:      "总公司B",
				ParentID:  0,
				Sort:      2,
				Status:    1,
			},
		}, nil
	}

	result, err := deptService.GetList(ctx, &query.DeptQuery{})

	assert.NoError(t, err)
	assert.Len(t, result, 2)
	assert.Equal(t, "总公司A", result[0].Name)
	assert.Equal(t, "总公司B", result[1].Name)
	assert.Len(t, result[0].Children, 1)
}

func TestDeptGetList_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return nil, errors.New("database error")
	}

	result, err := deptService.GetList(ctx, &query.DeptQuery{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
	assert.Nil(t, result)
}

func TestDeptGetFormData_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	id := int64(1)
	mockRepo.GetFormDataFunc = func(ctx context.Context, deptID int64) (*bo.DeptFormBO, error) {
		return &bo.DeptFormBO{
			ID:       &id,
			Name:     "技术部",
			ParentID: 0,
			Status:   1,
			Sort:     1,
		}, nil
	}

	result, err := deptService.GetFormData(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, int64(1), *result.ID)
	assert.Equal(t, "技术部", result.Name)
	assert.Equal(t, int64(0), result.ParentID)
	assert.Equal(t, int8(1), result.Status)
	assert.Equal(t, 1, result.Sort)
}

func TestDeptGetFormData_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.GetFormDataFunc = func(ctx context.Context, deptID int64) (*bo.DeptFormBO, error) {
		return nil, nil
	}

	result, err := deptService.GetFormData(ctx, 999)

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestDeptCreate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{
			{BaseModel: model.BaseModel{ID: 1}, Name: "总公司", ParentID: 0, TreePath: "0"},
		}, nil
	}

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDept, error) {
		if id == 1 {
			return &model.SysDept{BaseModel: model.BaseModel{ID: 1}, Name: "总公司", ParentID: 0, TreePath: "0"}, nil
		}
		return nil, nil
	}

	var createdDept *model.SysDept
	mockRepo.CreateFunc = func(ctx context.Context, dept *model.SysDept) error {
		createdDept = dept
		return nil
	}

	form := &bo.DeptFormBO{
		Name:     "技术部",
		ParentID: 1,
		Status:   1,
		Sort:     1,
	}

	err := deptService.Create(ctx, form)

	assert.NoError(t, err)
	assert.NotNil(t, createdDept)
	assert.Equal(t, "技术部", createdDept.Name)
	assert.Equal(t, int64(1), createdDept.ParentID)
	assert.Equal(t, "0,1", createdDept.TreePath)
	assert.Equal(t, int8(1), createdDept.Status)
	assert.Equal(t, 1, createdDept.Sort)
}

func TestDeptCreate_RootDept(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{}, nil
	}

	var createdDept *model.SysDept
	mockRepo.CreateFunc = func(ctx context.Context, dept *model.SysDept) error {
		createdDept = dept
		return nil
	}

	form := &bo.DeptFormBO{
		Name:     "总公司",
		ParentID: 0,
		Status:   1,
		Sort:     1,
	}

	err := deptService.Create(ctx, form)

	assert.NoError(t, err)
	assert.NotNil(t, createdDept)
	assert.Equal(t, "总公司", createdDept.Name)
	assert.Equal(t, int64(0), createdDept.ParentID)
	assert.Equal(t, "0", createdDept.TreePath)
}

func TestDeptCreate_DuplicateName(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{
			{BaseModel: model.BaseModel{ID: 1}, Name: "技术部", ParentID: 0, TreePath: "0"},
		}, nil
	}

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDept, error) {
		return &model.SysDept{BaseModel: model.BaseModel{ID: 1}, Name: "技术部", ParentID: 0, TreePath: "0"}, nil
	}

	form := &bo.DeptFormBO{
		Name:     "技术部",
		ParentID: 0,
		Status:   1,
		Sort:     1,
	}

	err := deptService.Create(ctx, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "部门名称已存在")
}

func TestDeptCreate_ParentNotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{}, nil
	}

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDept, error) {
		return nil, nil
	}

	form := &bo.DeptFormBO{
		Name:     "技术部",
		ParentID: 999,
		Status:   1,
		Sort:     1,
	}

	err := deptService.Create(ctx, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "父部门不存在")
}

func TestDeptUpdate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	now := time.Now()

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{
			{BaseModel: model.BaseModel{ID: 1}, Name: "总公司", ParentID: 0, TreePath: "0"},
		}, nil
	}

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDept, error) {
		return &model.SysDept{
			BaseModel: model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
			Name:      "技术部",
			ParentID:  0,
			TreePath:  "0",
			Sort:      1,
			Status:    1,
		}, nil
	}

	var updatedDept *model.SysDept
	mockRepo.UpdateFunc = func(ctx context.Context, dept *model.SysDept) error {
		updatedDept = dept
		return nil
	}

	form := &bo.DeptFormBO{
		Name:     "技术部(新)",
		ParentID: 1,
		Status:   1,
		Sort:     2,
	}

	err := deptService.Update(ctx, 1, form)

	assert.NoError(t, err)
	assert.NotNil(t, updatedDept)
	assert.Equal(t, "技术部(新)", updatedDept.Name)
	assert.Equal(t, int64(1), updatedDept.ParentID)
	assert.Equal(t, "0,1", updatedDept.TreePath)
	assert.Equal(t, 2, updatedDept.Sort)
}

func TestDeptUpdate_DeptNotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{}, nil
	}

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDept, error) {
		return nil, nil
	}

	form := &bo.DeptFormBO{
		Name:     "技术部",
		ParentID: 0,
		Status:   1,
		Sort:     1,
	}

	err := deptService.Update(ctx, 999, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "部门不存在")
}

func TestDeptUpdate_DuplicateName(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	now := time.Now()

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
		return []model.SysDept{
			{BaseModel: model.BaseModel{ID: 2}, Name: "市场部", ParentID: 0, TreePath: "0"},
		}, nil
	}

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysDept, error) {
		return &model.SysDept{
			BaseModel: model.BaseModel{ID: 1, CreatedAt: now, UpdatedAt: now},
			Name:      "技术部",
			ParentID:  0,
			TreePath:  "0",
			Sort:      1,
			Status:    1,
		}, nil
	}

	form := &bo.DeptFormBO{
		Name:     "市场部",
		ParentID: 0,
		Status:   1,
		Sort:     1,
	}

	err := deptService.Update(ctx, 1, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "部门名称已存在")
}

func TestDeptDelete_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.HasChildrenFunc = func(ctx context.Context, id int64) (bool, error) {
		return false, nil
	}

	mockRepo.HasUsersFunc = func(ctx context.Context, deptID int64) (bool, error) {
		return false, nil
	}

	mockRepo.DeleteFunc = func(ctx context.Context, id int64) error {
		return nil
	}

	err := deptService.Delete(ctx, 1)

	assert.NoError(t, err)
}

func TestDeptDelete_HasChildren(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.HasChildrenFunc = func(ctx context.Context, id int64) (bool, error) {
		return true, nil
	}

	err := deptService.Delete(ctx, 1)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "子部门")
}

func TestDeptDelete_HasUsers(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.HasChildrenFunc = func(ctx context.Context, id int64) (bool, error) {
		return false, nil
	}

	mockRepo.HasUsersFunc = func(ctx context.Context, deptID int64) (bool, error) {
		return true, nil
	}

	err := deptService.Delete(ctx, 1)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "用户")
}

func TestDeptDelete_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.HasChildrenFunc = func(ctx context.Context, id int64) (bool, error) {
		return false, errors.New("database error")
	}

	err := deptService.Delete(ctx, 1)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestDeptGetOptions_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.GetOptionsFunc = func(ctx context.Context) ([]vo.Option, error) {
		return []vo.Option{
			{Value: int64(1), Label: "总公司"},
			{Value: int64(2), Label: "技术部"},
		}, nil
	}

	result, err := deptService.GetOptions(ctx)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 2)
	assert.Equal(t, int64(1), result[0].Value)
	assert.Equal(t, "总公司", result[0].Label)
}

func TestDeptGetOptions_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.GetOptionsFunc = func(ctx context.Context) ([]vo.Option, error) {
		return []vo.Option{}, nil
	}

	result, err := deptService.GetOptions(ctx)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 0)
}

func TestDeptGetOptions_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockDeptRepository)
	deptService := NewDeptService(mockRepo)

	mockRepo.GetOptionsFunc = func(ctx context.Context) ([]vo.Option, error) {
		return nil, errors.New("database error")
	}

	result, err := deptService.GetOptions(ctx)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
	assert.Nil(t, result)
}
