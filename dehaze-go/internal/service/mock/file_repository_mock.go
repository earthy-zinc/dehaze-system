package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
)

// MockFileRepository 文件仓储 Mock
type MockFileRepository struct {
	FindByIDFunc         func(ctx context.Context, id int64) (*model.SysFile, error)
	FindByMD5Func        func(ctx context.Context, md5 string) (*model.SysFile, error)
	FindByObjectNameFunc func(ctx context.Context, objectName string) (*model.SysFile, error)
	CreateFunc           func(ctx context.Context, file *model.SysFile) (*model.SysFile, error)
	DeleteFunc           func(ctx context.Context, ids []int64) error
	FindByPathFunc       func(ctx context.Context, path string) (*model.SysFile, error)
}

func (m *MockFileRepository) FindByID(ctx context.Context, id int64) (*model.SysFile, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockFileRepository) FindByMD5(ctx context.Context, md5 string) (*model.SysFile, error) {
	if m.FindByMD5Func != nil {
		return m.FindByMD5Func(ctx, md5)
	}
	return nil, nil
}

func (m *MockFileRepository) FindByObjectName(ctx context.Context, objectName string) (*model.SysFile, error) {
	if m.FindByObjectNameFunc != nil {
		return m.FindByObjectNameFunc(ctx, objectName)
	}
	return nil, nil
}

func (m *MockFileRepository) Create(ctx context.Context, file *model.SysFile) (*model.SysFile, error) {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, file)
	}
	return nil, nil
}

func (m *MockFileRepository) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

func (m *MockFileRepository) FindByPath(ctx context.Context, path string) (*model.SysFile, error) {
	if m.FindByPathFunc != nil {
		return m.FindByPathFunc(ctx, path)
	}
	return nil, nil
}
