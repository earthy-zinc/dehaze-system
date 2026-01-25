package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

type MockItemFileRepository struct {
	FindByIDFunc        func(ctx context.Context, id int64) (*model.SysItemFile, error)
	FindByItemIDFunc    func(ctx context.Context, itemID int64) ([]model.SysItemFile, error)
	CreateFunc          func(ctx context.Context, itemFile *model.SysItemFile) error
	UpdateFunc          func(ctx context.Context, itemFile *model.SysItemFile) error
	DeleteFunc          func(ctx context.Context, id int64) error
	DeleteByItemIDFunc  func(ctx context.Context, itemID int64) error
	UpdateThumbnailFunc func(ctx context.Context, itemFileID, thumbnailFileID int64) error
}

func (m *MockItemFileRepository) FindByID(ctx context.Context, id int64) (*model.SysItemFile, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockItemFileRepository) FindByItemID(ctx context.Context, itemID int64) ([]model.SysItemFile, error) {
	if m.FindByItemIDFunc != nil {
		return m.FindByItemIDFunc(ctx, itemID)
	}
	return nil, nil
}

func (m *MockItemFileRepository) Create(ctx context.Context, itemFile *model.SysItemFile) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, itemFile)
	}
	return nil
}

func (m *MockItemFileRepository) Update(ctx context.Context, itemFile *model.SysItemFile) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, itemFile)
	}
	return nil
}

func (m *MockItemFileRepository) Delete(ctx context.Context, id int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, id)
	}
	return nil
}

func (m *MockItemFileRepository) DeleteByItemID(ctx context.Context, itemID int64) error {
	if m.DeleteByItemIDFunc != nil {
		return m.DeleteByItemIDFunc(ctx, itemID)
	}
	return nil
}

func (m *MockItemFileRepository) UpdateThumbnail(ctx context.Context, itemFileID, thumbnailFileID int64) error {
	if m.UpdateThumbnailFunc != nil {
		return m.UpdateThumbnailFunc(ctx, itemFileID, thumbnailFileID)
	}
	return nil
}

var _ repository.IItemFileRepository = (*MockItemFileRepository)(nil)
