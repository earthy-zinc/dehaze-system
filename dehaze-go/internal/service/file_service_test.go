package service

import (
	"context"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	mock "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
	"gorm.io/gorm"
)

func TestSaveFile_ExistingByMD5(t *testing.T) {
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	existingFile := &model.SysFile{
		ID:         1,
		Name:       "test.jpg",
		ObjectName: "uploads/test.jpg",
		MD5:        "abc123def456",
	}

	mockRepo.FindByMD5Func = func(ctx context.Context, md5 string) (*model.SysFile, error) {
		return existingFile, nil
	}

	fileBO := bo.FileBO{
		Name:       "test.jpg",
		ObjectName: "uploads/test.jpg",
		MD5:        "abc123def456",
	}

	result, err := fileService.SaveFile(fileBO)

	assert.NoError(t, err)
	assert.Equal(t, 1, result.ID)
	assert.Equal(t, "test.jpg", result.Name)
	assert.Equal(t, "abc123def456", result.MD5)
}

func TestSaveFile_NewFile(t *testing.T) {
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	mockRepo.FindByMD5Func = func(ctx context.Context, md5 string) (*model.SysFile, error) {
		return nil, nil
	}

	mockRepo.CreateFunc = func(ctx context.Context, file *model.SysFile) (*model.SysFile, error) {
		file.ID = 2
		return file, nil
	}

	fileBO := bo.FileBO{
		Name:       "newfile.png",
		Extension:  ".png",
		ObjectName: "uploads/newfile.png",
		MD5:        "xyz789uvw012",
		Size:       1024,
		URL:        "http://example.com/uploads/newfile.png",
		Path:       "/uploads",
	}

	result, err := fileService.SaveFile(fileBO)

	assert.NoError(t, err)
	assert.Equal(t, 2, result.ID)
	assert.Equal(t, "newfile.png", result.Name)
	assert.Equal(t, "xyz789uvw012", result.MD5)
}

func TestSaveFile_CreateError(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	mockRepo.FindByMD5Func = func(ctx context.Context, md5 string) (*model.SysFile, error) {
		return nil, nil
	}

	mockRepo.CreateFunc = func(ctx context.Context, file *model.SysFile) (*model.SysFile, error) {
		return nil, errors.New("database error")
	}

	fileBO := bo.FileBO{
		Name:       "error.jpg",
		ObjectName: "uploads/error.jpg",
		MD5:        "error123",
	}

	result, err := fileService.SaveFile(fileBO)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
	assert.Equal(t, model.SysFile{}, result)
}

func TestCheckFile_Exists(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	existingFile := &model.SysFile{
		ID:  1,
		MD5: "abc123def456",
	}

	mockRepo.FindByMD5Func = func(ctx context.Context, md5 string) (*model.SysFile, error) {
		return existingFile, nil
	}

	result := fileService.CheckFile("abc123def456")

	assert.True(t, result)
}

func TestCheckFile_NotExists(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	mockRepo.FindByMD5Func = func(ctx context.Context, md5 string) (*model.SysFile, error) {
		return nil, nil
	}

	result := fileService.CheckFile("notexists123")

	assert.False(t, result)
}

func TestDeleteFile_Success(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	existingFile := &model.SysFile{
		ID:   1,
		Name: "todelete.jpg",
	}

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysFile, error) {
		return existingFile, nil
	}

	deletedIDs := []int64{}
	mockRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		deletedIDs = ids
		return nil
	}

	err := fileService.DeleteFile(1)

	assert.NoError(t, err)
	assert.Equal(t, []int64{1}, deletedIDs)
}

func TestDeleteFile_NotFound(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysFile, error) {
		return nil, gorm.ErrRecordNotFound
	}

	err := fileService.DeleteFile(999)

	assert.Error(t, err)
	assert.Equal(t, gorm.ErrRecordNotFound, err)
}

func TestGetFileById_Success(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	existingFile := &model.SysFile{
		ID:         1,
		Name:       "test.jpg",
		ObjectName: "uploads/test.jpg",
		MD5:        "abc123def456",
		Path:       "/uploads",
	}

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysFile, error) {
		return existingFile, nil
	}

	result, err := fileService.GetFileById(1)

	assert.NoError(t, err)
	assert.Equal(t, 1, result.ID)
	assert.Equal(t, "test.jpg", result.Name)
	assert.Equal(t, "abc123def456", result.MD5)
}

func TestGetFileById_NotFound(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysFile, error) {
		return nil, gorm.ErrRecordNotFound
	}

	result, err := fileService.GetFileById(999)

	assert.Error(t, err)
	assert.Equal(t, gorm.ErrRecordNotFound, err)
	assert.Equal(t, model.SysFile{}, result)
}

func TestDownloadFile_Success(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	existingFile := &model.SysFile{
		ID:         1,
		Name:       "test.jpg",
		ObjectName: "uploads/test.jpg",
		Path:       "/data/uploads/test.jpg",
	}

	mockRepo.FindByObjectNameFunc = func(ctx context.Context, objectName string) (*model.SysFile, error) {
		return existingFile, nil
	}

	result, err := fileService.DownloadFile("uploads/test.jpg")

	assert.NoError(t, err)
	assert.Equal(t, "/data/uploads/test.jpg", result)
}

func TestDownloadFile_NotFound(t *testing.T) {
	_ = context.Background()
	mockRepo := new(mock.MockFileRepository)
	fileService := NewSysFileService(mockRepo)

	mockRepo.FindByObjectNameFunc = func(ctx context.Context, objectName string) (*model.SysFile, error) {
		return nil, gorm.ErrRecordNotFound
	}

	result, err := fileService.DownloadFile("nonexistent.jpg")

	assert.Error(t, err)
	assert.Equal(t, gorm.ErrRecordNotFound, err)
	assert.Empty(t, result)
}
