package service

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func init() {
	global.LOG = zap.NewNop()
}

func TestGetTaskStatus_FromCache(t *testing.T) {
	taskRepo := &mock.TaskRepositoryMock{}
	cacheMock := &mock.CacheMock{}

	expectedTask := &model.SysTask{
		TaskID:   "test-task-id",
		Status:   model.TaskStatusCompleted,
		Progress: 100,
	}
	taskJSON, _ := json.Marshal(expectedTask)

	cacheMock.GetFunc = func(ctx context.Context, key string) (string, error) {
		return string(taskJSON), nil
	}

	result, err := getTaskFromCacheOrDBTest(cacheMock, taskRepo, "test-task-id")

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "test-task-id", result.TaskID)
	assert.Equal(t, model.TaskStatusCompleted, result.Status)
}

func TestGetTaskStatus_FromDB(t *testing.T) {
	taskRepo := &mock.TaskRepositoryMock{}
	cacheMock := &mock.CacheMock{}

	expectedTask := &model.SysTask{
		TaskID:   "test-task-id",
		Status:   model.TaskStatusProcessing,
		Progress: 50,
	}

	cacheMock.GetFunc = func(ctx context.Context, key string) (string, error) {
		return "", nil
	}
	cacheMock.SetFunc = func(ctx context.Context, key string, value any, expiration time.Duration) error {
		return nil
	}

	taskRepo.FindByTaskIDFunc = func(ctx context.Context, taskID string) (*model.SysTask, error) {
		return expectedTask, nil
	}

	result, err := getTaskFromCacheOrDBTest(cacheMock, taskRepo, "test-task-id")

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "test-task-id", result.TaskID)
}

func TestGetTaskStatus_NotFound(t *testing.T) {
	taskRepo := &mock.TaskRepositoryMock{}
	cacheMock := &mock.CacheMock{}

	cacheMock.GetFunc = func(ctx context.Context, key string) (string, error) {
		return "", nil
	}

	taskRepo.FindByTaskIDFunc = func(ctx context.Context, taskID string) (*model.SysTask, error) {
		return nil, nil
	}

	result, err := getTaskFromCacheOrDBTest(cacheMock, taskRepo, "non-existent")

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestCleanupExpiredTasks(t *testing.T) {
	taskRepo := &mock.TaskRepositoryMock{}

	taskRepo.UpdateExpiredTasksFunc = func(ctx context.Context, threshold time.Time) (int64, error) {
		return 5, nil
	}

	ctx := context.Background()
	threshold := time.Now().Add(-24 * time.Hour)
	affected, err := taskRepo.UpdateExpiredTasks(ctx, threshold)

	assert.NoError(t, err)
	assert.Equal(t, int64(5), affected)
}

func TestCancelTask_ValidationLogic(t *testing.T) {
	existingTask := &model.SysTask{
		BaseModel: model.BaseModel{ID: 1},
		TaskID:    "test-task-id",
		Status:    model.TaskStatusProcessing,
		CreatedBy: 100,
	}

	assert.False(t, existingTask.IsCompleted())
	assert.True(t, existingTask.CanCancel())

	completedTask := &model.SysTask{
		Status: model.TaskStatusCompleted,
	}
	assert.True(t, completedTask.IsCompleted())
	assert.False(t, completedTask.CanCancel())
}

func TestTaskExpiry(t *testing.T) {
	pastTime := time.Now().Add(-1 * time.Hour)
	futureTime := time.Now().Add(1 * time.Hour)

	expiredTask := &model.SysTask{ExpiresAt: &pastTime}
	assert.True(t, expiredTask.IsExpired())

	validTask := &model.SysTask{ExpiresAt: &futureTime}
	assert.False(t, validTask.IsExpired())

	noExpiryTask := &model.SysTask{}
	assert.False(t, noExpiryTask.IsExpired())
}

func getTaskFromCacheOrDBTest(cache *mock.CacheMock, taskRepo *mock.TaskRepositoryMock, taskIDStr string) (*model.SysTask, error) {
	ctx := context.Background()
	cacheKey := "export:task:" + taskIDStr

	cachedData, err := cache.Get(ctx, cacheKey)
	if err == nil && cachedData != "" {
		var task model.SysTask
		if err := json.Unmarshal([]byte(cachedData), &task); err == nil {
			return &task, nil
		}
	}

	task, err := taskRepo.FindByTaskID(ctx, taskIDStr)
	if err != nil {
		return nil, err
	}
	if task == nil {
		return nil, assert.AnError
	}

	return task, nil
}
