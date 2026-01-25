package service

import (
	"context"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
	"golang.org/x/crypto/bcrypt"
)

func TestLogin_Success(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	hashedPassword, _ := bcrypt.GenerateFromPassword([]byte("123456"), bcrypt.DefaultCost)
	testUser := &model.SysUser{
		BaseModel: model.BaseModel{ID: 1},
		Username:  "testuser",
		Password:  string(hashedPassword),
	}

	mockUserRepo.FindByUsernameFunc = func(ctx context.Context, username string) (*model.SysUser, error) {
		return testUser, nil
	}

	mockUserRepo.FindUserAuthInfoFunc = func(ctx context.Context, username string) (*model.UserAuthInfo, error) {
		return &model.UserAuthInfo{
			UserId:   testUser.ID,
			Username: testUser.Username,
			Roles:    []string{"ADMIN"},
			Perms:    []string{"user:view"},
		}, nil
	}

	authInfo, err := userService.Login(ctx, &model.SysUser{
		Username: "testuser",
		Password: "123456",
	})

	assert.NoError(t, err)
	assert.NotNil(t, authInfo)
	assert.Equal(t, testUser.ID, authInfo.UserId)
	assert.Equal(t, "testuser", authInfo.Username)
}

func TestLogin_UserNotFound(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByUsernameFunc = func(ctx context.Context, username string) (*model.SysUser, error) {
		return nil, nil
	}

	_, err := userService.Login(ctx, &model.SysUser{
		Username: "nonexistent",
		Password: "123456",
	})

	assert.Error(t, err)
	assert.Equal(t, ErrUserNotFound, err)
}

func TestLogin_WrongPassword(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	hashedPassword, _ := bcrypt.GenerateFromPassword([]byte("123456"), bcrypt.DefaultCost)
	testUser := &model.SysUser{
		BaseModel: model.BaseModel{ID: 1},
		Username:  "testuser",
		Password:  string(hashedPassword),
	}

	mockUserRepo.FindByUsernameFunc = func(ctx context.Context, username string) (*model.SysUser, error) {
		return testUser, nil
	}

	_, err := userService.Login(ctx, &model.SysUser{
		Username: "testuser",
		Password: "wrongpassword",
	})

	assert.Error(t, err)
	assert.Equal(t, ErrInvalidPassword, err)
}

func TestLogin_NilUser(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	_, err := userService.Login(ctx, nil)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "用户信息不能为空")
}

func TestLogin_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByUsernameFunc = func(ctx context.Context, username string) (*model.SysUser, error) {
		return nil, errors.New("database error")
	}

	_, err := userService.Login(ctx, &model.SysUser{
		Username: "testuser",
		Password: "123456",
	})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestGetUserAuthInfo_Success(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindUserAuthInfoFunc = func(ctx context.Context, username string) (*model.UserAuthInfo, error) {
		return &model.UserAuthInfo{
			UserId:   1,
			Username: "testuser",
			Roles:    []string{"ADMIN"},
			Perms:    []string{"user:view", "user:edit"},
		}, nil
	}

	authInfo, err := userService.GetUserAuthInfo(ctx, "testuser")

	assert.NoError(t, err)
	assert.NotNil(t, authInfo)
	assert.Equal(t, int64(1), authInfo.UserId)
	assert.Equal(t, "testuser", authInfo.Username)
	assert.Len(t, authInfo.Roles, 1)
	assert.Len(t, authInfo.Perms, 2)
}

func TestGetUserAuthInfo_NotFound(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindUserAuthInfoFunc = func(ctx context.Context, username string) (*model.UserAuthInfo, error) {
		return nil, nil
	}

	_, err := userService.GetUserAuthInfo(ctx, "nonexistent")

	assert.Error(t, err)
	assert.Equal(t, ErrUserNotFound, err)
}

func TestGetByID_Success(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return &model.SysUser{
			BaseModel: model.BaseModel{ID: 1},
			Username:  "testuser",
			Nickname:  "Test User",
			Gender:    1,
			Status:    1,
		}, nil
	}

	userVO, err := userService.GetByID(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, userVO)
	assert.Equal(t, int64(1), userVO.ID)
	assert.Equal(t, "testuser", userVO.Username)
	assert.Equal(t, "男", userVO.GenderLabel)
}

func TestGetByID_NotFound(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return nil, nil
	}

	_, err := userService.GetByID(ctx, 999)

	assert.Error(t, err)
	assert.Equal(t, ErrUserNotFound, err)
}

func TestDelete_Success(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return &model.SysUser{
			BaseModel: model.BaseModel{ID: 1},
			Username:  "testuser",
		}, nil
	}

	mockUserRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		return nil
	}

	err := userService.Delete(ctx, []int64{1})

	assert.NoError(t, err)
}

func TestDelete_RootUserProtected(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return &model.SysUser{
			BaseModel: model.BaseModel{ID: 1},
			Username:  "root",
		}, nil
	}

	err := userService.Delete(ctx, []int64{1})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "超级管理员不能删除")
}

func TestResetPassword_Success(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return &model.SysUser{
			BaseModel: model.BaseModel{ID: 1},
			Username:  "testuser",
		}, nil
	}

	var updatedPassword string
	mockUserRepo.UpdatePasswordFunc = func(ctx context.Context, id int64, password string) error {
		updatedPassword = password
		return nil
	}

	err := userService.ResetPassword(ctx, 1)

	assert.NoError(t, err)
	assert.NotEmpty(t, updatedPassword)
}

func TestResetPassword_UserNotFound(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return nil, nil
	}

	err := userService.ResetPassword(ctx, 999)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "用户不存在")
}

func TestUpdateStatus_Success(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return &model.SysUser{
			BaseModel: model.BaseModel{ID: 1},
			Username:  "testuser",
		}, nil
	}

	mockUserRepo.UpdateStatusFunc = func(ctx context.Context, id int64, status int8) error {
		return nil
	}

	err := userService.UpdateStatus(ctx, 1, 0)

	assert.NoError(t, err)
}

func TestUpdateStatus_RootUserProtected(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return &model.SysUser{
			BaseModel: model.BaseModel{ID: 1},
			Username:  "root",
		}, nil
	}

	err := userService.UpdateStatus(ctx, 1, 0)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "超级管理员不能修改状态")
}

func TestGetCurrentUserInfo_Success(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return &model.SysUser{
			BaseModel: model.BaseModel{ID: 1},
			Username:  "testuser",
			Nickname:  "Test User",
			Avatar:    "avatar.jpg",
		}, nil
	}

	mockUserRepo.GetUserRoleIDsFunc = func(ctx context.Context, userID int64) ([]int64, error) {
		return []int64{1, 2}, nil
	}

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		if id == 1 {
			return &model.SysRole{BaseModel: model.BaseModel{ID: 1}, Code: "ADMIN"}, nil
		}
		return &model.SysRole{BaseModel: model.BaseModel{ID: 2}, Code: "USER"}, nil
	}

	userInfo, err := userService.GetCurrentUserInfo(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, userInfo)
	assert.Equal(t, int64(1), userInfo.UserId)
	assert.Equal(t, "testuser", userInfo.Username)
	assert.Len(t, userInfo.Roles, 2)
	assert.Contains(t, userInfo.Roles, "ADMIN")
	assert.Contains(t, userInfo.Roles, "USER")
}

func TestGetCurrentUserInfo_UserNotFound(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysUser, error) {
		return nil, nil
	}

	_, err := userService.GetCurrentUserInfo(ctx, 999)

	assert.Error(t, err)
	assert.Equal(t, ErrUserNotFound, err)
}

func TestGetFormData_Success(t *testing.T) {
	ctx := context.Background()
	mockUserRepo := new(mock_repository.MockUserRepository)
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	userService := NewUserService(mockUserRepo, mockRoleRepo)

	mockUserRepo.GetFormDataFunc = func(ctx context.Context, userID int64) (*bo.UserFormBO, error) {
		return &bo.UserFormBO{
			ID:       1,
			Username: "testuser",
			Nickname: "Test User",
		}, nil
	}

	formData, err := userService.GetFormData(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, formData)
	assert.Equal(t, int64(1), formData.ID)
	assert.Equal(t, "testuser", formData.Username)
}
