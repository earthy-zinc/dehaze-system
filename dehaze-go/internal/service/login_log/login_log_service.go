package login_log

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	loginlogrepo "github.com/earthyzinc/dehaze-go/internal/repository/login_log"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

type LoginLogService struct {
	repo *loginlogrepo.LoginLogRepository
}

func NewLoginLogService(repo *loginlogrepo.LoginLogRepository) *LoginLogService {
	return &LoginLogService{repo: repo}
}

func (s *LoginLogService) RecordLogin(ctx context.Context, userID *int64, username, ip string, status int, message, browser, os, location string) error {
	log := &model.LoginLog{
		UserID:     userID,
		Username:   username,
		IP:         ip,
		Location:   location,
		Browser:    browser,
		OS:         os,
		Status:     status,
		Message:    message,
		CreateTime: time.Now(),
	}
	if err := s.repo.Create(ctx, log); err != nil {
		logger.Error("写入登录日志失败", zap.Error(err))
		return err
	}
	return nil
}
