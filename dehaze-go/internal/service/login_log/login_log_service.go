package login_log

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	loginlogrepo "github.com/earthyzinc/dehaze-go/internal/repository/login_log"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

type LoginLogService struct {
	repo *loginlogrepo.LoginLogRepository
}

func NewLoginLogService(repo *loginlogrepo.LoginLogRepository) *LoginLogService {
	return &LoginLogService{repo: repo}
}

func (s *LoginLogService) RecordLogin(ctx context.Context, userID *int64, username, ip string, status int, message, browser, os, deviceType string) error {
	if deviceType == "" {
		deviceType = "web"
	}
	log := &model.LoginLog{
		UserID:     userID,
		Username:   username,
		IP:         ip,
		Browser:    browser,
		OS:         os,
		DeviceType: deviceType,
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

// PageLogs 分页查询登录日志
func (s *LoginLogService) PageLogs(ctx context.Context, q *query.LoginLogQuery) ([]model.LoginLog, int64, error) {
	logs, total, err := s.repo.PageLogs(ctx, q)
	if err != nil {
		return nil, 0, common.WrapBizError(common.DATABASE_ERROR, "查询登录日志失败", err)
	}
	return logs, total, nil
}
