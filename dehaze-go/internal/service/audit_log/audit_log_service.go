package audit_log

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	auditlogrepo "github.com/earthyzinc/dehaze-go/internal/repository/audit_log"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

type AuditLogService struct {
	repo *auditlogrepo.AuditLogRepository
}

func NewAuditLogService(repo *auditlogrepo.AuditLogRepository) *AuditLogService {
	return &AuditLogService{repo: repo}
}

func (s *AuditLogService) RecordAudit(ctx context.Context, operatorID int64, targetType string, targetID interface{}, action, module string, beforeValue, afterValue interface{}, ip, userAgent string) error {
	log := &model.AuditLog{
		OperatorID:  operatorID,
		TargetType:  targetType,
		TargetID:    targetID,
		Action:      action,
		Module:      module,
		BeforeValue: beforeValue,
		AfterValue:  afterValue,
		IP:          ip,
		UserAgent:   userAgent,
		CreateTime:  time.Now(),
	}
	if err := s.repo.Create(ctx, log); err != nil {
		logger.Error("写入审计日志失败", zap.Error(err))
		return err
	}
	return nil
}

func (s *AuditLogService) RecordAuditAsync(ctx context.Context, operatorID int64, targetType string, targetID interface{}, action, module string, beforeValue, afterValue interface{}, ip, userAgent string) {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				logger.Warn("审计日志异步记录 panic", zap.Any("recover", r))
			}
		}()
		writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		if err := s.RecordAudit(writeCtx, operatorID, targetType, targetID, action, module, beforeValue, afterValue, ip, userAgent); err != nil {
			logger.Warn("审计日志异步记录失败", zap.Error(err))
		}
	}()
}
