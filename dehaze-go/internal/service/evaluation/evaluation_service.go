package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/metrics"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// EvaluationService 去雾效果评估服务
type EvaluationService struct {
	repo      evalrepo.IEvalLogRepository
	algoRepo  algorepo.IAlgorithmRepository
	client    *algo.Client
	memberSvc memberservice.IMemberService
}

func NewEvaluationService(repo evalrepo.IEvalLogRepository, algoRepo algorepo.IAlgorithmRepository, client *algo.Client, memberSvc memberservice.IMemberService) *EvaluationService {
	return &EvaluationService{repo: repo, algoRepo: algoRepo, client: client, memberSvc: memberSvc}
}

// EvaluationResult 评估结果 VO
type EvaluationResult struct {
	LogID        int64           `json:"logId"`
	Status       model.LogStatus `json:"status"`
	Metrics      map[string]float64 `json:"metrics,omitempty"`
	Time         int             `json:"time"`
	ErrorMessage string          `json:"errorMessage,omitempty"`
}

// Evaluate 提交效果评估任务（异步）
// 流程：校验算法 → 校验权益扣减配额 → 写日志(processing) → 启动 goroutine 执行 → 立即返回
func (s *EvaluationService) Evaluate(ctx context.Context, algorithmID int64, predURL, gtURL string, userID int64) (*EvaluationResult, error) {
	algorithm, err := s.algoRepo.FindByID(ctx, algorithmID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
	}

	if s.memberSvc != nil {
		if err := s.memberSvc.CheckAndDeductQuota(ctx, userID, memberservice.QuotaTypeEvaluate); err != nil {
			return nil, err
		}
	}

	evalLog := &model.SysEvalLog{
		BaseModel:   model.BaseModel{CreateBy: userID},
		AlgorithmID: algorithmID,
		PredMD5:     utils.MD5Hex(predURL),
		PredURL:     predURL,
		GtMD5:       utils.MD5Hex(gtURL),
		GtURL:       gtURL,
		Status:      model.LogStatusProcessing,
	}
	if err := s.repo.Create(ctx, evalLog); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建评估日志失败", err)
	}

	logID := evalLog.ID
	go s.executeAsync(logID, algorithmID, predURL, gtURL, userID)

	return &EvaluationResult{
		LogID:  logID,
		Status: model.LogStatusProcessing,
	}, nil
}

// executeAsync 异步执行评估任务，更新日志状态
func (s *EvaluationService) executeAsync(logID, algorithmID int64, predURL, gtURL string, userID int64) {
	ctx := context.Background()
	startTime := time.Now()

	resp, err := s.client.Evaluate(ctx, &algo.EvaluationRequest{
		AlgorithmID: algorithmID,
		PredURL:     predURL,
		GtURL:       gtURL,
	})

	if err != nil {
		elapsed := int(time.Since(startTime).Seconds())
		logger.Error("异步效果评估失败",
			zap.Int64("algorithmID", algorithmID),
			zap.Int64("logID", logID),
			zap.Error(err))
		errMsg := err.Error()
		if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
			logger.Error("更新评估日志失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
		}
		metrics.RecordEvaluation("failure", time.Since(startTime).Seconds())
		s.refundQuota(userID)
		return
	}

	if model.LogStatus(resp.Status) == model.LogStatusProcessing {
		var pollErr error
		resp, pollErr = s.pollEvalTask(ctx, resp.LogID)
		if pollErr != nil {
			elapsed := int(time.Since(startTime).Seconds())
			errMsg := pollErr.Error()
			if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
				logger.Error("更新评估日志失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
			}
			metrics.RecordEvaluation("failure", time.Since(startTime).Seconds())
			s.refundQuota(userID)
			return
		}
	}

	elapsed := int(time.Since(startTime).Seconds())

	if model.LogStatus(resp.Status) == model.LogStatusFailed {
		errMsg := resp.ErrorMessage
		if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
			logger.Error("更新评估日志失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
		}
		metrics.RecordEvaluation("failure", time.Since(startTime).Seconds())
		s.refundQuota(userID)
		return
	}

	metricsJSON, _ := json.Marshal(resp.Metrics)
	resultStr := string(metricsJSON)
	if err := s.repo.UpdateResult(ctx, logID, model.LogStatusCompleted, resultStr, resp.Time); err != nil {
		logger.Error("更新评估日志完成状态失败", zap.Int64("logID", logID), zap.Error(err))
	}

	metrics.RecordEvaluation("success", time.Since(startTime).Seconds())

	logger.Info("异步效果评估完成",
		zap.Int64("algorithmID", algorithmID),
		zap.Int64("logID", logID))
}

// refundQuota 评估失败时回补用户配额
func (s *EvaluationService) refundQuota(userID int64) {
	if s.memberSvc != nil {
		ctx := context.Background()
		if err := s.memberSvc.RefundQuota(ctx, userID, memberservice.QuotaTypeEvaluate); err != nil {
			logger.Warn("回补评估配额失败", zap.Int64("userID", userID), zap.Error(err))
		}
	}
}

// pollEvalTask 轮询 Python 评估任务状态直到终态
func (s *EvaluationService) pollEvalTask(ctx context.Context, pythonLogID int64) (*algo.EvaluationResponse, error) {
	const interval = 2 * time.Second
	const timeout = 5 * time.Minute
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}

		result, err := s.client.GetEvalTaskStatus(ctx, pythonLogID)
		if err != nil {
			logger.Warn("轮询评估任务状态失败",
				zap.Int64("pythonLogID", pythonLogID),
				zap.Error(err))
			continue
		}
		if model.LogStatus(result.Status) == model.LogStatusCompleted || model.LogStatus(result.Status) == model.LogStatusFailed {
			return result, nil
		}
	}
	return nil, fmt.Errorf("Python 评估任务 %d 轮询超时", pythonLogID)
}

// GetTaskStatus 查询任务状态，根据 status 返回不同字段
func (s *EvaluationService) GetTaskStatus(ctx context.Context, id int64) (*EvaluationResult, error) {
	log, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "评估任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评估日志失败", err)
	}

	result := &EvaluationResult{
		LogID:  log.ID,
		Status: log.Status,
	}
	switch log.Status {
	case model.LogStatusCompleted:
		if log.Result != nil {
			_ = json.Unmarshal([]byte(*log.Result), &result.Metrics)
		}
		result.Time = log.Time
	case model.LogStatusFailed:
		if log.ErrorMessage != nil {
			result.ErrorMessage = *log.ErrorMessage
		}
		result.Time = log.Time
	}
	return result, nil
}

// GetLogByID 查询评估日志（用于列表展示）
func (s *EvaluationService) GetLogByID(ctx context.Context, id int64) (*model.SysEvalLog, error) {
	log, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "评估任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评估日志失败", err)
	}
	return log, nil
}

// GetLogPage 分页查询评估日志
func (s *EvaluationService) GetLogPage(ctx context.Context, algorithmID int64, pageNum, pageSize int) (*common.PageResult, error) {
	list, total, err := s.repo.FindPage(ctx, algorithmID, pageNum, pageSize)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评估日志列表失败", err)
	}
	return &common.PageResult{List: list, Total: total, Page: pageNum, PageSize: pageSize}, nil
}
