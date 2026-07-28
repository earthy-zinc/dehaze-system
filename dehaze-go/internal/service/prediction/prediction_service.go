package prediction

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

const (
	predCachePrefix = "pred:"
	predCacheTTL    = 24 * time.Hour
)

// PredictionService 去雾预测服务
type PredictionService struct {
	repo      predrepo.IPredLogRepository
	algoRepo  algorepo.IAlgorithmRepository
	client    *algo.Client
	cache     types.ICache
	memberSvc memberservice.IMemberService
}

func NewPredictionService(repo predrepo.IPredLogRepository, algoRepo algorepo.IAlgorithmRepository, client *algo.Client, cache types.ICache, memberSvc memberservice.IMemberService) *PredictionService {
	return &PredictionService{repo: repo, algoRepo: algoRepo, client: client, cache: cache, memberSvc: memberSvc}
}

// PredictionResult 预测结果 VO
type PredictionResult struct {
	LogID              int64           `json:"logId"`
	Status             model.LogStatus `json:"status"`
	ResultURL          string          `json:"resultUrl,omitempty"`
	ResultThumbnailURL string          `json:"resultThumbnailUrl,omitempty"`
	Time               int             `json:"time,omitempty"`
	ErrorMessage       string          `json:"errorMessage,omitempty"`
}

// Predict 提交去雾预测任务（异步）
// 流程：校验算法 → 校验权益扣减配额 → 检查缓存 → 写日志(processing) → 启动 goroutine 执行 → 立即返回
func (s *PredictionService) Predict(ctx context.Context, algorithmID int64, imageURL string, params string, userID int64) (*PredictionResult, error) {
	if _, err := s.algoRepo.FindByID(ctx, algorithmID); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}

	if s.memberSvc != nil {
		if err := s.memberSvc.CheckAndDeductQuota(ctx, userID, memberservice.QuotaTypeDehaze); err != nil {
			return nil, err
		}
	}

	imageMD5 := utils.MD5Hex(imageURL)

	if s.cache != nil {
		cacheKey := fmt.Sprintf("%s%d:%s", predCachePrefix, algorithmID, imageMD5)
		if cachedStr, err := s.cache.Get(ctx, cacheKey); err == nil && cachedStr != "" {
			var cached algo.PredictionResponse
			if json.Unmarshal([]byte(cachedStr), &cached) == nil {
				predLog := &model.SysPredLog{
					BaseModel:   model.BaseModel{CreateBy: userID},
					AlgorithmID: algorithmID,
					OriginMD5:   imageMD5,
					OriginURL:   imageURL,
					PredMD5:     utils.MD5Hex(cached.ResultURL),
					PredURL:     cached.ResultURL,
					Time:        cached.Time,
					Status:      model.LogStatusCompleted,
				}
				if err := s.repo.Create(ctx, predLog); err != nil {
					logger.Error("写入缓存命中预测日志失败", zap.Error(err))
				}
				return &PredictionResult{
					LogID:              predLog.ID,
					Status:             model.LogStatusCompleted,
					ResultURL:          cached.ResultURL,
					ResultThumbnailURL: cached.ResultThumbnailURL,
					Time:               cached.Time,
				}, nil
			}
		}
	}

	predLog := &model.SysPredLog{
		BaseModel:   model.BaseModel{CreateBy: userID},
		AlgorithmID: algorithmID,
		OriginMD5:   imageMD5,
		OriginURL:   imageURL,
		Status:      model.LogStatusProcessing,
	}
	if err := s.repo.Create(ctx, predLog); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建预测日志失败", err)
	}

	logID := predLog.ID
	go s.executeAsync(logID, algorithmID, imageURL, params, imageMD5, userID)

	return &PredictionResult{
		LogID:  logID,
		Status: model.LogStatusProcessing,
	}, nil
}

// executeAsync 异步执行预测任务，更新日志状态
func (s *PredictionService) executeAsync(logID, algorithmID int64, imageURL, params, imageMD5 string, userID int64) {
	ctx := context.Background()
	startTime := time.Now()

	resp, err := s.client.Predict(ctx, &algo.PredictionRequest{
		AlgorithmID: algorithmID,
		ImageURL:    imageURL,
		Params:      params,
	})

	if err != nil {
		elapsed := int(time.Since(startTime).Seconds())
		logger.Error("异步去雾预测失败",
			zap.Int64("algorithmID", algorithmID),
			zap.Int64("logID", logID),
			zap.Error(err))
		errMsg := err.Error()
		if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
			logger.Error("更新预测日志失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
		}
		s.refundQuota(userID)
		return
	}

	if model.LogStatus(resp.Status) == model.LogStatusProcessing {
		var pollErr error
		resp, pollErr = s.pollPredTask(ctx, resp.LogID)
		if pollErr != nil {
			elapsed := int(time.Since(startTime).Seconds())
			errMsg := pollErr.Error()
			if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
				logger.Error("更新预测日志失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
			}
			s.refundQuota(userID)
			return
		}
	}

	elapsed := int(time.Since(startTime).Seconds())

	if model.LogStatus(resp.Status) == model.LogStatusFailed {
		errMsg := resp.ErrorMessage
		if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
			logger.Error("更新预测日志失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
		}
		s.refundQuota(userID)
		return
	}

	if err := s.repo.UpdateResult(ctx, logID, model.LogStatusCompleted, resp.ResultURL, utils.MD5Hex(resp.ResultURL), resp.Time); err != nil {
		logger.Error("更新预测日志完成状态失败", zap.Int64("logID", logID), zap.Error(err))
	}

	if s.cache != nil {
		result := &algo.PredictionResponse{
			ResultURL:          resp.ResultURL,
			ResultThumbnailURL: resp.ResultThumbnailURL,
			Time:               resp.Time,
		}
		cacheKey := fmt.Sprintf("%s%d:%s", predCachePrefix, algorithmID, imageMD5)
		if data, err := json.Marshal(result); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), predCacheTTL)
		}
	}

	logger.Info("异步去雾预测完成",
		zap.Int64("algorithmID", algorithmID),
		zap.Int64("logID", logID))
}

// refundQuota 预测失败时回补用户配额
func (s *PredictionService) refundQuota(userID int64) {
	if s.memberSvc != nil {
		if err := s.memberSvc.RefundQuota(context.Background(), userID, memberservice.QuotaTypeDehaze); err != nil {
			logger.Warn("回补预测配额失败", zap.Int64("userID", userID), zap.Error(err))
		}
	}
}

// pollPredTask 轮询 Python 预测任务状态直到终态
func (s *PredictionService) pollPredTask(ctx context.Context, pythonLogID int64) (*algo.PredictionResponse, error) {
	const interval = 2 * time.Second
	const timeout = 5 * time.Minute
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}

		result, err := s.client.GetPredTaskStatus(ctx, pythonLogID)
		if err != nil {
			logger.Warn("轮询预测任务状态失败",
				zap.Int64("pythonLogID", pythonLogID),
				zap.Error(err))
			continue
		}
		if model.LogStatus(result.Status) == model.LogStatusCompleted || model.LogStatus(result.Status) == model.LogStatusFailed {
			return result, nil
		}
	}
	return nil, fmt.Errorf("Python 预测任务 %d 轮询超时", pythonLogID)
}

// GetTaskStatus 查询任务状态，根据 status 返回不同字段
func (s *PredictionService) GetTaskStatus(ctx context.Context, id int64) (*PredictionResult, error) {
	log, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "预测任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询预测日志失败", err)
	}

	result := &PredictionResult{
		LogID:  log.ID,
		Status: log.Status,
	}
	switch log.Status {
	case model.LogStatusCompleted:
		result.ResultURL = log.PredURL
		result.Time = log.Time
	case model.LogStatusFailed:
		if log.ErrorMessage != nil {
			result.ErrorMessage = *log.ErrorMessage
		}
		result.Time = log.Time
	}
	return result, nil
}

// GetLogByID 查询预测日志（用于列表展示）
func (s *PredictionService) GetLogByID(ctx context.Context, id int64) (*model.SysPredLog, error) {
	log, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "预测任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询预测日志失败", err)
	}
	return log, nil
}

// GetLogPage 分页查询预测日志
func (s *PredictionService) GetLogPage(ctx context.Context, algorithmID int64, pageNum, pageSize int) (*common.PageResult, error) {
	list, total, err := s.repo.FindPage(ctx, algorithmID, pageNum, pageSize)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询预测日志列表失败", err)
	}
	return &common.PageResult{List: list, Total: total, Page: pageNum, PageSize: pageSize}, nil
}
