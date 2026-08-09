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
	"github.com/earthyzinc/dehaze-go/pkg/lifecycle"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/metrics"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

const (
	predCachePrefix = "prediction:"
	predCacheTTL    = 24 * time.Hour
)

// PredictionService 去雾预测服务
type PredictionService struct {
	repo      predrepo.IPredLogRepository
	algoRepo  algorepo.IAlgorithmRepository
	client    *algo.Client
	cache     types.ICache
	memberSvc memberservice.IMemberService
	lifecycle *lifecycle.Manager
}

func NewPredictionService(repo predrepo.IPredLogRepository, algoRepo algorepo.IAlgorithmRepository, client *algo.Client, cache types.ICache, memberSvc memberservice.IMemberService, lm *lifecycle.Manager) *PredictionService {
	return &PredictionService{repo: repo, algoRepo: algoRepo, client: client, cache: cache, memberSvc: memberSvc, lifecycle: lm}
}

// PredictionResult 预测结果 VO
type PredictionResult struct {
	LogID              int64           `json:"logId"`
	Status             model.LogStatus `json:"status"`
	ResultURL          string          `json:"resultUrl,omitempty"`
	ResultThumbnailURL string          `json:"resultThumbnailUrl,omitempty"`
	Time               int             `json:"time"`
	ErrorMessage       string          `json:"errorMessage,omitempty"`
}

// Predict 提交去雾预测任务（异步）
// 流程：校验算法 → 校验权益扣减配额 → 检查缓存 → 写日志(processing) → 启动 goroutine 执行 → 立即返回
func (s *PredictionService) Predict(ctx context.Context, algorithmID int64, imageURL string, params string, userID int64) (*PredictionResult, error) {
	startTime := time.Now()
	algorithm, err := s.algoRepo.FindByID(ctx, algorithmID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
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
				metrics.RecordPrediction("success", time.Since(startTime).Seconds())
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
	s.lifecycle.Go(func(ctx context.Context) {
		s.executeAsync(ctx, logID, algorithmID, imageURL, params, imageMD5, userID)
	})

	return &PredictionResult{
		LogID:  logID,
		Status: model.LogStatusProcessing,
	}, nil
}

// executeAsync 异步执行预测任务，更新日志状态
func (s *PredictionService) executeAsync(ctx context.Context, logID, algorithmID int64, imageURL, params, imageMD5 string, userID int64) {
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
		metrics.RecordPrediction("failure", time.Since(startTime).Seconds())
		s.refundQuota(ctx, userID)
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
			metrics.RecordPrediction("failure", time.Since(startTime).Seconds())
			s.refundQuota(ctx, userID)
			return
		}
	}

	elapsed := int(time.Since(startTime).Seconds())

	if model.LogStatus(resp.Status) == model.LogStatusFailed {
		errMsg := resp.ErrorMessage
		if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
			logger.Error("更新预测日志失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
		}
		metrics.RecordPrediction("failure", time.Since(startTime).Seconds())
		s.refundQuota(ctx, userID)
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

	metrics.RecordPrediction("success", time.Since(startTime).Seconds())

	logger.Info("异步去雾预测完成",
		zap.Int64("algorithmID", algorithmID),
		zap.Int64("logID", logID))
}

// refundQuota 预测失败时回补用户配额
func (s *PredictionService) refundQuota(ctx context.Context, userID int64) {
	if s.memberSvc != nil {
		if err := s.memberSvc.RefundQuota(ctx, userID, memberservice.QuotaTypeDehaze); err != nil {
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

// BatchPredictionInput 批量预测单项
type BatchPredictionInput struct {
	FileID   *int64 `json:"fileId"`
	ImageURL string `json:"imageUrl"`
	Params   string `json:"params"`
}

// BatchPredict 批量处理
func (s *PredictionService) BatchPredict(ctx context.Context, algorithmID int64, items []BatchPredictionInput, userID int64) ([]PredictionResult, error) {
	// 校验批量上限
	levelCode, err := s.memberSvc.GetLevelCode(ctx, userID)
	if err != nil {
		return nil, err
	}
	batchLimit, err := s.memberSvc.GetBatchLimit(ctx, levelCode)
	if err != nil {
		return nil, err
	}
	if batchLimit <= 0 {
		batchLimit = 5
	}
	if len(items) > batchLimit {
		return nil, common.NewBizError(common.BUSINESS_ERROR, "批量处理数量超过上限")
	}
	if len(items) > 20 {
		return nil, common.NewBizError(common.BUSINESS_ERROR, "批量处理最多20张")
	}

	results := make([]PredictionResult, 0, len(items))
	for _, item := range items {
		imageURL := item.ImageURL
		result, err := s.Predict(ctx, algorithmID, imageURL, item.Params, userID)
		if err != nil {
			results = append(results, PredictionResult{
				Status:       model.LogStatusFailed,
				ErrorMessage: err.Error(),
			})
			continue
		}
		results = append(results, *result)
	}
	return results, nil
}

// QuotaVO 配额视图
type QuotaVO struct {
	Remaining int `json:"remaining"`
	Total     int `json:"total"`
	Used      int `json:"used"`
}

// GetQuota 查询剩余处理次数
func (s *PredictionService) GetQuota(ctx context.Context, userID int64) (*QuotaVO, error) {
	profile, err := s.memberSvc.GetProfile(ctx, userID)
	if err != nil {
		return nil, err
	}

	totalQuota := profile.MonthlyDehazeQuota
	used := profile.MonthlyDehazeUsed
	remaining := totalQuota - used
	if remaining < 0 {
		remaining = 0
	}
	return &QuotaVO{
		Remaining: remaining,
		Total:     totalQuota,
		Used:      used,
	}, nil
}
