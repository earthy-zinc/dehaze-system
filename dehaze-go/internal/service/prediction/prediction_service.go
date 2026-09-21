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
func (s *PredictionService) Predict(ctx context.Context, algorithmID int64, imageURL string, params string, userID int64, recommendedBy *int64) (*PredictionResult, error) {
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
					BaseModel:     model.BaseModel{CreateBy: userID},
					AlgorithmID:   algorithmID,
					OriginMD5:     imageMD5,
					OriginURL:     imageURL,
					PredMD5:       utils.MD5Hex(cached.ResultURL),
					PredURL:       cached.ResultURL,
					RecommendedBy: recommendedBy,
					Time:          cached.Time,
					Status:        model.LogStatusCompleted,
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
		BaseModel:     model.BaseModel{CreateBy: userID},
		AlgorithmID:   algorithmID,
		OriginMD5:     imageMD5,
		OriginURL:     imageURL,
		RecommendedBy: recommendedBy,
		Status:        model.LogStatusProcessing,
	}
	if err := s.repo.Create(ctx, predLog); err != nil {
		// 日志创建失败时任务未启动，归还已扣减配额防泄漏
		s.refundQuota(ctx, userID)
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
// 归属校验：仅任务本人可查询（含结果图 URL），他人任务与不存在任务同口径防枚举
func (s *PredictionService) GetTaskStatus(ctx context.Context, id int64, userID int64) (*PredictionResult, error) {
	log, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "预测任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询预测日志失败", err)
	}
	if log.CreateBy != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "预测任务不存在")
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

// CancelTask 取消预测任务（幂等，对齐 python `cancel_task`）：
//   - 仅本人任务：不存在或非本人一律 A0401「预测任务不存在」（防枚举）；
//   - 仅"处理中"可取消：置已取消 + 回滚已扣配额（带 processing 前置，并发取消只回滚一次）；
//   - 已完成/已失败/已取消：幂等返回当前状态，不重复回滚配额。
func (s *PredictionService) CancelTask(ctx context.Context, logID, userID int64) (*PredictionResult, error) {
	log, err := s.repo.FindByID(ctx, logID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "预测任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询预测任务失败", err)
	}
	if log.CreateBy != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "预测任务不存在")
	}

	if log.Status != model.LogStatusProcessing {
		return &PredictionResult{LogID: log.ID, Status: log.Status}, nil
	}

	transitioned, err := s.repo.MarkCancelled(ctx, logID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "取消预测任务失败", err)
	}
	if transitioned {
		s.refundQuota(ctx, userID)
	}
	return &PredictionResult{LogID: log.ID, Status: model.LogStatusCancelled}, nil
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

// GetLogPage 分页查询当前用户的预测日志
func (s *PredictionService) GetLogPage(ctx context.Context, algorithmID int64, userID int64, pageNum, pageSize int) (*common.PageResult, error) {
	list, total, err := s.repo.FindPage(ctx, algorithmID, userID, pageNum, pageSize)
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
func (s *PredictionService) BatchPredict(ctx context.Context, algorithmID int64, items []BatchPredictionInput, userID int64, recommendedBy *int64) ([]PredictionResult, error) {
	// T-DH-027：空 items 直接参数校验失败（python `prediction_service.batch_predict` 同口径，
	// A0400 而非"成功返回空结果"）
	if len(items) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "批量处理图片列表不能为空")
	}

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

	results := make([]PredictionResult, 0, len(items))
	for _, item := range items {
		imageURL := item.ImageURL
		result, err := s.Predict(ctx, algorithmID, imageURL, item.Params, userID, recommendedBy)
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
	// ResetDate 配额重置日期（月度配额为下月 1 日，格式 yyyy-MM-dd；python QuotaResponse.resetDate 同口径）
	ResetDate string `json:"resetDate"`
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
	// 重置日期 = 下月 1 日（time.Date 自动进位，12 月 → 次年 1 月，与 python 的 if 分支等价）
	now := time.Now()
	resetDate := time.Date(now.Year(), now.Month()+1, 1, 0, 0, 0, 0, now.Location())

	return &QuotaVO{
		Remaining: remaining,
		Total:     totalQuota,
		Used:      used,
		ResetDate: resetDate.Format("2006-01-02"),
	}, nil
}
