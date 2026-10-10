package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/metrics"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// EvaluationService 去雾效果评估服务
type EvaluationService struct {
	repo      evalrepo.IEvalLogRepository
	modelRepo *airepo.ModelRepository
	client    *algo.Client
}

func NewEvaluationService(repo evalrepo.IEvalLogRepository, modelRepo *airepo.ModelRepository, client *algo.Client) *EvaluationService {
	return &EvaluationService{repo: repo, modelRepo: modelRepo, client: client}
}

// EvaluationResult 评估结果 VO
type EvaluationResult struct {
	LogID        int64              `json:"logId"`
	Status       model.LogStatus    `json:"status"`
	Metrics      map[string]float64 `json:"metrics,omitempty"`
	PredMd5      string             `json:"predMd5,omitempty"`
	GtMd5        string             `json:"gtMd5,omitempty"`
	Time         int                `json:"time"`
	ErrorMessage string             `json:"errorMessage,omitempty"`
}

// Evaluate 提交效果评估任务（异步）
// 流程：校验模型存在 → 写日志(processing) → 启动 goroutine 执行 → 立即返回
func (s *EvaluationService) Evaluate(ctx context.Context, modelID, predURL, gtURL string, userID int64) (*EvaluationResult, error) {
	imgModel, err := s.modelRepo.GetByModelID(ctx, modelID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询图像模型失败", err)
	}
	if imgModel == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "图像模型不存在")
	}

	// pred_md5/gt_md5 是图片内容 MD5，由算法服务下载图片后计算并随响应返回；
	// 创建 processing 日志时先留空，完成后回填（禁止用 URL 的 MD5 冒充）
	evalLog := &model.SysEvalLog{
		BaseModel: model.BaseModel{CreateBy: userID},
		ModelID:   modelID,
		PredURL:   predURL,
		GtURL:     gtURL,
		Status:    model.LogStatusProcessing,
		TaskType:  "evaluation",
	}
	if err := s.repo.Create(ctx, evalLog); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建评估日志失败", err)
	}

	logID := evalLog.ID
	go s.executeAsync(logID, modelID, predURL, gtURL)

	return &EvaluationResult{
		LogID:  logID,
		Status: model.LogStatusProcessing,
	}, nil
}

// executeAsync 异步执行评估任务，更新日志状态
func (s *EvaluationService) executeAsync(logID int64, modelID, predURL, gtURL string) {
	ctx := context.Background()
	startTime := time.Now()

	resp, err := s.client.Evaluate(ctx, &algo.EvaluationRequest{
		ModelID: modelID,
		PredURL: predURL,
		GtURL:   gtURL,
	})

	if err != nil {
		elapsed := int(time.Since(startTime).Seconds())
		logger.Error("异步效果评估失败",
			zap.String("modelID", modelID),
			zap.Int64("logID", logID),
			zap.Error(err))
		errMsg := err.Error()
		if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
			logger.Error("更新评估日志失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
		}
		metrics.RecordEvaluation("failure", time.Since(startTime).Seconds())
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
		return
	}

	metricsJSON, _ := json.Marshal(resp.Metrics)
	resultStr := string(metricsJSON)
	// pred_md5/gt_md5 取算法服务返回的内容 MD5
	if err := s.repo.UpdateResult(ctx, logID, model.LogStatusCompleted, resultStr, resp.PredMd5, resp.GtMd5, resp.Time); err != nil {
		logger.Error("更新评估日志完成状态失败", zap.Int64("logID", logID), zap.Error(err))
	}

	metrics.RecordEvaluation("success", time.Since(startTime).Seconds())

	logger.Info("异步效果评估完成",
		zap.String("modelID", modelID),
		zap.Int64("logID", logID))
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
// 归属校验：仅任务本人可查询，他人任务与不存在任务同口径防枚举
func (s *EvaluationService) GetTaskStatus(ctx context.Context, id int64, userID int64) (*EvaluationResult, error) {
	log, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "评估任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评估日志失败", err)
	}
	if log.CreateBy != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "评估任务不存在")
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
		result.PredMd5 = log.PredMD5
		result.GtMd5 = log.GtMD5
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

// GetUserMetricsPage 获取当前用户的评估指标历史（仅已完成状态）
func (s *EvaluationService) GetUserMetricsPage(ctx context.Context, userID int64, modelID string, pageNum, pageSize int) (*common.PageResult, error) {
	list, total, err := s.repo.FindPageByUser(ctx, userID, modelID, pageNum, pageSize)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评估指标历史失败", err)
	}

	type EvalMetricsVO struct {
		ID         int64              `json:"id"`
		ModelID    string             `json:"modelId"`
		PredURL    string             `json:"predUrl"`
		GtURL      string             `json:"gtUrl"`
		Time       int                `json:"time"`
		Status     model.LogStatus    `json:"status"`
		Metrics    map[string]float64 `json:"metrics"`
		CreateTime string             `json:"createTime"`
	}
	vos := make([]EvalMetricsVO, 0, len(list))
	for _, log := range list {
		vo := EvalMetricsVO{
			ID:         log.ID,
			ModelID:    log.ModelID,
			PredURL:    log.PredURL,
			GtURL:      log.GtURL,
			Time:       log.Time,
			Status:     log.Status,
			CreateTime: log.CreatedAt.Format("2006-01-02 15:04:05"),
		}
		if log.Result != nil {
			_ = json.Unmarshal([]byte(*log.Result), &vo.Metrics)
		}
		vos = append(vos, vo)
	}
	return &common.PageResult{List: vos, Total: total, Page: pageNum, PageSize: pageSize}, nil
}

// GetLogPage 分页查询当前用户的评估日志
func (s *EvaluationService) GetLogPage(ctx context.Context, modelID string, userID int64, pageNum, pageSize int) (*common.PageResult, error) {
	list, total, err := s.repo.FindLogPageByUser(ctx, userID, modelID, pageNum, pageSize)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评估日志列表失败", err)
	}
	return &common.PageResult{List: list, Total: total, Page: pageNum, PageSize: pageSize}, nil
}
