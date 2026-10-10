package prediction

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/lifecycle"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/metrics"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// imageEditModelType 图像编辑模型类型（统一模型体系）
const imageEditModelType = "image_edit"

// imageEditDefaultModelDictKey 平台默认图像模型字典键（sys_dict: ai_model.default_model.image_edit）
const imageEditDefaultModelDictKey = "default_model.image_edit"

// PredictionService 图像任务编排服务（提交-轮询任务模式）
// 模型路由解析：请求 model > 平台默认（字典 ai_model.default_model.image_edit）；实际推理经算法服务
// （Python `/api/v1/images/edits`）执行，本服务管理任务日志与状态流转。
type PredictionService struct {
	repo         predrepo.IPredLogRepository
	modelRepo    *airepo.ModelRepository
	providerRepo *airepo.ProviderRepository
	dictSvc      dictservice.IDictService
	client       *algo.Client
	lifecycle    *lifecycle.Manager
}

func NewPredictionService(
	repo predrepo.IPredLogRepository,
	modelRepo *airepo.ModelRepository,
	providerRepo *airepo.ProviderRepository,
	dictSvc dictservice.IDictService,
	client *algo.Client,
	lm *lifecycle.Manager,
) *PredictionService {
	return &PredictionService{
		repo:         repo,
		modelRepo:    modelRepo,
		providerRepo: providerRepo,
		dictSvc:      dictSvc,
		client:       client,
		lifecycle:    lm,
	}
}

// PredictionResult 图像任务结果 VO
type PredictionResult struct {
	TaskID             int64           `json:"taskId"`
	Status             model.LogStatus `json:"status"`
	ResultURL          string          `json:"resultUrl,omitempty"`
	ResultMd5          string          `json:"resultMd5,omitempty"`
	OriginMd5          string          `json:"originMd5,omitempty"`
	ResultThumbnailURL string          `json:"resultThumbnailUrl,omitempty"`
	Time               int             `json:"time"`
	ErrorMessage       string          `json:"errorMessage,omitempty"`
}

// resolveModelID 解析图像模型标识：请求显式 model 优先，其次平台默认；不可用即拒绝调用。
// 与 python `resolve_image_model` 同口径：模型须存在、启用、类型 image_edit，供应商须存在且启用。
func (s *PredictionService) resolveModelID(ctx context.Context, modelSlug string) (string, int64, error) {
	modelID := strings.TrimSpace(modelSlug)
	if modelID == "" {
		defaultID, err := s.defaultImageModelID(ctx)
		if err != nil {
			return "", 0, err
		}
		modelID = defaultID
	}
	row, err := s.modelRepo.GetByModelID(ctx, modelID)
	if err != nil {
		return "", 0, common.WrapBizError(common.DATABASE_ERROR, "查询图像模型失败", err)
	}
	if row == nil || row.Status != 1 || row.ModelType != imageEditModelType {
		return "", 0, common.NewBizError(common.PARAM_ERROR,
			"图像模型不可用: "+modelID+"（不存在/已停用/类型不符）")
	}
	provider, err := s.providerRepo.GetByID(ctx, row.ProviderID)
	if err != nil {
		return "", 0, common.WrapBizError(common.DATABASE_ERROR, "查询模型供应商失败", err)
	}
	if provider == nil || provider.Status != 1 {
		return "", 0, common.NewBizError(common.BUSINESS_ERROR, "模型 "+modelID+" 的供应商不可用")
	}
	return modelID, row.ProviderID, nil
}

// defaultImageModelID 平台默认图像模型：sys_dict 类型 ai_model / 键 default_model.image_edit。
// 未配置即报 A0500，不回落代码常量（对齐 python `resolve_image_model`）。
func (s *PredictionService) defaultImageModelID(ctx context.Context) (string, error) {
	if s.dictSvc == nil {
		return "", common.NewBizError(common.BUSINESS_ERROR, "平台未配置默认图像模型")
	}
	options, err := s.dictSvc.GetByTypeCode(ctx, "ai_model")
	if err != nil {
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询平台默认图像模型失败", err)
	}
	for _, option := range options {
		if option.Label != imageEditDefaultModelDictKey {
			continue
		}
		if value := strings.TrimSpace(fmt.Sprint(option.Value)); value != "" {
			return value, nil
		}
	}
	return "", common.NewBizError(common.BUSINESS_ERROR,
		"平台未配置默认图像模型（sys_dict: ai_model.default_model.image_edit 缺项）")
}

// Predict 提交图像任务（异步）
// 流程：解析模型路由 → 写日志(processing) → 启动 goroutine 执行 → 立即返回
func (s *PredictionService) Predict(ctx context.Context, modelSlug, imageURL, params string, userID int64, recommendedBy *int64) (*PredictionResult, error) {
	modelID, providerID, err := s.resolveModelID(ctx, modelSlug)
	if err != nil {
		return nil, err
	}

	// origin_md5 是原图内容 MD5，只有算法服务下载图片后能算出（Go 侧只有 URL，不得用 URL 的 MD5 冒充），
	// 创建时先留空，完成后由算法服务响应回填
	predLog := &model.SysPredLog{
		BaseModel:     model.BaseModel{CreateBy: userID},
		ModelID:       modelID,
		ProviderID:    &providerID,
		ActualModel:   &modelID,
		OriginURL:     imageURL,
		RecommendedBy: recommendedBy,
		Status:        model.LogStatusProcessing,
	}
	if err := s.repo.Create(ctx, predLog); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建图像任务失败", err)
	}

	logID := predLog.ID
	s.lifecycle.Go(func(ctx context.Context) {
		s.executeAsync(ctx, logID, modelID, imageURL, params)
	})

	return &PredictionResult{
		TaskID: logID,
		Status: model.LogStatusProcessing,
	}, nil
}

// executeAsync 异步执行图像任务，更新日志状态
func (s *PredictionService) executeAsync(ctx context.Context, logID int64, modelID, imageURL, params string) {
	startTime := time.Now()

	resp, err := s.client.Predict(ctx, &algo.PredictionRequest{
		Model:    modelID,
		ImageURL: imageURL,
		Params:   params,
	})

	if err != nil {
		elapsed := int(time.Since(startTime).Seconds())
		logger.Error("异步图像任务失败",
			zap.String("modelID", modelID),
			zap.Int64("logID", logID),
			zap.Error(err))
		errMsg := err.Error()
		if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
			logger.Error("更新图像任务失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
		}
		metrics.RecordPrediction("failure", time.Since(startTime).Seconds())
		return
	}

	if model.LogStatus(resp.Status) == model.LogStatusProcessing {
		var pollErr error
		resp, pollErr = s.pollPredTask(ctx, resp.TaskID)
		if pollErr != nil {
			elapsed := int(time.Since(startTime).Seconds())
			errMsg := pollErr.Error()
			if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
				logger.Error("更新图像任务失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
			}
			metrics.RecordPrediction("failure", time.Since(startTime).Seconds())
			return
		}
	}

	elapsed := int(time.Since(startTime).Seconds())

	if model.LogStatus(resp.Status) == model.LogStatusFailed {
		errMsg := resp.ErrorMessage
		if updateErr := s.repo.UpdateStatus(ctx, logID, model.LogStatusFailed, errMsg, elapsed); updateErr != nil {
			logger.Error("更新图像任务失败状态失败", zap.Int64("logID", logID), zap.Error(updateErr))
		}
		metrics.RecordPrediction("failure", time.Since(startTime).Seconds())
		return
	}

	// pred_md5/origin_md5 一律取算法服务返回的内容 MD5，不能用 URL 的 MD5 冒充
	if err := s.repo.UpdateResult(ctx, logID, model.LogStatusCompleted, resp.ResultURL, resp.ResultMd5, resp.OriginMd5, resp.Time); err != nil {
		logger.Error("更新图像任务完成状态失败", zap.Int64("logID", logID), zap.Error(err))
	}

	metrics.RecordPrediction("success", time.Since(startTime).Seconds())

	logger.Info("异步图像任务完成",
		zap.String("modelID", modelID),
		zap.Int64("logID", logID))
}

// pollPredTask 轮询算法服务图像任务状态直到终态
func (s *PredictionService) pollPredTask(ctx context.Context, taskID int64) (*algo.PredictionResponse, error) {
	const interval = 2 * time.Second
	const timeout = 5 * time.Minute
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}

		result, err := s.client.GetPredTaskStatus(ctx, taskID)
		if err != nil {
			logger.Warn("轮询图像任务状态失败",
				zap.Int64("taskID", taskID),
				zap.Error(err))
			continue
		}
		if model.LogStatus(result.Status) == model.LogStatusCompleted || model.LogStatus(result.Status) == model.LogStatusFailed {
			return result, nil
		}
	}
	return nil, fmt.Errorf("算法服务图像任务 %d 轮询超时", taskID)
}

// GetTaskStatus 查询任务状态，根据 status 返回不同字段
// 归属校验：仅任务本人可查询（含结果图 URL），他人任务与不存在任务同口径防枚举
func (s *PredictionService) GetTaskStatus(ctx context.Context, id int64, userID int64) (*PredictionResult, error) {
	log, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "图像任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询图像任务失败", err)
	}
	if log.CreateBy != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "图像任务不存在")
	}

	result := &PredictionResult{
		TaskID: log.ID,
		Status: log.Status,
	}
	switch log.Status {
	case model.LogStatusCompleted:
		result.ResultURL = log.PredURL
		result.ResultMd5 = log.PredMD5
		result.OriginMd5 = log.OriginMD5
		result.Time = log.Time
	case model.LogStatusFailed:
		if log.ErrorMessage != nil {
			result.ErrorMessage = *log.ErrorMessage
		}
		result.Time = log.Time
	}
	return result, nil
}

// CancelTask 取消图像任务（幂等，对齐 python `cancel_task`）：
//   - 仅本人任务：不存在或非本人一律 A0401「图像任务不存在」（防枚举）；
//   - 仅"处理中"可取消：置已取消（带 processing 前置，并发取消只生效一次）；
//   - 已完成/已失败/已取消：幂等返回当前状态。
func (s *PredictionService) CancelTask(ctx context.Context, logID, userID int64) (*PredictionResult, error) {
	log, err := s.repo.FindByID(ctx, logID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "图像任务不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询图像任务失败", err)
	}
	if log.CreateBy != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "图像任务不存在")
	}

	if log.Status != model.LogStatusProcessing {
		return &PredictionResult{TaskID: log.ID, Status: log.Status}, nil
	}

	if _, err := s.repo.MarkCancelled(ctx, logID); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "取消图像任务失败", err)
	}
	return &PredictionResult{TaskID: log.ID, Status: model.LogStatusCancelled}, nil
}

// GetLogPage 分页查询当前用户的图像任务历史
func (s *PredictionService) GetLogPage(ctx context.Context, modelID string, userID int64, pageNum, pageSize int) (*common.PageResult, error) {
	list, total, err := s.repo.FindPage(ctx, modelID, userID, pageNum, pageSize)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询图像任务列表失败", err)
	}
	return &common.PageResult{List: list, Total: total, Page: pageNum, PageSize: pageSize}, nil
}
