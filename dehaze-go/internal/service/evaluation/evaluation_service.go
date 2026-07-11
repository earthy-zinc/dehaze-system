package evaluation

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/earthyzinc/dehaze-go/internal/model"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// EvaluationService 去雾效果评估服务
type EvaluationService struct {
	repo   evalrepo.IEvalLogRepository
	client *algo.Client
}

// NewEvaluationService 创建评估服务实例
func NewEvaluationService(repo evalrepo.IEvalLogRepository, client *algo.Client) *EvaluationService {
	return &EvaluationService{repo: repo, client: client}
}

// Evaluate 执行效果评估
func (s *EvaluationService) Evaluate(ctx context.Context, algorithmID int64, predURL, gtURL string, userID int64) (*algo.EvaluationResponse, error) {
	resp, err := s.client.Evaluate(ctx, &algo.EvaluationRequest{
		AlgorithmID: algorithmID,
		PredURL:     predURL,
		GtURL:       gtURL,
	})
	if err != nil {
		logger.Error("效果评估失败", zap.Int64("algorithmID", algorithmID), zap.Error(err))
		return nil, common.WrapBizError(common.CALL_THIRD_PARTY_SERVICE_ERROR, "效果评估失败", err)
	}

	// 写入评估日志
	metricsJSON, _ := json.Marshal(resp.Metrics)
	resultStr := string(metricsJSON)
	evalLog := &model.SysEvalLog{
		AlgorithmID: algorithmID,
		PredMD5:     fmt.Sprintf("%x", []byte(predURL))[:32],
		PredURL:     predURL,
		GtMD5:       fmt.Sprintf("%x", []byte(gtURL))[:32],
		GtURL:       gtURL,
		Time:        resp.Time,
		Result:      &resultStr,
		CreateBy:    &userID,
	}
	if err := s.repo.Create(ctx, evalLog); err != nil {
		logger.Error("写入评估日志失败", zap.Error(err))
	}
	resp.LogID = evalLog.ID

	logger.Info("效果评估完成", zap.Int64("algorithmID", algorithmID), zap.Int64("logID", evalLog.ID), zap.Bool("qualified", resp.Qualified))
	return resp, nil
}

// GetLogByID 查询评估日志
func (s *EvaluationService) GetLogByID(ctx context.Context, id int64) (*model.SysEvalLog, error) {
	log, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
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
