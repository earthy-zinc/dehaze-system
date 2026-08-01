package compare

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// CompareService 效果对比服务
type CompareService struct {
	evalRepo evalrepo.IEvalLogRepository
	predRepo predrepo.IPredLogRepository
	algoRepo algorepo.IAlgorithmRepository
}

func NewCompareService(evalRepo evalrepo.IEvalLogRepository, predRepo predrepo.IPredLogRepository, algoRepo algorepo.IAlgorithmRepository) *CompareService {
	return &CompareService{evalRepo: evalRepo, predRepo: predRepo, algoRepo: algoRepo}
}

// CompareReportForm 对比报告生成表单
type CompareReportForm struct {
	LogID          int64  `json:"logId" binding:"required"`
	Format         string `json:"format" binding:"required"`
	IncludeMetrics *bool  `json:"includeMetrics"`
	IncludeFilters *bool  `json:"includeFilters"`
}

// CompareReportResultVO 对比报告结果
type CompareReportResultVO struct {
	TaskID       int64           `json:"taskId"`
	Status       model.LogStatus `json:"status"`
	DownloadURL  string          `json:"downloadUrl,omitempty"`
	ErrorMessage string          `json:"errorMessage,omitempty"`
}

// GenerateReport 生成对比报告（异步任务）
func (s *CompareService) GenerateReport(ctx context.Context, userID int64, form *CompareReportForm) (*CompareReportResultVO, error) {
	predLog, err := s.predRepo.FindByID(ctx, form.LogID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "处理记录不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询处理记录失败", err)
	}
	if predLog.Status != model.LogStatusCompleted {
		return nil, common.NewBizError(common.BUSINESS_ERROR, "处理任务尚未完成，无法生成报告")
	}

	includeMetrics := form.IncludeMetrics != nil && *form.IncludeMetrics
	includeFilters := form.IncludeFilters != nil && *form.IncludeFilters

	params := map[string]any{
		"logId":          form.LogID,
		"format":         form.Format,
		"includeMetrics": includeMetrics,
		"includeFilters": includeFilters,
	}
	paramsJSON, _ := json.Marshal(params)

	reportTask := &model.SysEvalLog{
		BaseModel:   model.BaseModel{CreateBy: userID},
		AlgorithmID: predLog.AlgorithmID,
		PredURL:     predLog.OriginURL,
		GtURL:       predLog.PredURL,
		PredMD5:     predLog.OriginMD5,
		GtMD5:       predLog.PredMD5,
		Status:      model.LogStatusProcessing,
	}
	paramsStr := string(paramsJSON)
	reportTask.Result = &paramsStr

	if err := s.evalRepo.Create(ctx, reportTask); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建报告任务失败", err)
	}

	go s.generateReportAsync(reportTask.ID, predLog)

	return &CompareReportResultVO{
		TaskID: reportTask.ID,
		Status: model.LogStatusProcessing,
	}, nil
}

// generateReportAsync 异步生成对比报告
func (s *CompareService) generateReportAsync(taskID int64, predLog *model.SysPredLog) {
	ctx := context.Background()
	startTime := time.Now()

	algorithmName := "未知算法"
	if predLog.AlgorithmID > 0 {
		algo, err := s.algoRepo.FindByID(ctx, predLog.AlgorithmID)
		if err == nil && algo != nil {
			algorithmName = algo.Name
		}
	}

	html := buildReportHTML(predLog, algorithmName)
	generatedAt := time.Now().Format("2006-01-02 15:04:05")

	result := map[string]string{
		"reportHtml":  html,
		"generatedAt": generatedAt,
	}
	resultJSON, _ := json.Marshal(result)
	resultStr := string(resultJSON)
	elapsed := int(time.Since(startTime).Seconds())

	if err := s.evalRepo.UpdateResult(ctx, taskID, model.LogStatusCompleted, resultStr, elapsed); err != nil {
		logger.Error("更新报告任务完成状态失败", zap.Int64("taskID", taskID), zap.Error(err))
		return
	}

	logger.Info("对比报告生成完成", zap.Int64("taskID", taskID))
}

// GetReportTaskStatus 查询报告任务状态
func (s *CompareService) GetReportTaskStatus(ctx context.Context, taskID int64) (*CompareReportResultVO, error) {
	log, err := s.evalRepo.FindByID(ctx, taskID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "报告不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询报告任务失败", err)
	}

	vo := &CompareReportResultVO{
		TaskID: log.ID,
		Status: log.Status,
	}
	switch log.Status {
	case model.LogStatusCompleted:
		vo.DownloadURL = fmt.Sprintf("/api/v1/compare/report/%d?download=true", log.ID)
	case model.LogStatusFailed:
		if log.ErrorMessage != nil {
			vo.ErrorMessage = *log.ErrorMessage
		}
	}
	return vo, nil
}

// GetReportHTML 获取报告HTML内容（用于下载）
func (s *CompareService) GetReportHTML(ctx context.Context, taskID int64) (string, error) {
	log, err := s.evalRepo.FindByID(ctx, taskID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return "", common.NewBizError(common.RESOURCE_NOT_FOUND, "报告不存在")
		}
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询报告任务失败", err)
	}

	if log.Status == model.LogStatusProcessing {
		return "", common.NewBizError(common.BUSINESS_ERROR, "报告尚未生成完成")
	}
	if log.Status == model.LogStatusFailed {
		errMsg := "报告生成失败"
		if log.ErrorMessage != nil {
			errMsg += "：" + *log.ErrorMessage
		}
		return "", common.NewBizError(common.SYSTEM_EXECUTION_ERROR, errMsg)
	}

	if log.Result == nil || *log.Result == "" {
		return "", common.NewBizError(common.RESOURCE_NOT_FOUND, "报告内容为空")
	}

	var result map[string]string
	if err := json.Unmarshal([]byte(*log.Result), &result); err != nil {
		return "", common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "报告内容解析失败", err)
	}

	html := result["reportHtml"]
	if html == "" {
		return "", common.NewBizError(common.RESOURCE_NOT_FOUND, "报告内容为空")
	}
	return html, nil
}

// buildReportHTML 生成对比报告HTML
func buildReportHTML(predLog *model.SysPredLog, algorithmName string) string {
	now := time.Now().Format("2006-01-02 15:04:05")
	originURL := predLog.OriginURL
	resultURL := predLog.PredURL
	algoID := predLog.AlgorithmID
	processTime := predLog.Time

	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>去雾效果对比报告</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); color: #fff; padding: 30px; }
        .header h1 { font-size: 24px; margin-bottom: 8px; }
        .header .meta { font-size: 14px; opacity: 0.85; }
        .section { padding: 24px 30px; border-bottom: 1px solid #eee; }
        .section:last-child { border-bottom: none; }
        .section h2 { font-size: 18px; color: #667eea; margin-bottom: 16px; }
        .comparison { display: flex; gap: 20px; flex-wrap: wrap; }
        .image-card { flex: 1; min-width: 280px; }
        .image-card .label { font-size: 14px; color: #666; margin-bottom: 8px; font-weight: 500; }
        .image-card img { width: 100%%; border-radius: 6px; border: 1px solid #e0e0e0; }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
        .info-item { background: #f8f9ff; padding: 12px 16px; border-radius: 6px; }
        .info-item .label { font-size: 12px; color: #999; margin-bottom: 4px; }
        .info-item .value { font-size: 16px; font-weight: 500; }
        .footer { text-align: center; padding: 20px; color: #999; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>去雾效果对比报告</h1>
            <div class="meta">算法：%s | 生成时间：%s</div>
        </div>
        <div class="section">
            <h2>图片对比</h2>
            <div class="comparison">
                <div class="image-card">
                    <div class="label">原图</div>
                    <img src="%s" alt="原图" onerror="this.style.display='none'" />
                </div>
                <div class="image-card">
                    <div class="label">处理结果</div>
                    <img src="%s" alt="处理结果" onerror="this.style.display='none'" />
                </div>
            </div>
        </div>
        <div class="section">
            <h2>处理信息</h2>
            <div class="info-grid">
                <div class="info-item">
                    <div class="label">算法名称</div>
                    <div class="value">%s</div>
                </div>
                <div class="info-item">
                    <div class="label">算法ID</div>
                    <div class="value">%d</div>
                </div>
                <div class="info-item">
                    <div class="label">处理时间</div>
                    <div class="value">%d ms</div>
                </div>
                <div class="info-item">
                    <div class="label">任务状态</div>
                    <div class="value">已完成</div>
                </div>
            </div>
        </div>
        <div class="footer">
            本报告由 Dehaze 系统自动生成
        </div>
    </div>
</body>
</html>`, algorithmName, now, originURL, resultURL, algorithmName, algoID, processTime)
}
