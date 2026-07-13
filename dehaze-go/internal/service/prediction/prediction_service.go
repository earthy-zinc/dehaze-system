package prediction

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	algo "github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// md5Hex 计算字符串的 MD5 十六进制表示（32 位）
func md5Hex(s string) string {
	h := md5.Sum([]byte(s))
	return fmt.Sprintf("%x", h)
}

const (
	predCachePrefix = "pred:"
	predCacheTTL    = 24 * time.Hour
)

// PredictionService 去雾预测服务
type PredictionService struct {
	repo     predrepo.IPredLogRepository
	client   *algo.Client
	cache    types.ICache
}

// NewPredictionService 创建预测服务实例
func NewPredictionService(repo predrepo.IPredLogRepository, client *algo.Client, cache types.ICache) *PredictionService {
	return &PredictionService{repo: repo, client: client, cache: cache}
}

// PredictionResult 预测结果
type PredictionResult struct {
	LogID              int64  `json:"logId"`
	ResultURL          string `json:"resultUrl"`
	ResultThumbnailURL string `json:"resultThumbnailUrl"`
	Time               int    `json:"time"`
	FromCache          bool   `json:"fromCache"`
}

// Predict 执行去雾预测（带 Redis 缓存：key = pred:{algorithmId}:{imageMd5}）
func (s *PredictionService) Predict(ctx context.Context, algorithmID int64, imageURL string, params string, userID int64) (*PredictionResult, error) {
	// 1. 计算图片 URL 的 MD5 作为缓存键
	imageMD5 := md5Hex(imageURL)

	// 2. 检查 Redis 缓存
	if s.cache != nil {
		cacheKey := fmt.Sprintf("%s%d:%s", predCachePrefix, algorithmID, imageMD5)
		if cached, err := s.cache.Get(ctx, cacheKey); err == nil && cached != "" {
			var result PredictionResult
			if json.Unmarshal([]byte(cached), &result) == nil {
				logger.Info("预测结果命中缓存", zap.Int64("algorithmID", algorithmID))
				result.FromCache = true
				return &result, nil
			}
		}
	}

	// 3. 调用 Python 算法服务
	resp, err := s.client.Predict(ctx, &algo.PredictionRequest{
		AlgorithmID: algorithmID,
		ImageURL:    imageURL,
		Params:      params,
	})
	if err != nil {
		logger.Error("去雾预测失败", zap.Int64("algorithmID", algorithmID), zap.Error(err))
		return nil, common.WrapBizError(common.CALL_THIRD_PARTY_SERVICE_ERROR, "去雾处理失败", err)
	}

	// 4. 写入预测日志
	predLog := &model.SysPredLog{
		AlgorithmID: algorithmID,
		OriginMD5:   imageMD5,
		OriginURL:   imageURL,
		PredMD5:     md5Hex(resp.ResultURL),
		PredURL:     resp.ResultURL,
		Time:        resp.Time,
		CreateBy:    &userID,
	}
	if err := s.repo.Create(ctx, predLog); err != nil {
		logger.Error("写入预测日志失败", zap.Error(err))
	}

	// 5. 缓存结果到 Redis
	result := &PredictionResult{
		LogID:              predLog.ID,
		ResultURL:          resp.ResultURL,
		ResultThumbnailURL: resp.ResultThumbnailURL,
		Time:               resp.Time,
	}
	if s.cache != nil {
		cacheKey := fmt.Sprintf("%s%d:%s", predCachePrefix, algorithmID, imageMD5)
		if data, err := json.Marshal(result); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), predCacheTTL)
		}
	}

	logger.Info("去雾预测完成", zap.Int64("algorithmID", algorithmID), zap.Int64("logID", predLog.ID))
	return result, nil
}

// GetLogByID 查询预测日志
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
