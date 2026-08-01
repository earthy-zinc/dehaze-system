package algorithm_select

import (
	"context"
	"math"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	predservice "github.com/earthyzinc/dehaze-go/internal/service/prediction"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

// AlgorithmSelectService 算法选择服务
type AlgorithmSelectService struct {
	db           *gorm.DB
	algorithmRepo algorepo.IAlgorithmRepository
	predLogRepo   predrepo.IPredLogRepository
	predService   *predservice.PredictionService
}

// NewAlgorithmSelectService 创建算法选择服务实例
func NewAlgorithmSelectService(
	db *gorm.DB,
	algorithmRepo algorepo.IAlgorithmRepository,
	predLogRepo predrepo.IPredLogRepository,
	predService *predservice.PredictionService,
) *AlgorithmSelectService {
	return &AlgorithmSelectService{
		db:           db,
		algorithmRepo: algorithmRepo,
		predLogRepo:   predLogRepo,
		predService:   predService,
	}
}

// GetTree 获取算法选择树（仅已发布状态 status=4）
func (s *AlgorithmSelectService) GetTree(ctx context.Context) ([]vo.AlgorithmSelectVO, error) {
	algorithms, err := s.algorithmRepo.FindAll(ctx, nil)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法列表失败", err)
	}

	// 过滤仅已发布
	published := make([]read.Algorithm, 0)
	for _, algo := range algorithms {
		if algo.Status == 4 {
			published = append(published, algo)
		}
	}

	// 按 ParentID 分组构建树
	childrenMap := make(map[int64][]read.Algorithm)
	for _, algo := range published {
		childrenMap[algo.ParentID] = append(childrenMap[algo.ParentID], algo)
	}

	tree := make([]vo.AlgorithmSelectVO, 0)
	for _, algo := range published {
		if algo.ParentID == 0 {
			tree = append(tree, buildSelectTree(algo, childrenMap))
		}
	}
	return tree, nil
}

func buildSelectTree(algo read.Algorithm, childrenMap map[int64][]read.Algorithm) vo.AlgorithmSelectVO {
	item := vo.AlgorithmSelectVO{
		ID:          algo.ID,
		ParentID:    algo.ParentID,
		Name:        algo.Name,
		Type:        algo.Type,
		Img:         algo.Img,
		Description: algo.Description,
		Path:        algo.Path,
		Flops:       algo.Flops,
		Params:      algo.Params,
		ImportPath:  algo.ImportPath,
		Status:      algo.Status,
		Size:        algo.Size,
	}
	for _, child := range childrenMap[algo.ID] {
		item.Children = append(item.Children, buildSelectTree(child, childrenMap))
	}
	return item
}

// GetDetail 获取算法详情（含样例效果图/评分/使用次数）
func (s *AlgorithmSelectService) GetDetail(ctx context.Context, id int64) (*vo.AlgorithmDetailVO, error) {
	algorithm, err := s.algorithmRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil || algorithm.Status != 4 {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在或未发布")
	}

	detail := &vo.AlgorithmDetailVO{
		AlgorithmSelectVO: vo.AlgorithmSelectVO{
			ID:          algorithm.ID,
			ParentID:    algorithm.ParentID,
			Name:        algorithm.Name,
			Type:        algorithm.Type,
			Img:         algorithm.Img,
			Description: algorithm.Description,
			Path:        algorithm.Path,
			Flops:       algorithm.Flops,
			Params:      algorithm.Params,
			ImportPath:  algorithm.ImportPath,
			Status:      int(algorithm.Status),
			Size:        algorithm.Size,
		},
	}

	// 评分统计（从 sys_rating 聚合）
	ratingStats := s.getRatingStats(ctx, id)
	detail.RatingStats = ratingStats
	if ratingStats != nil {
		detail.Rating = ratingStats.Average
	}

	// 使用次数（从 pred_log 统计）
	var usageCount int64
	s.db.WithContext(ctx).Model(&model.SysPredLog{}).
		Where("algorithm_id = ?", id).Count(&usageCount)
	detail.UsageCount = usageCount

	// 样例效果图（最近3条完成的预测记录）
	detail.SampleImages = s.getSampleImages(ctx, id)

	return detail, nil
}

func (s *AlgorithmSelectService) getRatingStats(ctx context.Context, algorithmID int64) *vo.AlgorithmRatingStatsVO {
	type ratingRow struct {
		Rating int8
		Count  int64
	}
	var rows []ratingRow
	err := s.db.WithContext(ctx).Model(&model.SysRating{}).
		Select("rating, COUNT(*) as count").
		Where("algorithm_id = ? AND is_hidden = 0 AND deleted = 0", algorithmID).
		Group("rating").Scan(&rows).Error
	if err != nil {
		return nil
	}

	var totalCount int64
	var totalScore float64
	dist := make(map[int8]int64)
	for _, r := range rows {
		dist[r.Rating] = r.Count
		totalCount += r.Count
		totalScore += float64(r.Rating) * float64(r.Count)
	}

	if totalCount == 0 {
		return nil
	}

	return &vo.AlgorithmRatingStatsVO{
		Average:      math.Round(totalScore/float64(totalCount)*10) / 10,
		Count:        totalCount,
		Distribution: dist,
	}
}

func (s *AlgorithmSelectService) getSampleImages(ctx context.Context, algorithmID int64) []vo.AlgorithmSampleVO {
	var logs []model.SysPredLog
	s.db.WithContext(ctx).Model(&model.SysPredLog{}).
		Where("algorithm_id = ? AND pred_url IS NOT NULL AND pred_url <> ''", algorithmID).
		Order("create_time DESC").Limit(3).Find(&logs)

	samples := make([]vo.AlgorithmSampleVO, 0, len(logs))
	for _, l := range logs {
		samples = append(samples, vo.AlgorithmSampleVO{
			OriginURL: l.OriginURL,
			PredURL:   l.PredURL,
		})
	}
	return samples
}

// Search 搜索算法（按名称模糊匹配已发布算法）
func (s *AlgorithmSelectService) Search(ctx context.Context, keyword string, pageNum, pageSize int) (*vo.PageResult[vo.AlgorithmSelectVO], error) {
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	query := s.db.WithContext(ctx).Model(&model.SysAlgorithm{}).
		Where("status = 4 AND deleted = 0")

	if keyword != "" {
		like := "%" + keyword + "%"
		query = query.Where("name LIKE ? OR type LIKE ? OR description LIKE ?", like, like, like)
	}

	var total int64
	if err := query.Count(&total).Error; err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法数量失败", err)
	}

	var algorithms []model.SysAlgorithm
	if err := query.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&algorithms).Error; err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "搜索算法失败", err)
	}

	list := make([]vo.AlgorithmSelectVO, 0, len(algorithms))
	for _, algo := range algorithms {
		item := vo.AlgorithmSelectVO{
			ID:          algo.ID,
			ParentID:    algo.ParentID,
			Name:        algo.Name,
			Type:        algo.Type,
			Img:         algo.Img,
			Description: algo.Description,
			Path:        algo.Path,
			Flops:       algo.Flops,
			Params:      algo.Params,
			ImportPath:  algo.ImportPath,
			Status:      int(algo.Status),
			Size:        algo.Size,
		}
		// 附加评分
		if stats := s.getRatingStats(ctx, algo.ID); stats != nil {
			item.Rating = stats.Average
		}
		// 附加使用次数
		var count int64
		s.db.WithContext(ctx).Model(&model.SysPredLog{}).
			Where("algorithm_id = ?", algo.ID).Count(&count)
		item.UsageCount = count
		list = append(list, item)
	}

	return &vo.PageResult[vo.AlgorithmSelectVO]{List: list, Total: total}, nil
}

// Test 上传图片测试算法效果（调用预测服务执行单次预测）
func (s *AlgorithmSelectService) Test(ctx context.Context, algorithmID int64, imageURL string, userID int64) (int64, int, error) {
	algorithm, err := s.algorithmRepo.FindByID(ctx, algorithmID)
	if err != nil {
		return 0, 0, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil || algorithm.Status != 4 {
		return 0, 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在或未发布")
	}

	result, err := s.predService.Predict(ctx, algorithmID, imageURL, "", userID)
	if err != nil {
		return 0, 0, err
	}
	return result.LogID, int(result.Status), nil
}

// Compare 算法对比（最多3个算法+图片，分别预测返回结果）
func (s *AlgorithmSelectService) Compare(ctx context.Context, form *bo.AlgorithmCompareForm, userID int64) ([]vo.AlgorithmCompareVO, error) {
	if len(form.AlgorithmIDs) > 3 {
		return nil, common.NewBizError(common.BUSINESS_ERROR, "对比算法数量不能超过3个")
	}
	if len(form.AlgorithmIDs) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "请选择要对比的算法")
	}

	results := make([]vo.AlgorithmCompareVO, 0, len(form.AlgorithmIDs))
	for _, algoID := range form.AlgorithmIDs {
		algorithm, err := s.algorithmRepo.FindByID(ctx, algoID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
		}
		if algorithm == nil || algorithm.Status != 4 {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在或未发布")
		}

		item := vo.AlgorithmCompareVO{
			ID:          algorithm.ID,
			Name:        algorithm.Name,
			Type:        algorithm.Type,
			Description: algorithm.Description,
			Flops:       algorithm.Flops,
			Params:      algorithm.Params,
		}

		// 评分
		if stats := s.getRatingStats(ctx, algoID); stats != nil {
			item.Rating = stats.Average
		}

		// 使用次数
		var usageCount int64
		s.db.WithContext(ctx).Model(&model.SysPredLog{}).
			Where("algorithm_id = ?", algoID).Count(&usageCount)
		item.UsageCount = usageCount

		// 平均处理时间和成功率
		var avgTime float64
		var totalPred int64
		var successCount int64
		s.db.WithContext(ctx).Model(&model.SysPredLog{}).
			Where("algorithm_id = ? AND time IS NOT NULL", algoID).
			Select("COALESCE(AVG(time), 0)").Row().Scan(&avgTime)
		s.db.WithContext(ctx).Model(&model.SysPredLog{}).
			Where("algorithm_id = ?", algoID).Count(&totalPred)
		s.db.WithContext(ctx).Model(&model.SysPredLog{}).
			Where("algorithm_id = ? AND pred_url IS NOT NULL AND pred_url <> ''", algoID).Count(&successCount)
		item.AvgTime = math.Round(avgTime*100) / 100
		if totalPred > 0 {
			item.SuccessRate = math.Round(float64(successCount)/float64(totalPred)*10000) / 100
		}

		// 执行对比预测
		predResult, err := s.predService.Predict(ctx, algoID, form.ImageURL, "", userID)
		if err == nil && predResult.Status == model.LogStatusCompleted {
			item.PredResultURL = predResult.ResultURL
			item.PredTime = predResult.Time
		}

		results = append(results, item)
	}

	return results, nil
}
