package recommendation

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	recrepo "github.com/earthyzinc/dehaze-go/internal/repository/recommendation"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

var (
	validHazeLevels  = []string{"light", "moderate", "heavy"}
	validSceneTypes  = []string{"urban", "landscape", "building", "night", "backlight", "indoor"}
	validLightings   = []string{"bright", "normal", "dark", "veryDark", "backlight"}
	validResolutions = []string{"sd", "hd", "uhd"}
	validNoiseLevels = []string{"low", "medium", "high"}
	imageExtensions  = []string{".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
	topN             = 3

	reasonTemplates = map[string]string{
		"urban":     "处理速度快，对城市雾霾效果出色",
		"landscape": "在自然场景下表现稳定，色彩还原度高",
		"building":  "深度模型，对建筑场景处理能力强",
		"night":     "低光照增强组合，避免过度暗化",
		"backlight": "HDR预处理提升暗部细节",
		"indoor":    "室内场景适配，细节保留好",
	}
)

type RecommendationService struct {
	db        *gorm.DB
	recRepo   recrepo.RecommendationRepository
	ruleRepo  recrepo.RuleRepository
	ruleCache []model.SysRecommendationRule
	mu        sync.RWMutex
}

func NewRecommendationService(db *gorm.DB, recRepo recrepo.RecommendationRepository, ruleRepo recrepo.RuleRepository) *RecommendationService {
	return &RecommendationService{
		db:       db,
		recRepo:  recRepo,
		ruleRepo: ruleRepo,
	}
}

func (s *RecommendationService) Analyze(ctx context.Context, form *bo.AnalyzeForm) (*vo.ImageFeatureAnalysisVO, error) {
	imageURL := s.resolveImageURL(form)
	if err := s.validateImageFormat(imageURL); err != nil {
		return nil, err
	}

	hash := fmt.Sprintf("%x", md5.Sum([]byte(imageURL)))
	seed := int(math.Abs(float64(hashCode(hash))))

	return &vo.ImageFeatureAnalysisVO{
		ImageMd5:        hash,
		HazeLevel:       validHazeLevels[seed%len(validHazeLevels)],
		HazeConfidence:  0.5 + float64(seed%50)/100.0,
		SceneType:       validSceneTypes[seed%len(validSceneTypes)],
		SceneConfidence: 0.5 + float64((seed/7)%50)/100.0,
		Lighting:        validLightings[seed%len(validLightings)],
		Complexity:      0.3 + float64((seed/11)%70)/100.0,
		ColorDistribution: vo.ColorDistribution{
			Temperature: 4000.0 + float64(seed%6000),
			Saturation:  0.3 + float64((seed/13)%70)/100.0,
		},
		Resolution: validResolutions[seed%len(validResolutions)],
		NoiseLevel: validNoiseLevels[seed%len(validNoiseLevels)],
	}, nil
}

func (s *RecommendationService) GetAlgorithmRecommendations(ctx context.Context, userID int64, analysisID *int64, imageMd5 string) ([]vo.RecommendedAlgorithmVO, error) {
	rules := s.getEnabledRules(ctx)
	if len(rules) == 0 {
		return []vo.RecommendedAlgorithmVO{}, nil
	}

	// 确定场景类型：优先从 analysisId 查，其次 imageMd5，默认 urban
	sceneType := "urban"
	if analysisID != nil && *analysisID > 0 {
		rec, err := s.recRepo.FindByID(ctx, *analysisID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询推荐记录失败", err)
		}
		if rec != nil && rec.AnalysisResult != nil {
			var ar map[string]any
			if json.Unmarshal([]byte(*rec.AnalysisResult), &ar) == nil {
				if st, ok := ar["sceneType"].(string); ok && containsString(validSceneTypes, st) {
					sceneType = st
				}
			}
		}
	}
	if sceneType == "urban" && imageMd5 != "" {
		var rec model.SysRecommendation
		err := s.db.WithContext(ctx).
			Where("image_md5 = ?", imageMd5).
			Order("id DESC").
			Limit(1).
			First(&rec).Error
		if err == nil && rec.AnalysisResult != nil {
			var ar map[string]any
			if json.Unmarshal([]byte(*rec.AnalysisResult), &ar) == nil {
				if st, ok := ar["sceneType"].(string); ok && containsString(validSceneTypes, st) {
					sceneType = st
				}
			}
		}
	}

	// 规则匹配：按 sceneType 匹配
	matchedRules := s.matchRulesByScene(rules, sceneType)
	if len(matchedRules) == 0 {
		// 无匹配规则时仍写入 sys_recommendation 记录，确保 feedback 能关联
		rec := &model.SysRecommendation{
			UserID:        userID,
			ImageMd5:      imageMd5,
			TargetType:    "algorithm",
			TopAlgorithms: "[]",
			Feedback:      0,
		}
		if createErr := s.recRepo.Create(ctx, rec); createErr != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "创建推荐记录失败", createErr)
		}
		return []vo.RecommendedAlgorithmVO{}, nil
	}

	// 获取已发布算法作为候选池
	var publishedAlgs []model.SysAlgorithm
	err := s.db.WithContext(ctx).
		Where("status = 4 AND deleted = 0").
		Find(&publishedAlgs).Error
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法列表失败", err)
	}
	if len(publishedAlgs) == 0 {
		// 无已发布算法时仍写入 sys_recommendation 记录
		rec := &model.SysRecommendation{
			UserID:        userID,
			ImageMd5:      imageMd5,
			TargetType:    "algorithm",
			TopAlgorithms: "[]",
			Feedback:      0,
		}
		if createErr := s.recRepo.Create(ctx, rec); createErr != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "创建推荐记录失败", createErr)
		}
		return []vo.RecommendedAlgorithmVO{}, nil
	}

	// 收集候选算法ID
	candidateIDs := make(map[int64]bool)
	for _, r := range matchedRules {
		var ids []int64
		if err := json.Unmarshal([]byte(r.AlgorithmIds), &ids); err == nil {
			for _, id := range ids {
				candidateIDs[id] = true
			}
		}
	}

	// 筛选已发布算法中的候选
	var candidates []model.SysAlgorithm
	for _, alg := range publishedAlgs {
		if candidateIDs[alg.ID] {
			candidates = append(candidates, alg)
		}
	}
	if len(candidates) == 0 {
		// 无候选算法时仍写入 sys_recommendation 记录
		rec := &model.SysRecommendation{
			UserID:        userID,
			ImageMd5:      imageMd5,
			TargetType:    "algorithm",
			TopAlgorithms: "[]",
			Feedback:      0,
		}
		if createErr := s.recRepo.Create(ctx, rec); createErr != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "创建推荐记录失败", createErr)
		}
		return []vo.RecommendedAlgorithmVO{}, nil
	}

	// 计算得分并排序
	result := make([]vo.RecommendedAlgorithmVO, 0, len(candidates))
	for _, alg := range candidates {
		matchScore := s.computeMatchScore(alg.ID, matchedRules)
		result = append(result, vo.RecommendedAlgorithmVO{
			AlgorithmID:       alg.ID,
			AlgorithmName:     alg.Name,
			MatchScore:        matchScore,
			Reason:            s.buildReason(sceneType, alg.Name),
			Rating:            3.5,
			EstimatedTime:     5000,
			EffectDescription: fmt.Sprintf("该算法在%s场景下表现稳定", sceneType),
		})
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].MatchScore > result[j].MatchScore
	})

	if len(result) > topN {
		result = result[:topN]
	}

	// 写入 sys_recommendation 记录，确保 feedback 能找到记录
	topAlgorithms := make([]map[string]interface{}, 0, len(result))
	for _, vo := range result {
		topAlgorithms = append(topAlgorithms, map[string]interface{}{
			"algorithmId":   vo.AlgorithmID,
			"algorithmName": vo.AlgorithmName,
			"matchScore":    vo.MatchScore,
		})
	}
	topAlgJSON, _ := json.Marshal(topAlgorithms)

	rec := &model.SysRecommendation{
		UserID:        userID,
		ImageMd5:      imageMd5,
		TargetType:    "algorithm",
		TopAlgorithms: string(topAlgJSON),
		Feedback:      0,
	}
	if createErr := s.recRepo.Create(ctx, rec); createErr != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建推荐记录失败", createErr)
	}

	// 回填 recommendationId 到 VO
	recommendationID := rec.ID
	for i := range result {
		result[i].RecommendationId = recommendationID
	}

	return result, nil
}

func (s *RecommendationService) SubmitFeedback(ctx context.Context, form *bo.FeedbackForm) (int64, error) {
	rec, err := s.recRepo.FindByID(ctx, form.RecommendationID)
	if err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "查询推荐记录失败", err)
	}
	if rec == nil {
		return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "推荐记录不存在")
	}

	feedback := int8(2)
	if form.Useful {
		feedback = 1
	}
	if err := s.recRepo.Update(ctx, rec.ID, map[string]interface{}{"feedback": feedback}); err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "更新反馈失败", err)
	}
	return rec.ID, nil
}

func (s *RecommendationService) GetRules(ctx context.Context) ([]vo.RecommendationRuleVO, error) {
	rules, err := s.ruleRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询规则列表失败", err)
	}
	result := make([]vo.RecommendationRuleVO, 0, len(rules))
	for _, r := range rules {
		result = append(result, s.toRuleVO(&r))
	}
	return result, nil
}

func (s *RecommendationService) UpdateRule(ctx context.Context, id int64, form *bo.RuleForm) (int64, error) {
	if form.Weight == nil || *form.Weight < 0 || *form.Weight > 100 {
		return 0, common.NewBizError(common.BUSINESS_ERROR, "规则权重必须在0-100之间")
	}

	if id == 0 {
		// 新增
		algIDsJSON, _ := json.Marshal(form.AlgorithmIds)
		enabled := int8(1)
		if form.Enabled != nil && !*form.Enabled {
			enabled = 0
		}
		rule := &model.SysRecommendationRule{
			RuleName:     form.RuleName,
			SceneType:    form.SceneType,
			AlgorithmIds: string(algIDsJSON),
			Weight:       *form.Weight,
			Enabled:      enabled,
		}
		if err := s.ruleRepo.Create(ctx, rule); err != nil {
			return 0, common.WrapBizError(common.DATABASE_ERROR, "创建规则失败", err)
		}
		s.refreshRuleCache(ctx)
		return rule.ID, nil
	}

	// 更新
	rule, err := s.ruleRepo.FindByID(ctx, id)
	if err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "查询规则失败", err)
	}
	if rule == nil {
		return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "规则不存在")
	}

	algIDsJSON, _ := json.Marshal(form.AlgorithmIds)
	updates := map[string]interface{}{
		"rule_name":     form.RuleName,
		"scene_type":    form.SceneType,
		"algorithm_ids": string(algIDsJSON),
		"weight":        *form.Weight,
	}
	if form.Enabled != nil {
		if *form.Enabled {
			updates["enabled"] = int8(1)
		} else {
			updates["enabled"] = int8(0)
		}
	}
	if err := s.ruleRepo.Update(ctx, id, updates); err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "更新规则失败", err)
	}
	s.refreshRuleCache(ctx)
	return id, nil
}

func (s *RecommendationService) GetReport(ctx context.Context, startDate, endDate string) (*vo.RecommendationReportVO, error) {
	startTime, endTime := s.parseDateRange(startDate, endDate)

	total, err := s.recRepo.CountTotal(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "统计推荐总数失败", err)
	}
	usefulCount, err := s.recRepo.CountUseful(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "统计有用反馈失败", err)
	}
	feedbackTotal, err := s.recRepo.CountFeedbackTotal(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "统计反馈总数失败", err)
	}
	adoptedDistinct, err := s.recRepo.CountAdoptedAlgorithmDistinct(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "统计采纳算法数失败", err)
	}

	// 获取已发布算法总数
	var publishedCount int64
	s.db.WithContext(ctx).Model(&model.SysAlgorithm{}).Where("status = 4 AND deleted = 0").Count(&publishedCount)

	report := &vo.RecommendationReportVO{
		TotalRecommendations: total,
		ColdStartSuccessRate: 0.0,
	}
	if feedbackTotal > 0 {
		report.AdoptionRate = float64(usefulCount) / float64(feedbackTotal)
		report.SatisfactionRate = float64(usefulCount) / float64(feedbackTotal)
	}
	if publishedCount > 0 {
		report.CoverageRate = float64(adoptedDistinct) / float64(publishedCount)
	}

	// 趋势按日聚合
	dailyRows, err := s.recRepo.FindDailyAdoptionRate(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询每日趋势失败", err)
	}
	trend := make([]vo.TrendItem, 0, len(dailyRows))
	for _, row := range dailyRows {
		trend = append(trend, vo.TrendItem{
			Date:         row.Date,
			AdoptionRate: row.AdoptionRate,
		})
	}
	report.Trend = trend

	return report, nil
}

// ==================== 内部方法 ====================

func (s *RecommendationService) resolveImageURL(form *bo.AnalyzeForm) string {
	if form.ImageID != nil && *form.ImageID > 0 {
		return "" // 触发 imageId 不支持错误
	}
	if form.ImageURL != "" {
		return form.ImageURL
	}
	return ""
}

func (s *RecommendationService) validateImageFormat(imageURL string) error {
	if imageURL == "" {
		return nil
	}
	lower := strings.ToLower(imageURL)
	if idx := strings.Index(lower, "?"); idx > 0 {
		lower = lower[:idx]
	}
	for _, ext := range imageExtensions {
		if strings.HasSuffix(lower, ext) {
			return nil
		}
	}
	return common.NewBizError(common.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "不支持的文件类型，仅支持图片格式")
}

func (s *RecommendationService) getEnabledRules(ctx context.Context) []model.SysRecommendationRule {
	s.mu.RLock()
	if len(s.ruleCache) > 0 {
		rules := s.ruleCache
		s.mu.RUnlock()
		return rules
	}
	s.mu.RUnlock()
	return s.refreshRuleCache(ctx)
}

func (s *RecommendationService) refreshRuleCache(ctx context.Context) []model.SysRecommendationRule {
	rules, err := s.ruleRepo.FindEnabled(ctx)
	if err != nil || len(rules) == 0 {
		return nil
	}
	s.mu.Lock()
	s.ruleCache = rules
	s.mu.Unlock()
	return rules
}

func (s *RecommendationService) matchRulesByScene(rules []model.SysRecommendationRule, sceneType string) []model.SysRecommendationRule {
	var matched []model.SysRecommendationRule
	for _, r := range rules {
		if r.SceneType == sceneType {
			matched = append(matched, r)
		}
	}
	sort.Slice(matched, func(i, j int) bool {
		return matched[i].Weight > matched[j].Weight
	})
	return matched
}

func (s *RecommendationService) computeMatchScore(algorithmID int64, matchedRules []model.SysRecommendationRule) int {
	maxWeight := 0
	for _, r := range matchedRules {
		var ids []int64
		if err := json.Unmarshal([]byte(r.AlgorithmIds), &ids); err != nil {
			continue
		}
		for _, id := range ids {
			if id == algorithmID && r.Weight > maxWeight {
				maxWeight = r.Weight
			}
		}
	}
	if maxWeight > 100 {
		maxWeight = 100
	}
	return maxWeight
}

func (s *RecommendationService) buildReason(sceneType, algorithmName string) string {
	reason := reasonTemplates[sceneType]
	if reason == "" {
		reason = "综合表现优秀"
	}
	return algorithmName + "：" + reason
}

func (s *RecommendationService) toRuleVO(entity *model.SysRecommendationRule) vo.RecommendationRuleVO {
	var ids []int64
	json.Unmarshal([]byte(entity.AlgorithmIds), &ids)
	if ids == nil {
		ids = []int64{}
	}
	return vo.RecommendationRuleVO{
		ID:           entity.ID,
		RuleName:     entity.RuleName,
		SceneType:    entity.SceneType,
		AlgorithmIds: ids,
		Weight:       entity.Weight,
		Enabled:      entity.Enabled == 1,
	}
}

func (s *RecommendationService) parseDateRange(startDate, endDate string) (string, string) {
	startTime := ""
	endTime := ""
	if startDate != "" {
		startTime = startDate + " 00:00:00"
	}
	if endDate != "" {
		endTime = endDate + " 23:59:59"
	}
	return startTime, endTime
}

func containsString(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

func hashCode(s string) int {
	h := 0
	for _, c := range s {
		h = 31*h + int(c)
	}
	if h < 0 {
		h = -h
	}
	return h
}
