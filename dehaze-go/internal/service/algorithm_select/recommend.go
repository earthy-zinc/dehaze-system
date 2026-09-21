package algorithm_select

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// Recommend 算法推荐匹配（逐条对齐 python `algorithm_select_service.recommend`，F-M03-007）。
//
// 评分规则（基于已发布算法）：关键词命中 名称+60 / 描述+30 / 类型+20；
// 与样例算法同类型 +40、同父分类 +30；taskType 匹配 +10。
// 排序取前 topN（score<=0 的剔除），matchScore 上限 100；keyword/taskType/样例算法全空时返回空结果
// （对齐 python：不因空条件返回全量）。
func (s *AlgorithmSelectService) Recommend(ctx context.Context, form *bo.AlgorithmRecommendForm) (*vo.AlgorithmRecommendResultVO, error) {
	topN := 3
	if form.TopN != nil {
		topN = *form.TopN
	}
	if topN < 1 || topN > 10 {
		return nil, common.NewBizError(common.PARAM_ERROR, "topN 超出 1-10 范围")
	}

	var sample *model.SysAlgorithm
	if form.SampleAlgorithmID != nil && *form.SampleAlgorithmID > 0 {
		algo, err := s.algorithmRepo.FindByID(ctx, *form.SampleAlgorithmID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询样例算法失败", err)
		}
		if algo == nil {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
		}
		if algo.Status != bo.AlgorithmStatusPublished {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法未发布")
		}
		sample = algo
	}

	keyword := ""
	if form.Keyword != nil {
		keyword = strings.TrimSpace(strings.ToLower(*form.Keyword))
	}
	taskType := ""
	if form.TaskType != nil {
		taskType = strings.TrimSpace(*form.TaskType)
	}

	algorithms, err := s.algorithmRepo.FindAllPublished(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询已发布算法失败", err)
	}

	type candidate struct {
		score int
		algo  model.SysAlgorithm
	}
	candidates := make([]candidate, 0, len(algorithms))
	for _, algo := range algorithms {
		if sample != nil && algo.ID == sample.ID {
			continue
		}
		// taskType 过滤：指定时仅保留类型匹配或类型为空的算法
		if taskType != "" && algo.Type != "" && algo.Type != taskType {
			continue
		}
		score := 0
		if keyword != "" {
			if strings.Contains(strings.ToLower(algo.Name), keyword) {
				score += 60
			}
			if strings.Contains(strings.ToLower(algo.Description), keyword) {
				score += 30
			}
			if strings.Contains(strings.ToLower(algo.Type), keyword) {
				score += 20
			}
		}
		if sample != nil {
			if algo.Type != "" && algo.Type == sample.Type {
				score += 40
			}
			if algo.ParentID != 0 && algo.ParentID == sample.ParentID {
				score += 30
			}
		}
		if taskType != "" && algo.Type != "" && algo.Type == taskType {
			score += 10
		}
		candidates = append(candidates, candidate{score: score, algo: algo})
	}

	if keyword == "" && sample == nil && taskType == "" {
		return &vo.AlgorithmRecommendResultVO{Total: 0, Items: []vo.AlgorithmRecommendItemVO{}}, nil
	}

	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
	if len(candidates) > topN {
		candidates = candidates[:topN]
	}

	items := make([]vo.AlgorithmRecommendItemVO, 0, len(candidates))
	for _, c := range candidates {
		if c.score <= 0 {
			continue
		}
		reason := "基于任务类型与分类综合匹配"
		if keyword != "" {
			reason = fmt.Sprintf("算法名称/描述与关键词「%s」匹配", keyword)
		}
		if taskType != "" && c.algo.Type == taskType {
			reason = fmt.Sprintf("匹配任务类型「%s」", taskType)
		}
		if sample != nil {
			reason = fmt.Sprintf("与样例算法「%s」同类/同分类推荐", sample.Name)
		}
		score := c.score
		if score > 100 {
			score = 100
		}
		items = append(items, vo.AlgorithmRecommendItemVO{
			AlgorithmID:   c.algo.ID,
			AlgorithmName: c.algo.Name,
			MatchScore:    score,
			Reason:        reason,
		})
	}
	return &vo.AlgorithmRecommendResultVO{Total: len(items), Items: items}, nil
}
