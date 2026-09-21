package algorithm_select

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// IAlgorithmSelectService 算法选择服务接口
type IAlgorithmSelectService interface {
	// GetTree 获取算法选择树（仅已发布状态）
	GetTree(ctx context.Context) ([]vo.AlgorithmSelectNodeVO, error)

	// GetDetail 获取算法详情（含样例效果图/评分/使用次数）
	GetDetail(ctx context.Context, id int64) (*vo.AlgorithmDetailVO, error)

	// Search 搜索算法（关键词/拼音/标签）
	Search(ctx context.Context, keyword string) ([]vo.AlgorithmSelectVO, error)

	// Test 上传图片测试算法效果
	Test(ctx context.Context, algorithmID int64, imageURL string, userID int64) (int64, int, error)

	// Compare 算法对比（最多3个）
	Compare(ctx context.Context, form *bo.AlgorithmCompareForm, userID int64) ([]vo.AlgorithmCompareVO, error)

	// Recommend 算法推荐匹配（F-M03-007：关键词/任务类型/样例算法 → Top N）
	Recommend(ctx context.Context, form *bo.AlgorithmRecommendForm) (*vo.AlgorithmRecommendResultVO, error)
}
