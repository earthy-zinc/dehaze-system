package vo

// ColorDistribution 颜色分布
type ColorDistribution struct {
	Temperature float64 `json:"temperature"`
	Saturation  float64 `json:"saturation"`
}

// ImageFeatureAnalysisVO 图像特征分析结果
type ImageFeatureAnalysisVO struct {
	ImageMd5          string            `json:"imageMd5"`
	HazeLevel         string            `json:"hazeLevel"`
	HazeConfidence    float64           `json:"hazeConfidence"`
	SceneType         string            `json:"sceneType"`
	SceneConfidence   float64           `json:"sceneConfidence"`
	Lighting          string            `json:"lighting"`
	Complexity        float64           `json:"complexity"`
	ColorDistribution ColorDistribution `json:"colorDistribution"`
	Resolution        string            `json:"resolution"`
	NoiseLevel        string            `json:"noiseLevel"`
}

// RecommendedModelVO 推荐图像模型项
type RecommendedModelVO struct {
	RecommendationId  int64    `json:"recommendationId"`
	ModelID           string   `json:"modelId"`
	ModelName         string   `json:"modelName"`
	MatchScore        int      `json:"matchScore"`
	Reason            string   `json:"reason"`
	Rating            *float64 `json:"rating"`
	EstimatedTime     *int     `json:"estimatedTime"`
	EffectDescription string   `json:"effectDescription"`
}

// RecommendationRuleVO 推荐规则
type RecommendationRuleVO struct {
	ID        int64    `json:"id"`
	RuleName  string   `json:"ruleName"`
	SceneType string   `json:"sceneType"`
	ModelIds  []string `json:"modelIds"`
	Weight    int      `json:"weight"`
	Enabled   bool     `json:"enabled"`
}

// TrendItem 趋势数据项
type TrendItem struct {
	Date         string  `json:"date"`
	AdoptionRate float64 `json:"adoptionRate"`
}

// RecommendationReportVO 推荐效果报表
type RecommendationReportVO struct {
	TotalRecommendations int64       `json:"totalRecommendations"`
	AdoptionRate         float64     `json:"adoptionRate"`
	SatisfactionRate     float64     `json:"satisfactionRate"`
	CoverageRate         float64     `json:"coverageRate"`
	ColdStartSuccessRate float64     `json:"coldStartSuccessRate"`
	Trend                []TrendItem `json:"trend"`
}
