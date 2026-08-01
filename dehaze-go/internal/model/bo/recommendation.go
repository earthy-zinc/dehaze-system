package bo

// AnalyzeForm 图像特征分析表单
type AnalyzeForm struct {
	ImageID  *int64 `json:"imageId"`
	ImageURL string `json:"imageUrl"`
}

// FeedbackForm 推荐反馈表单
type FeedbackForm struct {
	RecommendationID int64 `json:"recommendationId"`
	Useful           bool  `json:"useful"`
}

// RuleForm 推荐规则表单
type RuleForm struct {
	RuleName     string  `json:"ruleName"`
	SceneType    string  `json:"sceneType"`
	AlgorithmIds []int64 `json:"algorithmIds"`
	Weight       *int    `json:"weight"`
	Enabled      *bool   `json:"enabled"`
}
