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
	ID int64 `json:"id"`
	// 更新与新增共用：body 携带 id 即为更新（python PUT /recommendations/rules 同口径）
	RuleName     string  `json:"ruleName" binding:"required"`
	SceneType    string  `json:"sceneType" binding:"required"`
	AlgorithmIds []int64 `json:"algorithmIds" binding:"required,min=1"`
	Weight       *int    `json:"weight" binding:"required,min=0,max=100"`
	Enabled      *bool   `json:"enabled"`
}
