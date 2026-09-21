package bo

// AlgorithmRecommendForm 算法推荐匹配表单（对齐 python RecommendRequest，F-M03-007）。
// topN 的 1-10 边界由 service 校验（python 用 pydantic `ge=1, le=10` → 400+A0400）。
type AlgorithmRecommendForm struct {
	Keyword           *string `json:"keyword"`
	TaskType          *string `json:"taskType"`
	SampleAlgorithmID *int64  `json:"sampleAlgorithmId"`
	TopN              *int    `json:"topN"`
}
