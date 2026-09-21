package vo

// AlgorithmRecommendItemVO 推荐项（字段与 python RecommendItemVO 逐项对应）
type AlgorithmRecommendItemVO struct {
	AlgorithmID   int64  `json:"algorithmId"`
	AlgorithmName string `json:"algorithmName"`
	MatchScore    int    `json:"matchScore"`
	Reason        string `json:"reason"`
	// EstimatedTime 固定为 null（python 该字段尚未实现，当前恒为 None），保留契约位以对齐响应形状
	EstimatedTime *string `json:"estimatedTime"`
}

// AlgorithmRecommendResultVO 推荐结果（对齐 python RecommendResultVO）
type AlgorithmRecommendResultVO struct {
	Total int                        `json:"total"`
	Items []AlgorithmRecommendItemVO `json:"items"`
}
