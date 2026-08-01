package vo

// AlgorithmSelectVO 算法选择模块 - 算法视图对象（含评分/使用次数等扩展信息）
type AlgorithmSelectVO struct {
	ID          int64               `json:"id"`
	ParentID    int64               `json:"parentId"`
	Name        string              `json:"name"`
	Type        string              `json:"type"`
	Img         string              `json:"img"`
	Description string              `json:"description"`
	Path        string              `json:"path"`
	Flops       string              `json:"flops"`
	Params      string              `json:"params"`
	ImportPath  string              `json:"importPath"`
	Status      int                 `json:"status"`
	Size        string              `json:"size"`
	Rating      float64             `json:"rating"`      // 平均评分
	UsageCount  int64               `json:"usageCount"`  // 使用次数（pred_log 统计）
	Children    []AlgorithmSelectVO `json:"children"`
}

// AlgorithmDetailVO 算法详情视图对象（含样例效果图/评分/使用次数）
type AlgorithmDetailVO struct {
	AlgorithmSelectVO
	SampleImages []AlgorithmSampleVO `json:"sampleImages"` // 样例效果图列表
	RatingStats  *AlgorithmRatingStatsVO `json:"ratingStats"`  // 评分统计
}

// AlgorithmSampleVO 算法样例效果图
type AlgorithmSampleVO struct {
	OriginURL string `json:"originUrl"` // 原始图片URL
	PredURL   string `json:"predUrl"`   // 处理后图片URL
}

// AlgorithmRatingStatsVO 算法评分统计
type AlgorithmRatingStatsVO struct {
	Average      float64        `json:"average"`      // 平均分
	Count        int64          `json:"count"`        // 评价总数
	Distribution map[int8]int64 `json:"distribution"` // 各星级分布 {1: count, 2: count, ...}
}

// AlgorithmCompareVO 算法对比结果
type AlgorithmCompareVO struct {
	ID              int64   `json:"id"`
	Name            string  `json:"name"`
	Type            string  `json:"type"`
	Description     string  `json:"description"`
	Flops           string  `json:"flops"`
	Params          string  `json:"params"`
	Rating          float64 `json:"rating"`
	UsageCount      int64   `json:"usageCount"`
	AvgTime         float64 `json:"avgTime"`         // 平均处理时间（秒）
	SuccessRate     float64 `json:"successRate"`     // 成功率（百分比）
	PredResultURL   string  `json:"predResultUrl"`   // 对比预测结果URL
	PredTime        int     `json:"predTime"`        // 本次对比预测耗时
}
