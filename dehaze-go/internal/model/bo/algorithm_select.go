package bo

// AlgorithmCompareForm 算法对比表单（T-AS-055：数量需在 2-3 个之间）
type AlgorithmCompareForm struct {
	AlgorithmIDs []int64 `json:"algorithmIds" binding:"required,min=2,max=3"`
	ImageURL     string  `json:"imageUrl" binding:"required"`
}
