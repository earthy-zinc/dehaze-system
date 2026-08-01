package bo

// AlgorithmCompareForm 算法对比表单
type AlgorithmCompareForm struct {
	AlgorithmIDs []int64 `json:"algorithmIds" binding:"required,min=1,max=3"`
	ImageURL     string  `json:"imageUrl" binding:"required"`
}
