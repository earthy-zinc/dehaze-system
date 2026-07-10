package bo

// AlgorithmFormBO 算法表单业务对象
type AlgorithmFormBO struct {
	ID          int64  `json:"id"`
	ParentID    int64  `json:"parentId" binding:"required"`
	Type        string `json:"type" binding:"omitempty,max=32"`
	Name        string `json:"name" binding:"required,max=128"`
	Path        string `json:"path" binding:"omitempty,max=255"`
	ImportPath  string `json:"importPath" binding:"omitempty,max=255"`
	Description string `json:"description" binding:"omitempty,max=255"`
	Status      int    `json:"status" binding:"oneof=0 1"`
}
