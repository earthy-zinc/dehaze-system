package bo

// AlgorithmFormBO 算法表单业务对象
type AlgorithmFormBO struct {
	ID          int64  `json:"id"`
	ParentID    int64  `json:"parentId" binding:"required"`
	Type        string `json:"type"`
	Name        string `json:"name"`
	Path        string `json:"path"`
	ImportPath  string `json:"importPath"`
	Description string `json:"description"`
	Status      int    `json:"status"`
}
