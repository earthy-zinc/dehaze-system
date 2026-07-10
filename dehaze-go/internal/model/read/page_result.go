package read

// PageResult 分页结果读模型
type PageResult[T any] struct {
	List     []T  `json:"list"`
	Total    int64 `json:"total"`
	PageNum  int   `json:"pageNum"`
	PageSize int   `json:"pageSize"`
}
