package query

// FavoritePageQuery 收藏分页查询参数
type FavoritePageQuery struct {
	PageNum    int    `json:"pageNum"`
	PageSize   int    `json:"pageSize"`
	TargetType string `json:"targetType"`
	Keywords   string `json:"keywords"`
	SortBy     string `json:"sortBy"`
	SortOrder  string `json:"sortOrder"`
}
