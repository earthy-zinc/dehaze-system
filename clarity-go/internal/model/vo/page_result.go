package vo

// PageResult 分页结果对象
type PageResult[T any] struct {
	// 数据列表
	List []T `json:"list"`
	// 总记录数
	Total int64 `json:"total"`
}

// CursorResult 游标分页结果对象（before 游标翻页）：
// HasMore 表示是否仍存在比本页最后一条更早的数据（供前端继续按 before 拉取）。
type CursorResult[T any] struct {
	// 数据列表
	List []T `json:"list"`
	// 总记录数
	Total int64 `json:"total"`
	// 是否还有更早的历史数据
	HasMore bool `json:"hasMore"`
}
