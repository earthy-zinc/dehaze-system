package query

// DatasetQuery 数据集分页查询。
// 分页字段不参与 gin 绑定：绑定层会把非数字兜底成 B0001、越界值无从表达 A0400，
// 一律由 handler 用 parsePaginationWithSize 校验后回填，对齐 python
// `pageNum: Query(default=1, ge=1)` / `pageSize: Query(default=10, ge=1, le=100)`。
type DatasetQuery struct {
	Keywords string `form:"keyword" json:"keyword"`
	Type     string `form:"type" json:"type"`
	Status   *int   `form:"status" json:"status"`
	PageNum  int    `form:"-" json:"pageNum"`
	PageSize int    `form:"-" json:"pageSize"`
}
