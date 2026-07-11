package query

type DatasetQuery struct {
	Keywords string `form:"keyword" json:"keyword"`
	Type     string `form:"type" json:"type"`
	Status   *int   `form:"status" json:"status"`
	PageNum  int    `form:"pageNum" json:"pageNum"`
	PageSize int    `form:"pageSize" json:"pageSize"`
}
