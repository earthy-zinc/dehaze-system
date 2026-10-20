package query

// AnnouncementQuery 公告分页查询
type AnnouncementQuery struct {
	Title    string `json:"title"`
	Type     string `json:"type"`
	Status   int    `json:"status"`
	PageNum  int    `json:"pageNum"`
	PageSize int    `json:"pageSize"`
}
