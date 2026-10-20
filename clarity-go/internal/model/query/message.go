package query

// MessageQuery 消息列表查询
type MessageQuery struct {
	Type       string `json:"type"`
	ReadStatus *int   `json:"readStatus"`
	PageNum    int    `json:"pageNum"`
	PageSize   int    `json:"pageSize"`
}

// MessageSearchQuery 消息搜索查询
type MessageSearchQuery struct {
	Keyword  string `json:"keyword"`
	PageNum  int    `json:"pageNum"`
	PageSize int    `json:"pageSize"`
}
