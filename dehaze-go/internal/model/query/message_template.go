package query

// MessageTemplateQuery 消息模板分页查询
type MessageTemplateQuery struct {
	Name     string `json:"name"`
	Type     string `json:"type"`
	Status   int    `json:"status"`
	PageNum  int    `json:"pageNum"`
	PageSize int    `json:"pageSize"`
}
