package query

type PromotionPageQuery struct {
	PageNum   int    `json:"pageNum"`
	PageSize  int    `json:"pageSize"`
	Name      string `json:"name"`
	Type      string `json:"type"`
	Status    *int   `json:"status"`
	StartTime string `json:"startTime"`
	EndTime   string `json:"endTime"`
}
