package query

type PackagePageQuery struct {
	PageNum   int    `json:"pageNum"`
	PageSize  int    `json:"pageSize"`
	Name      string `json:"name"`
	LevelCode string `json:"levelCode"`
	Period    string `json:"period"`
	Status    *int   `json:"status"`
	StartTime string `json:"startTime"`
	EndTime   string `json:"endTime"`
}

type CouponPageQuery struct {
	PageNum  int    `json:"pageNum"`
	PageSize int    `json:"pageSize"`
	Name     string `json:"name"`
	Type     string `json:"type"`
	Status   *int   `json:"status"`
}
