package query

type MemberPageQuery struct {
	PageNum        int    `json:"pageNum"`
	PageSize       int    `json:"pageSize"`
	Keywords       string `json:"keywords"`
	LevelCode      string `json:"levelCode"`
	Status         *int   `json:"status"`
	ExpireTimeStart string `json:"expireTimeStart"`
	ExpireTimeEnd   string `json:"expireTimeEnd"`
	GrowthMin      *int64 `json:"growthMin"`
	GrowthMax      *int64 `json:"growthMax"`
}

type GrowthLogQuery struct {
	PageNum    int    `json:"pageNum"`
	PageSize   int    `json:"pageSize"`
	ChangeType string `json:"changeType"`
	StartTime  string `json:"startTime"`
	EndTime    string `json:"endTime"`
}
