package query

type MyOrderQuery struct {
	PageNum  int    `json:"pageNum"`
	PageSize int    `json:"pageSize"`
	Status   string `json:"status"`
}

type OrderPageQuery struct {
	PageNum      int    `json:"pageNum"`
	PageSize     int    `json:"pageSize"`
	OrderNo      string `json:"orderNo"`
	Keywords     string `json:"keywords"`
	Status       string `json:"status"`
	PayMethod    string `json:"payMethod"`
	AmountMin    *int64 `json:"amountMin"`
	AmountMax    *int64 `json:"amountMax"`
	PaidTimeStart string `json:"paidTimeStart"`
	PaidTimeEnd   string `json:"paidTimeEnd"`
}

type RefundPageQuery struct {
	PageNum       int    `json:"pageNum"`
	PageSize      int    `json:"pageSize"`
	OrderNo       string `json:"orderNo"`
	Keywords      string `json:"keywords"`
	Status        string `json:"status"`
	ApplyTimeStart string `json:"applyTimeStart"`
	ApplyTimeEnd   string `json:"applyTimeEnd"`
}

type OrderStatsQuery struct {
	StartTime string `json:"startTime"`
	EndTime   string `json:"endTime"`
}
