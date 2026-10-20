package vo

// LoginLogVO 登录日志列表项（字段与 Python 端 _format_log_item 对齐）
type LoginLogVO struct {
	ID         string `json:"id"`
	UserID     *int64 `json:"userId"`
	Username   string `json:"username"`
	IP         string `json:"ip"`
	Location   string `json:"location"`
	Browser    string `json:"browser"`
	OS         string `json:"os"`
	DeviceType string `json:"deviceType"`
	Status     int    `json:"status"`
	Message    string `json:"message"`
	LoginTime  string `json:"loginTime"`
}
