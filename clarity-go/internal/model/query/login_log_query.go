package query

// LoginLogQuery 登录日志分页查询条件
type LoginLogQuery struct {
	Username   string `json:"username"`
	IP         string `json:"ip"`
	Status     *int   `json:"status"`
	DeviceType string `json:"deviceType"`
	StartTime  string `json:"startTime"`
	EndTime    string `json:"endTime"`
	// 可见用户范围（普通用户仅本人日志时由 API 层限定为 [当前用户ID]，管理员为空表示全量）
	UserIDs []int64 `json:"userIds"`
	// 页码
	PageNum int `json:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize"`
}
