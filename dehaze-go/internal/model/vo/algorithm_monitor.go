package vo

// AlgorithmMonitorVO 算法监控数据视图对象
type AlgorithmMonitorVO struct {
	// 调用次数
	CallCount int64 `json:"callCount"`
	// 平均处理时间
	AvgTime float64 `json:"avgTime"`
	// 成功率
	SuccessRate float64 `json:"successRate"`
	// 今日调用次数
	TodayCallCount int64 `json:"todayCallCount"`
}
