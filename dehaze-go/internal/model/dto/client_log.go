package dto

// ClientLogEntry 前端单条日志条目（字段规范见 07-日志架构设计.md §3.5）。
type ClientLogEntry struct {
	Timestamp      string   `json:"timestamp"`
	Level          string   `json:"level"`
	Message        string   `json:"message"`
	App            string   `json:"app"`
	AppVersion     string   `json:"app_version"`
	URL            string   `json:"url"`
	UserAgent      string   `json:"user_agent"`
	TraceID        string   `json:"trace_id"`
	ErrorType      string   `json:"error_type"`
	ErrorSource    string   `json:"error_source"`
	ErrorStack     string   `json:"error_stack"`
	Method         string   `json:"method"`
	Path           string   `json:"path"`
	Status         *int     `json:"status"`
	Duration       *float64 `json:"duration"`
	Code           string   `json:"code"`
	Type           string   `json:"type"`
	MetricName     string   `json:"metric_name"`
	MetricValue    *float64 `json:"metric_value"`
	NavigationType string   `json:"navigation_type"`
	ResourceURL    string   `json:"resource_url"`
}

// ClientLogBatch 前端日志批量上报请求体（单次最多 50 条）。
type ClientLogBatch struct {
	Logs []ClientLogEntry `json:"logs" binding:"required,min=1,max=50"`
}
