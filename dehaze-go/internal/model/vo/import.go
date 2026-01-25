package vo

// ImportResultVO 导入结果视图对象
type ImportResultVO struct {
	// 总数
	Total int `json:"total"`
	// 成功数
	Success int `json:"success"`
	// 失败数
	Failed int `json:"failed"`
	// 失败明细
	Failures []ImportFailureVO `json:"failures"`
}

// ImportFailureVO 导入失败明细
type ImportFailureVO struct {
	// 行号
	Row int `json:"row"`
	// 用户名
	Username string `json:"username"`
	// 错误信息
	Message string `json:"message"`
}
