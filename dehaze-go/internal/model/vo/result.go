package vo

// Result 通用返回结果
type Result struct {
	// 状态码
	Code int `json:"code"`
	// 返回消息
	Message string `json:"message"`
	// 数据
	Data interface{} `json:"data,omitempty"`
	// 总数（用于分页）
	Total *int64 `json:"total,omitempty"`
	// 当前页码（用于分页）
	PageNum *int `json:"pageNum,omitempty"`
	// 每页条数（用于分页）
	PageSize *int `json:"pageSize,omitempty"`
}

// Success 返回成功结果
func Success(data interface{}) Result {
	return Result{
		Code:    200,
		Message: "操作成功",
		Data:    data,
	}
}

// SuccessWithMessage 返回带消息的成功结果
func SuccessWithMessage(message string) Result {
	return Result{
		Code:    200,
		Message: message,
	}
}

// SuccessWithDetailed 返回详细的成功结果
func SuccessWithDetailed(data interface{}, message string) Result {
	return Result{
		Code:    200,
		Message: message,
		Data:    data,
	}
}

// Fail 返回失败结果
func Fail() Result {
	return Result{
		Code:    500,
		Message: "操作失败",
	}
}

// FailWithMessage 返回带消息的失败结果
func FailWithMessage(message string) Result {
	return Result{
		Code:    500,
		Message: message,
	}
}
