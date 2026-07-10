package common

import "errors"

// BizError 业务错误类型，封装错误码和原始错误
type BizError struct {
	code    *ResultCode
	message string
	cause   error
}

// Error 实现 error 接口
func (e *BizError) Error() string {
	if e.cause != nil {
		return e.message + ": " + e.cause.Error()
	}
	return e.message
}

// Unwrap 实现错误解包
func (e *BizError) Unwrap() error {
	return e.cause
}

// Code 返回错误码
func (e *BizError) Code() *ResultCode {
	return e.code
}

// Message 返回错误消息
func (e *BizError) Message() string {
	return e.message
}

// NewBizError 创建业务错误
func NewBizError(code *ResultCode, message string) *BizError {
	return &BizError{
		code:    code,
		message: message,
	}
}

// WrapBizError 包装原始错误
func WrapBizError(code *ResultCode, message string, cause error) *BizError {
	return &BizError{
		code:    code,
		message: message,
		cause:   cause,
	}
}

// AsBizError 尝试转换为业务错误
func AsBizError(err error) (*BizError, bool) {
	var bizErr *BizError
	if errors.As(err, &bizErr) {
		return bizErr, true
	}
	return nil, false
}
