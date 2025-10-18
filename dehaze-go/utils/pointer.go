package utils

// StringPtr 返回字符串的指针
func StringPtr(s string) *string {
	return &s
}

// StringVal 返回字符串指针的值，如果为nil则返回空字符串
func StringVal(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// IntPtr 返回int的指针
func IntPtr(i int) *int {
	return &i
}

// IntVal 返回int指针的值，如果为nil则返回0
func IntVal(i *int) int {
	if i == nil {
		return 0
	}
	return *i
}

// Int64Ptr 返回int64的指针
func Int64Ptr(i int64) *int64 {
	return &i
}

// Int64Val 返回int64指针的值，如果为nil则返回0
func Int64Val(i *int64) int64 {
	if i == nil {
		return 0
	}
	return *i
}

// BoolPtr 返回bool的指针
func BoolPtr(b bool) *bool {
	return &b
}

// BoolVal 返回bool指针的值，如果为nil则返回false
func BoolVal(b *bool) bool {
	if b == nil {
		return false
	}
	return *b
}
