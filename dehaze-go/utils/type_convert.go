package utils

import (
	"strings"
	"unicode"
)

// 类型转换
func InterfaceToInt(v any) (i int) {
	switch v := v.(type) {
	case int:
		i = v
	case int8:
		i = int(v)
	case int16:
		i = int(v)
	case int32:
		i = int(v)
	case int64:
		i = int(v)
	case uint:
		i = int(v)
	case uint8:
		i = int(v)
	case uint16:
		i = int(v)
	case uint32:
		i = int(v)
	case uint64:
		i = int(v)
	default:
		i = 0
	}
	return
}

// ToCamelCase 将字符串转换为驼峰命名（首字母大写）
// 例如: "user-management" -> "UserManagement"
//
//	"hello-world" -> "HelloWorld"
func ToCamelCase(s string) string {
	if s == "" {
		return ""
	}

	// 分割字符串（按连字符、下划线、空格）
	words := strings.FieldsFunc(s, func(r rune) bool {
		return r == '-' || r == '_' || unicode.IsSpace(r)
	})

	// 将每个单词首字母大写
	for i, word := range words {
		if len(word) > 0 {
			// 转换首字母为大写，其余保持原样
			runes := []rune(word)
			runes[0] = unicode.ToUpper(runes[0])
			words[i] = string(runes)
		}
	}

	return strings.Join(words, "")
}
