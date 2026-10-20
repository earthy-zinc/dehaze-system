package message

import "strings"

// EscapeLike 转义 SQL LIKE 通配符（\ % _），与 Python escape_like、Java escapeLike 口径一致，
// 保证用户输入中的通配符按字面匹配而非模糊模式。
func EscapeLike(keyword string) string {
	if keyword == "" {
		return keyword
	}
	replacer := strings.NewReplacer(
		`\`, `\\`,
		`%`, `\%`,
		`_`, `\_`,
	)
	return replacer.Replace(keyword)
}
