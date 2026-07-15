package utils

import (
	"regexp"
	"strings"
)

// SQLInjectionUtil SQL 注入防护工具类
//
// 重要说明：
// 1. 本工具提供基本的 SQL 注入模式检测和清理功能
// 2. 这不是完整的 SQL 注入防护，不能替代参数化查询
// 3. 必须在所有数据库查询中使用参数化查询（Prepared Statements）
// 4. 根据 OWASP 最佳实践，参数化查询是防止 SQL 注入的首要方法
//
// 安全最佳实践：
// - 始终使用参数化查询（Prepared Statements）
// - 对数据库用户进行最小权限配置
// - 使用 ORM 框架提供的安全查询功能
// - 对输入进行验证和清理，但这只是额外防护，不能替代参数化查询

type SQLInjectionUtil struct{}

// NewSQLInjectionUtil 创建 SQL 注入防护工具实例
func NewSQLInjectionUtil() *SQLInjectionUtil {
	return &SQLInjectionUtil{}
}

// 预编译正则，避免每次调用都重新编译
var (
	unionSelectRe       = regexp.MustCompile(`(?i)\bunion\s+select\b`)
	dangerousKeywordRes = compileDangerousKeywordRegexes()
)

func compileDangerousKeywordRegexes() []*regexp.Regexp {
	keywords := []string{
		"drop table",
		"truncate table",
		"delete from",
		"insert into",
		"update set",
		"exec(",
		"execute(",
		"xp_cmdshell",
	}
	res := make([]*regexp.Regexp, len(keywords))
	for i, kw := range keywords {
		res[i] = regexp.MustCompile(`(?i)` + regexp.QuoteMeta(kw))
	}
	return res
}

// StripSQLInjectionPatterns 移除常见的 SQL 注入模式
//
// 注意：这不是完整的 SQL 注入防护，应该配合参数化查询使用
//
// 参数:
//   input: 需要检查的输入
//
// 返回:
//   移除 SQL 注入模式后的字符串
//
// 安全原理:
// - 移除常见的 SQL 注入模式
// - 移除 SQL 注释和联合查询
// - 移除危险的 SQL 关键词
// - 警告：这不能替代参数化查询，仅作为深度防御的一部分
//
// 使用示例:
//
//	sqlUtil := NewSQLInjectionUtil()
//	input := "' OR '1'='1' --"
//	safe := sqlUtil.StripSQLInjectionPatterns(input)
//	// safe = " '' OR '1'='1'  "
func (u *SQLInjectionUtil) StripSQLInjectionPatterns(input string) string {
	if input == "" {
		return ""
	}

	result := input

	// 移除 SQL 注释
	result = strings.ReplaceAll(result, "--", "")
	result = strings.ReplaceAll(result, "#", "")
	result = strings.ReplaceAll(result, "/*", "")
	result = strings.ReplaceAll(result, "*/", "")

	// 移除联合查询
	result = unionSelectRe.ReplaceAllString(result, "")

	// 移除常见的 SQL 注入关键词（使用预编译正则）
	for _, re := range dangerousKeywordRes {
		result = re.ReplaceAllString(result, "")
	}

	return result
}

// DetectSQLInjection 检测输入是否可能包含 SQL 注入
//
// 参数:
//   input: 需要检测的输入
//
// 返回:
//   bool: 如果可能包含 SQL 注入返回 true，否则返回 false
//
// 使用示例:
//
//	sqlUtil := NewSQLInjectionUtil()
//	input := "' OR '1'='1'"
//	isDangerous := sqlUtil.DetectSQLInjection(input)
//	// isDangerous = true
func (u *SQLInjectionUtil) DetectSQLInjection(input string) bool {
	if input == "" {
		return false
	}

	lowerInput := strings.ToLower(input)

	// 检查常见的 SQL 注入模式
	injectionPatterns := []string{
		"' or '",
		"' or \"",
		"\" or '",
		"\" or \"",
		"' or 1=1",
		"' or 1 = 1",
		"\" or 1=1",
		"\" or 1 = 1",
		"' union",
		"' union select",
		"\" union",
		"\" union select",
		"--",
		"#",
		"/*",
		"*/",
		"drop table",
		"truncate table",
		"delete from",
		"insert into",
		"update set",
		"exec(",
		"execute(",
		"xp_cmdshell",
	}

	for _, pattern := range injectionPatterns {
		if strings.Contains(lowerInput, pattern) {
			return true
		}
	}

	return false
}

// ValidateColumnName 验证列名是否安全（用于动态 SQL）
//
// 注意：此函数仅用于白名单验证，不能替代参数化查询
//
// 参数:
//   columnName: 列名
//   allowedColumns: 允许的列名白名单
//
// 返回:
//   bool: 列名是否在白名单中
//   error: 错误信息
//
// 安全原理:
// - 使用白名单验证列名
// - 防止通过动态列名进行 SQL 注入
// - 只允许预先定义的安全列名
//
// 使用示例:
//
//	sqlUtil := NewSQLInjectionUtil()
//	allowed := []string{"id", "name", "email", "create_time"}
//	isValid := sqlUtil.ValidateColumnName("name", allowed)
//	// isValid = true
func (u *SQLInjectionUtil) ValidateColumnName(columnName string, allowedColumns []string) (bool, error) {
	if columnName == "" {
		return false, nil
	}

	for _, allowed := range allowedColumns {
		if columnName == allowed {
			return true, nil
		}
	}

	return false, nil
}

// ValidateSortDirection 验证排序方向是否安全
//
// 参数:
//   direction: 排序方向（ASC 或 DESC）
//
// 返回:
//   bool: 排序方向是否有效
//
// 使用示例:
//
//	sqlUtil := NewSQLInjectionUtil()
//	isValid := sqlUtil.ValidateSortDirection("ASC")
//	// isValid = true
func (u *SQLInjectionUtil) ValidateSortDirection(direction string) bool {
	upperDirection := strings.ToUpper(direction)
	return upperDirection == "ASC" || upperDirection == "DESC"
}
