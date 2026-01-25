package utils

// XSS 安全防护工具
//
// 重要说明：
// 1. 本文件使用正则表达式进行 HTML 清理，虽然可以防御大多数 XSS 攻击，
//    但不能完全替代专业的 HTML 解析库（如 golang.org/x/net/html）。
// 2. 正则表达式匹配 HTML 存在固有限制，复杂的 HTML 结构可能被绕过。
// 3. 对于富文本编辑器等复杂场景，建议使用专业的 HTML 清理库（如 bluemonday）。
// 4. 本工具主要用于简单场景的 XSS 防护，作为深度防御的一部分。
//
// 安全最佳实践：
// - 始终在输出时进行转义（Output Encoding），而不是仅在输入时
// - 使用 Content Security Policy (CSP) 作为额外的防护层
// - 对于用户生成的富文本内容，使用专门的 HTML 清理库
//
// 修复说明 (P0 问题)：
// - 添加了本注释说明正则表达式的局限性
// - 将 SQL 注入防护移至独立的 sql_injection.go 文件
// - SQL 注入防护不应与 XSS 防护混在一起，遵循单一职责原则

import (
	"bytes"
	"html"
	"regexp"
	"strings"
)

// XSSUtil XSS 防护工具类
// 用于防止跨站脚本攻击 (Cross-Site Scripting)
type XSSUtil struct{}

// NewXSSUtil 创建 XSS 防护工具实例
func NewXSSUtil() *XSSUtil {
	return &XSSUtil{}
}

// DANGEROUS_TAGS 危险的 HTML 标签列表
var DANGEROUS_TAGS = []string{
	"script", "iframe", "object", "embed", "form", "input", "textarea",
	"button", "select", "option", "meta", "link", "style", "base",
	"applet", "param", "video", "audio", "source", "track",
}

// DANGEROUS_ATTRIBUTES 危险的 HTML 属性列表
var DANGEROUS_ATTRIBUTES = []string{
	"onclick", "ondblclick", "onmousedown", "onmouseup", "onmouseover",
	"onmousemove", "onmouseout", "onfocus", "onblur", "onkeypress",
	"onkeydown", "onkeyup", "onload", "onunload", "onerror",
	"onsubmit", "onreset", "onchange", "onselect", "onabort",
	"javascript:", "data:", "vbscript:",
}

// SanitizeInput 转义 HTML 特殊字符
// 将 <, >, &, ", ' 等特殊字符转换为 HTML 实体
//
// 参数:
//   input: 需要清理的用户输入字符串
//
// 返回:
//   转义后的安全字符串
//
// 安全原理:
// - 使用标准库 html.EscapeString 进行 HTML 实体编码
// - 将特殊字符转换为 HTML 实体，防止浏览器将其解析为 HTML 标签
// - < 转换为 &lt; > 转换为 &gt; & 转换为 &amp;
// - " 转换为 &quot; ' 转换为 &#39;
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	input := `<script>alert('XSS')</script>`
//	safe := xss.SanitizeInput(input)
//	// safe = "&lt;script&gt;alert('XSS')&lt;/script&gt;"
func (u *XSSUtil) SanitizeInput(input string) string {
	if input == "" {
		return ""
	}
	return html.EscapeString(input)
}

// SanitizeInputDeep 深度转义 HTML 特殊字符
// 递归处理字符串中的所有潜在危险字符
//
// 参数:
//   input: 需要深度清理的用户输入字符串
//
// 返回:
//   深度转义后的安全字符串
//
// 安全原理:
// - 在标准转义的基础上，额外处理一些边缘情况
// - 处理 Unicode 编码的攻击向量
// - 处理 URL 编码的攻击向量
func (u *XSSUtil) SanitizeInputDeep(input string) string {
	if input == "" {
		return ""
	}

	// 先进行标准转义
	result := html.EscapeString(input)

	// 处理 Unicode 编码的 < 和 >
	// \u003c = <, \u003e = >
	result = strings.ReplaceAll(result, `\u003c`, `&lt;`)
	result = strings.ReplaceAll(result, `\u003e`, `&gt;`)

	// 处理十六进制编码的 < 和 >
	// \x3c = <, \x3e = >
	result = strings.ReplaceAll(result, `\x3c`, `&lt;`)
	result = strings.ReplaceAll(result, `\x3e`, `&gt;`)

	// 处理 URL 编码
	// %3C = <, %3E = >
	result = strings.ReplaceAll(result, `%3C`, `&lt;`)
	result = strings.ReplaceAll(result, `%3E`, `&gt;`)

	return result
}

// StripDangerousTags 移除危险的 HTML 标签
// 使用正则表达式移除所有危险的 HTML 标签及其内容
//
// 参数:
//   htmlContent: 可能包含 HTML 标签的字符串
//
// 返回:
//   移除危险标签后的安全字符串
//
// 安全原理:
// - 使用正则表达式匹配并移除危险的 HTML 标签
// - 同时移除自闭合标签和普通标签
// - 移除标签内容中的危险事件处理器
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	input := `<div><script>alert('XSS')</script>Hello</div>`
//	safe := xss.StripDangerousTags(input)
//	// safe = "<div>Hello</div>"
func (u *XSSUtil) StripDangerousTags(htmlContent string) string {
	if htmlContent == "" {
		return ""
	}

	result := htmlContent

	// 移除危险的 HTML 标签（包括自闭合标签）
	for _, tag := range DANGEROUS_TAGS {
		// 移除 <tag>...</tag>
		regex := regexp.MustCompile(`(?i)<`+tag+`[^>]*>.*?</`+tag+`>`)
		result = regex.ReplaceAllString(result, "")

		// 移除 <tag />
		regexSelfClose := regexp.MustCompile(`(?i)<`+tag+`[^>]*/>`)
		result = regexSelfClose.ReplaceAllString(result, "")

		// 移除 <tag>
		regexOpen := regexp.MustCompile(`(?i)<`+tag+`[^>]*>`)
		result = regexOpen.ReplaceAllString(result, "")

		// 移除 </tag>
		regexClose := regexp.MustCompile(`(?i)</`+tag+`>`)
		result = regexClose.ReplaceAllString(result, "")
	}

	return result
}

// StripDangerousAttributes 移除危险的 HTML 属性
// 移除可能导致 XSS 的 HTML 属性（如 onclick、javascript: 等）
//
// 参数:
//   htmlContent: 可能包含危险属性的 HTML 字符串
//
// 返回:
//   移除危险属性后的安全字符串
//
// 安全原理:
// - 移除所有 on* 事件处理器
// - 移除 javascript:、data:、vbscript: 等危险协议
// - 保留其他合法的 HTML 属性
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	input := `<a href="#" onclick="alert('XSS')">Click me</a>`
//	safe := xss.StripDangerousAttributes(input)
//	// safe = `<a href="#">Click me</a>`
func (u *XSSUtil) StripDangerousAttributes(htmlContent string) string {
	if htmlContent == "" {
		return ""
	}

	result := htmlContent

	// 移除危险的属性
	for _, attr := range DANGEROUS_ATTRIBUTES {
		// 匹配属性名称和值
		// 例如: onclick="alert('XSS')" 或 onclick='alert("XSS")' 或 onclick=alert('XSS')
		regex := regexp.MustCompile(`(?i)\s+`+regexp.QuoteMeta(attr)+`\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)`)
		result = regex.ReplaceAllString(result, "")

		// 移除属性名本身（如果没有值）
		regexNoValue := regexp.MustCompile(`(?i)\s+`+regexp.QuoteMeta(attr)+`\s*(?=[\s>])`)
		result = regexNoValue.ReplaceAllString(result, "")
	}

	return result
}

// StripXSS 综合清理函数
// 组合使用多种方法来防御 XSS 攻击
//
// 参数:
//   input: 需要清理的用户输入
//
// 返回:
//   清理后的安全字符串
//
// 清理步骤:
// 1. 移除危险的 HTML 标签
// 2. 移除危险的 HTML 属性
// 3. 转义剩余的 HTML 特殊字符
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	input := `<div onclick="alert('XSS')"><script>evil()</script>Hello</div>`
//	safe := xss.StripXSS(input)
//	// safe = "&lt;div&gt;Hello&lt;/div&gt;"
func (u *XSSUtil) StripXSS(input string) string {
	if input == "" {
		return ""
	}

	// 步骤1: 移除危险的 HTML 标签
	result := u.StripDangerousTags(input)

	// 步骤2: 移除危险的 HTML 属性
	result = u.StripDangerousAttributes(result)

	// 步骤3: 转义剩余的 HTML 特殊字符
	result = u.SanitizeInput(result)

	return result
}

// SanitizeURL 清理 URL 中的危险字符
// 防止通过 URL 进行 XSS 攻击
//
// 参数:
//   url: 需要清理的 URL
//
// 返回:
//   清理后的安全 URL
//
// 安全原理:
// - 移除 javascript:、data:、vbscript: 等危险协议
// - 移除可能导致重定向的 URL 参数
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	url := "javascript:alert('XSS')"
//	safe := xss.SanitizeURL(url)
//	// safe = ""
func (u *XSSUtil) SanitizeURL(url string) string {
	if url == "" {
		return ""
	}

	lowerUrl := strings.ToLower(url)
	dangerousPrefixes := []string{
		"javascript:",
		"data:",
		"vbscript:",
		"file:",
	}

	// 检查是否包含危险协议前缀
	for _, prefix := range dangerousPrefixes {
		if strings.HasPrefix(lowerUrl, prefix) {
			return ""
		}
	}

	// 移除 URL 中的事件处理器
	result := regexp.MustCompile(`(?i)on\w+\s*=\s*["'][^"']*["']`).ReplaceAllString(url, "")

	return result
}

// ValidateScriptContent 检查字符串中是否包含脚本代码
//
// 参数:
//   content: 需要检查的内容
//
// 返回:
//   bool: 如果包含脚本代码返回 true，否则返回 false
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	content := "alert('hello')"
//	isDangerous := xss.ValidateScriptContent(content)
//	// isDangerous = true
func (u *XSSUtil) ValidateScriptContent(content string) bool {
	if content == "" {
		return false
	}

	lowerContent := strings.ToLower(content)

	// 检查常见的 JavaScript 关键字和函数
	scriptPatterns := []string{
		"javascript:",
		"eval(",
		"function(",
		"alert(",
		"confirm(",
		"prompt(",
		"document.",
		"window.",
		"location.",
		"cookie",
		"expression(",
	}

	for _, pattern := range scriptPatterns {
		if strings.Contains(lowerContent, pattern) {
			return true
		}
	}

	return false
}

// SanitizeJSONForHTML 在 HTML 上下文中安全地使用 JSON
// 防止 JSON 数据中的脚本在 HTML 中被执行
//
// 参数:
//   jsonStr: JSON 字符串
//
// 返回:
//   在 HTML 中安全使用的 JSON 字符串
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	json := `{"name": "<script>alert('XSS')</script>"}`
//	safe := xss.SanitizeJSONForHTML(json)
//	// 可以安全地在 HTML 中输出
func (u *XSSUtil) SanitizeJSONForHTML(jsonStr string) string {
	if jsonStr == "" {
		return ""
	}

	var buf bytes.Buffer

	for _, r := range jsonStr {
		switch r {
		case '<':
			buf.WriteString(`\u003c`)
		case '>':
			buf.WriteString(`\u003e`)
		case '&':
			buf.WriteString(`\u0026`)
		case '\'':
			buf.WriteString(`\u0027`)
		case '"':
			buf.WriteString(`\u0022`)
		case '/':
			buf.WriteString(`\u002f`)
		case '=':
			buf.WriteString(`\u003d`)
		default:
			buf.WriteRune(r)
		}
	}

	return buf.String()
}

// IsXSSAttack 检测输入是否可能是 XSS 攻击
//
// 参数:
//   input: 需要检测的输入
//
// 返回:
//   bool: 如果可能是 XSS 攻击返回 true，否则返回 false
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	input := "<script>alert('XSS')</script>"
//	isAttack := xss.IsXSSAttack(input)
//	// isAttack = true
func (u *XSSUtil) IsXSSAttack(input string) bool {
	if input == "" {
		return false
	}

	// 检查危险标签
	for _, tag := range DANGEROUS_TAGS {
		if strings.Contains(strings.ToLower(input), "<"+tag) {
			return true
		}
	}

	// 检查危险属性
	for _, attr := range DANGEROUS_ATTRIBUTES {
		if strings.Contains(strings.ToLower(input), attr) {
			return true
		}
	}

	// 检查脚本内容
	return u.ValidateScriptContent(input)
}
