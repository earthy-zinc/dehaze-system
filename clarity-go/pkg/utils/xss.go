package utils

// XSS 安全防护工具
//
// 实现策略：
// - 简单场景（纯文本）：使用标准库 html.EscapeString 进行 HTML 实体编码
// - 富文本场景：使用 bluemonday 进行专业的 HTML 解析和清理
//
// bluemonday 基于 golang.org/x/net/html 进行真正的 HTML 解析，相比正则表达式：
// - 能正确处理嵌套、畸形 HTML
// - 使用白名单机制，默认拒绝所有，更安全
// - 支持 CSS 清理（通过 gorilla/css）
//
// 安全最佳实践：
// - 始终在输出时进行转义（Output Encoding），而不是仅在输入时
// - 使用 Content Security Policy (CSP) 作为额外的防护层
// - 根据上下文选择合适的清理方法

import (
	"bytes"
	"html"
	"regexp"
	"strings"

	"github.com/microcosm-cc/bluemonday"
)

// XSSUtil XSS 防护工具类
// 用于防止跨站脚本攻击 (Cross-Site Scripting)
type XSSUtil struct {
	// UGCPolicy: User Generated Content 策略，允许安全的 HTML 标签
	// 适用于评论区、文章内容等富文本场景
	UGCPolicy *bluemonday.Policy
	// StrictPolicy: 严格策略，只允许极少数安全标签
	// 适用于需要更多限制的场景
	StrictPolicy *bluemonday.Policy
	// StripTagsPolicy: 移除所有 HTML 标签，只保留文本内容
	StripTagsPolicy *bluemonday.Policy
}

// NewXSSUtil 创建 XSS 防护工具实例
func NewXSSUtil() *XSSUtil {
	return &XSSUtil{
		UGCPolicy:       bluemonday.UGCPolicy(),
		StrictPolicy:    bluemonday.StrictPolicy(),
		StripTagsPolicy: bluemonday.StripTagsPolicy(),
	}
}

// DANGEROUS_TAGS 危险的 HTML 标签列表（用于检测）
var DANGEROUS_TAGS = []string{
	"script", "iframe", "object", "embed", "form", "input", "textarea",
	"button", "select", "option", "meta", "link", "style", "base",
	"applet", "param", "video", "audio", "source", "track",
}

// DANGEROUS_ATTRIBUTES 危险的 HTML 属性列表（用于检测）
var DANGEROUS_ATTRIBUTES = []string{
	"onclick", "ondblclick", "onmousedown", "onmouseup", "onmouseover",
	"onmousemove", "onmouseout", "onfocus", "onblur", "onkeypress",
	"onkeydown", "onkeyup", "onload", "onunload", "onerror",
	"onsubmit", "onreset", "onchange", "onselect", "onabort",
	"javascript:", "data:", "vbscript:",
}

// ==================== 简单场景方法（纯文本转义）====================

// SanitizeInput 转义 HTML 特殊字符
// 将 <, >, &, ", ' 等特殊字符转换为 HTML 实体
//
// 适用场景：纯文本输入，不需要保留任何 HTML 格式
//
// 参数:
//
//	input: 需要清理的用户输入字符串
//
// 返回:
//
//	转义后的安全字符串
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
// 递归处理字符串中的所有潜在危险字符，包括 Unicode 和 URL 编码
//
// 适用场景：需要额外防护的纯文本场景
//
// 参数:
//
//	input: 需要深度清理的用户输入字符串
//
// 返回:
//
//	深度转义后的安全字符串
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

// ==================== 富文本场景方法（基于 bluemonday）====================

// SanitizeHTML 清理富文本 HTML 内容
// 使用 bluemonday UGC 策略进行安全的 HTML 清理
//
// 适用场景：
// - 评论区、文章内容、用户生成的富文本
// - 需要保留部分安全的 HTML 格式（如 <p>, <b>, <a> 等）
//
// UGC 策略允许的标签包括：
// - 文本格式：<p>, <br>, <b>, <i>, <u>, <strong>, <em>, <s>, <del>
// - 标题：<h1>-<h6>
// - 列表：<ul>, <ol>, <li>
// - 链接：<a>（仅允许 http/https/mailto 协议，自动添加 rel="nofollow"）
// - 图片：<img>（仅允许 http/https 协议）
// - 引用：<blockquote>
// - 代码：<code>, <pre>
//
// 参数:
//
//	htmlContent: 可能包含 HTML 的用户输入
//
// 返回:
//
//	清理后的安全 HTML
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	input := `<p onclick="alert('XSS')">Hello</p><script>evil()</script>`
//	safe := xss.SanitizeHTML(input)
//	// safe = "<p>Hello</p>"
func (u *XSSUtil) SanitizeHTML(htmlContent string) string {
	if htmlContent == "" {
		return ""
	}
	return u.UGCPolicy.Sanitize(htmlContent)
}

// SanitizeHTMLStrict 严格清理富文本 HTML 内容
// 使用 bluemonday Strict 策略，只允许极少数安全标签
//
// 适用场景：
// - 需要更多限制的富文本场景
// - 只允许基本格式，不允许图片、链接等
//
// Strict 策略允许的标签：
// - 文本格式：<b>, <i>, <u>, <strong>, <em>
//
// 参数:
//
//	htmlContent: 可能包含 HTML 的用户输入
//
// 返回:
//
//	严格清理后的安全 HTML
func (u *XSSUtil) SanitizeHTMLStrict(htmlContent string) string {
	if htmlContent == "" {
		return ""
	}
	return u.StrictPolicy.Sanitize(htmlContent)
}

// StripAllHTML 移除所有 HTML 标签，只保留纯文本
//
// 适用场景：
// - 需要提取纯文本内容
// - 不需要保留任何 HTML 格式
//
// 参数:
//
//	htmlContent: 可能包含 HTML 的字符串
//
// 返回:
//
//	移除所有标签后的纯文本
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	input := `<p>Hello <b>World</b></p>`
//	safe := xss.StripAllHTML(input)
//	// safe = "Hello World"
func (u *XSSUtil) StripAllHTML(htmlContent string) string {
	if htmlContent == "" {
		return ""
	}
	return u.StripTagsPolicy.Sanitize(htmlContent)
}

// StripXSS 综合清理函数
// 使用 bluemonday 进行安全的 HTML 清理，然后转义剩余的特殊字符
//
// 适用场景：需要最严格防护的场景，不保留任何 HTML 格式
//
// 清理步骤:
// 1. 使用 bluemonday StripTagsPolicy 移除所有 HTML 标签
// 2. 转义剩余的 HTML 特殊字符
//
// 参数:
//
//	input: 需要清理的用户输入
//
// 返回:
//
//	清理后的安全字符串
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	input := `<div onclick="alert('XSS')"><script>evil()</script>Hello</div>`
//	safe := xss.StripXSS(input)
//	// safe = "Hello"（或转义后的纯文本）
func (u *XSSUtil) StripXSS(input string) string {
	if input == "" {
		return ""
	}

	// 步骤1: 使用 bluemonday 移除所有 HTML 标签
	result := u.StripTagsPolicy.Sanitize(input)

	// 步骤2: 转义剩余的 HTML 特殊字符
	result = html.EscapeString(result)

	return result
}

// ==================== URL 清理方法 ====================

// SanitizeURL 清理 URL 中的危险字符
// 防止通过 URL 进行 XSS 攻击
//
// 参数:
//
//	url: 需要清理的 URL
//
// 返回:
//
//	清理后的安全 URL
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

// ==================== 检测方法 ====================

// ValidateScriptContent 检查字符串中是否包含脚本代码
//
// 参数:
//
//	content: 需要检查的内容
//
// 返回:
//
//	bool: 如果包含脚本代码返回 true，否则返回 false
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
//
//	jsonStr: JSON 字符串
//
// 返回:
//
//	在 HTML 中安全使用的 JSON 字符串
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
//
//	input: 需要检测的输入
//
// 返回:
//
//	bool: 如果可能是 XSS 攻击返回 true，否则返回 false
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

// ==================== 自定义策略 ====================

// SanitizeWithPolicy 使用自定义策略清理 HTML
//
// 适用场景：需要自定义允许的标签和属性
//
// 参数:
//
//	htmlContent: 可能包含 HTML 的字符串
//	policy: bluemonday 策略实例
//
// 返回:
//
//	按自定义策略清理后的安全字符串
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	p := bluemonday.NewPolicy().
//	    AllowElements("p", "br").
//	    AllowAttrs("class").OnElements("p")
//	safe := xss.SanitizeWithPolicy(input, p)
func (u *XSSUtil) SanitizeWithPolicy(htmlContent string, policy *bluemonday.Policy) string {
	if htmlContent == "" || policy == nil {
		return ""
	}
	return policy.Sanitize(htmlContent)
}

// CreateCustomPolicy 创建自定义清理策略
//
// 返回:
//
//	一个新的 bluemonday 策略构建器，可链式调用配置
//
// 使用示例:
//
//	xss := NewXSSUtil()
//	p := xss.CreateCustomPolicy().
//	    AllowElements("p", "b", "i", "u").
//	    AllowAttrs("class").OnElements("p")
//	safe := p.Sanitize(input)
func (u *XSSUtil) CreateCustomPolicy() *bluemonday.Policy {
	return bluemonday.NewPolicy()
}
