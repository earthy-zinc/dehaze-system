package validator

import (
	"reflect"
	"strings"

	"github.com/gin-gonic/gin/binding"
	"github.com/go-playground/locales/zh"
	ut "github.com/go-playground/universal-translator"
	"github.com/go-playground/validator/v10"
	zhTranslations "github.com/go-playground/validator/v10/translations/zh"
)

// Trans 中文翻译器实例
var Trans ut.Translator

// Init 初始化 validator 中文翻译器，应在 Gin 引擎创建后调用
func Init() {
	v, ok := binding.Validator.Engine().(*validator.Validate)
	if !ok {
		return
	}

	// 使用 json tag 名称作为字段名，使前端能直接识别
	v.RegisterTagNameFunc(func(fld reflect.StructField) string {
		name := strings.SplitN(fld.Tag.Get("json"), ",", 2)[0]
		if name == "-" {
			return ""
		}
		return name
	})

	// 注册中文翻译器
	zhLocale := zh.New()
	uni := ut.New(zhLocale, zhLocale)
	Trans, _ = uni.GetTranslator("zh")
	_ = zhTranslations.RegisterDefaultTranslations(v, Trans)

	// 注册自定义 XSS 校验器，拒绝包含 HTML 标签（如 <script、<img）和 javascript: 协议的输入
	// 允许裸字符 <>&"'，仅拦截 < 后跟字母的标签起始模式
	_ = v.RegisterValidation("no_xss", func(fl validator.FieldLevel) bool {
		s := fl.Field().String()
		lowerS := strings.ToLower(s)
		if strings.Contains(lowerS, "javascript:") {
			return false
		}
		// 拒绝 HTML 标签起始模式：< 后跟字母（如 <script、<img、<svg）
		for i := 0; i < len(s)-1; i++ {
			if s[i] == '<' && ((s[i+1] >= 'a' && s[i+1] <= 'z') || (s[i+1] >= 'A' && s[i+1] <= 'Z')) {
				return false
			}
		}
		return true
	})

	// 注册 no_xss 的中文翻译
	_ = v.RegisterTranslation("no_xss", Trans, func(ut ut.Translator) error {
		return ut.Add("no_xss", "{0}不能包含 HTML 标签或 javascript: 脚本", true)
	}, func(ut ut.Translator, fe validator.FieldError) string {
		t, _ := ut.T("no_xss", fe.Field())
		return t
	})
}

// TranslateValidationErrors 将 validator.ValidationErrors 翻译为中文消息
func TranslateValidationErrors(err error) string {
	errs, ok := err.(validator.ValidationErrors)
	if !ok {
		return err.Error()
	}

	msgs := make([]string, 0, len(errs))
	for _, e := range errs {
		msgs = append(msgs, e.Translate(Trans))
	}
	return strings.Join(msgs, "; ")
}
