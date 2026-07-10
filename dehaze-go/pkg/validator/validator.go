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
