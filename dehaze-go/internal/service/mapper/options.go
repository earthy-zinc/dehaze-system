package mapper

import (
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// OptionsFromRead 将读模型选项转换为 VO 选项
func OptionsFromRead(options []read.Option) []vo.Option {
	if len(options) == 0 {
		return []vo.Option{}
	}

	result := make([]vo.Option, 0, len(options))
	for _, item := range options {
		result = append(result, vo.Option{
			Value:    item.Value,
			Label:    item.Label,
			Children: OptionsFromRead(item.Children),
		})
	}
	return result
}
