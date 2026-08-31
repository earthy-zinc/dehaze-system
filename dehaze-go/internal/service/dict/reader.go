package dict

import (
	"context"
	"fmt"
	"strconv"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// GetIntValue 从字典配置读取整型值（营销激励/商业容量参数字典化读取入口）。
//
// 复用 IDictService.GetByTypeCode 的下拉选项缓存（1 小时 TTL，字典更新时失效），
// 缺键/值解析失败/服务不可用时 warn 日志并回退 fallback，不抛异常阻断业务。
// 字典项约定见 config/sql/data/sys_dict.sql：字典项 Name 为键、Value 为整型值字符串。
func GetIntValue(ctx context.Context, svc IDictService, typeCode, key string, fallback int64) int64 {
	if svc == nil {
		return fallback
	}
	options, err := svc.GetByTypeCode(ctx, typeCode)
	if err != nil {
		logger.Warn("读取字典整型值失败，回退默认值",
			zap.String("type", typeCode), zap.String("key", key), zap.Error(err))
		return fallback
	}
	for _, opt := range options {
		if opt.Label != key {
			continue
		}
		n, err := strconv.ParseInt(fmt.Sprint(opt.Value), 10, 64)
		if err != nil {
			logger.Warn("字典整型值解析失败，回退默认值",
				zap.String("type", typeCode), zap.String("key", key),
				zap.Any("value", opt.Value), zap.Error(err))
			return fallback
		}
		return n
	}
	logger.Warn("字典配置缺键，回退默认值",
		zap.String("type", typeCode), zap.String("key", key))
	return fallback
}
