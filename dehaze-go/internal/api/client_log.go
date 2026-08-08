package api

import (
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

const (
	maxMessageLength = 2000
	maxErrorStackLen = 8000
)

type ClientLogApi struct{}

func NewClientLogApi() *ClientLogApi {
	return &ClientLogApi{}
}

// Collect 接收前端 SDK 批量上报的日志并落盘 client.log。
//
// 匿名（未登录）仅允许上报 ERROR 且必须携带 trace_id；已登录用户从会话解析 user_id 注入。
func (a *ClientLogApi) Collect(c *gin.Context) {
	var batch dto.ClientLogBatch
	if err := c.ShouldBind(&batch); err != nil {
		_ = c.Error(err)
		return
	}

	// security.GetUserID 未登录返回 0
	userID := security.GetUserID(c)
	clientLogger := logger.GetClientLogger()
	for _, entry := range batch.Logs {
		writeEntry(&entry, userID, clientLogger)
	}

	common.Ok(c)
}

// writeEntry 处理单条日志条目：匿名过滤、字段构建、按级别落盘。
// 拆分为接收 logger 参数，便于单元测试注入 observer logger。
func writeEntry(entry *dto.ClientLogEntry, userID int64, log *zap.Logger) {
	traceID := strings.TrimSpace(entry.TraceID)
	// 匿名仅允许上报 ERROR 且必须携带 trace_id，否则丢弃该条，避免被滥用刷日志
	if userID == 0 && (!isError(entry.Level) || traceID == "") {
		return
	}

	fields := buildFields(entry, traceID)
	if userID > 0 {
		fields = append(fields, zap.Int64("user_id", userID))
	}

	message := truncate(entry.Message, maxMessageLength)
	level := strings.ToUpper(strings.TrimSpace(entry.Level))
	switch level {
	case "ERROR":
		log.Error(message, fields...)
	case "WARN":
		log.Warn(message, fields...)
	default:
		log.Info(message, fields...)
	}
}

func buildFields(entry *dto.ClientLogEntry, traceID string) []zap.Field {
	fields := make([]zap.Field, 0, 20)
	// 与 Java 端 CharSequenceUtil.isNotBlank 对齐：过滤 null/空字符串/纯空白
	putIfNotBlank := func(key, value string) {
		if strings.TrimSpace(value) != "" {
			fields = append(fields, zap.String(key, value))
		}
	}

	// 不注入前端 timestamp：FileEncoder 已输出服务端接收时间的 timestamp，避免 JSON 同键冲突
	putIfNotBlank("app", entry.App)
	putIfNotBlank("app_version", entry.AppVersion)
	putIfNotBlank("url", entry.URL)
	putIfNotBlank("user_agent", entry.UserAgent)
	putIfNotBlank("error_type", entry.ErrorType)
	putIfNotBlank("error_source", entry.ErrorSource)
	putIfNotBlank("error_stack", truncate(entry.ErrorStack, maxErrorStackLen))
	putIfNotBlank("method", entry.Method)
	putIfNotBlank("path", entry.Path)
	putIfNotBlank("code", entry.Code)
	putIfNotBlank("type", entry.Type)
	putIfNotBlank("metric_name", entry.MetricName)
	putIfNotBlank("navigation_type", entry.NavigationType)
	putIfNotBlank("resource_url", entry.ResourceURL)
	putIfNotBlank("trace_id", traceID)
	if entry.Status != nil {
		fields = append(fields, zap.Int("status", *entry.Status))
	}
	if entry.Duration != nil {
		fields = append(fields, zap.Float64("duration", *entry.Duration))
	}
	if entry.MetricValue != nil {
		fields = append(fields, zap.Float64("metric_value", *entry.MetricValue))
	}
	return fields
}

func isError(level string) bool {
	return strings.EqualFold(strings.TrimSpace(level), "ERROR")
}

func truncate(value string, maxLength int) string {
	if len(value) > maxLength {
		return value[:maxLength]
	}
	return value
}
