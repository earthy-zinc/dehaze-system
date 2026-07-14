package trace

import (
	"context"
	"strings"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

type contextKey string

const (
	traceIDKey contextKey = "trace_id"
	// traceParentKey 用于在 context 中缓存 traceparent
	traceParentKey contextKey = "trace_parent"
	// traceSw8Key 用于在 context 中缓存 sw8
	traceSw8Key contextKey = "trace_sw8"
	// HeaderName HTTP 头字段名
	HeaderName = "X-Trace-ID"
	// HeaderNameTraceParent W3C Trace Context 头字段名
	HeaderNameTraceParent = "traceparent"
	// HeaderNameSw8 SkyWalking 头字段名
	HeaderNameSw8 = "sw8"
	// TraceFieldName 结构化日志字段名
	TraceFieldName = "trace_id"
)

// loggerKey 用于在 context 中缓存带 TraceID 的 logger 实例
type loggerKey struct{}

// NewTraceID 生成 32 字符 hex TraceID（去连字符，对齐 W3C Trace Context 格式）
func NewTraceID() string {
	return strings.ToLower(strings.ReplaceAll(uuid.New().String(), "-", ""))
}

// NewTraceParent 生成符合 W3C Trace Context 的 traceparent
func NewTraceParent(traceID string) string {
	traceID = NormalizeTraceID(traceID)
	if traceID == "" {
		traceID = NewTraceID()
	}
	parentID := strings.ToLower(strings.ReplaceAll(uuid.New().String(), "-", ""))[:16]
	return "00-" + traceID + "-" + parentID + "-01"
}

// NormalizeTraceID 标准化并校验 TraceID（32位小写hex），非法则返回空
func NormalizeTraceID(traceID string) string {
	traceID = strings.ToLower(strings.TrimSpace(traceID))
	if !isHexString(traceID, 32) {
		return ""
	}
	return traceID
}

// ParseTraceParent 从 traceparent 头解析 TraceID，失败返回空
func ParseTraceParent(traceParent string) string {
	traceParent = strings.TrimSpace(traceParent)
	if traceParent == "" {
		return ""
	}
	parts := strings.Split(traceParent, "-")
	if len(parts) != 4 {
		return ""
	}
	if !isHexString(strings.ToLower(parts[0]), 2) {
		return ""
	}
	traceID := NormalizeTraceID(parts[1])
	if traceID == "" {
		return ""
	}
	if !isHexString(strings.ToLower(parts[2]), 16) {
		return ""
	}
	if !isHexString(strings.ToLower(parts[3]), 2) {
		return ""
	}
	return traceID
}

// ExtractTraceIDFromHeaders 从请求头提取 TraceID 与 traceparent
//
// X-Trace-Id 头接受任意非空字符串（与 Java/Python 端行为一致，便于跨服务透传）；
// traceparent 头仍按 W3C 标准严格解析，仅用于补充缺失的 TraceID。
func ExtractTraceIDFromHeaders(traceIDHeader, traceParentHeader string) (string, string) {
	traceID := strings.TrimSpace(traceIDHeader)
	traceParent := strings.TrimSpace(traceParentHeader)
	if traceParent != "" {
		parsed := ParseTraceParent(traceParent)
		if parsed == "" {
			traceParent = ""
		} else if traceID == "" {
			traceID = parsed
		}
	}
	return traceID, traceParent
}

func isHexString(value string, length int) bool {
	if len(value) != length {
		return false
	}
	for i := 0; i < len(value); i++ {
		c := value[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// FromContext 从 context 中提取 TraceID
func FromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if id, ok := ctx.Value(traceIDKey).(string); ok {
		return id
	}
	return ""
}

// GetTraceID 是 FromContext 的别名，便于调用
func GetTraceID(ctx context.Context) string {
	return FromContext(ctx)
}

// WithTraceID 将 TraceID 写入 context
func WithTraceID(ctx context.Context, traceID string) context.Context {
	return context.WithValue(ctx, traceIDKey, traceID)
}

// WithTraceParent 将 traceparent 写入 context
func WithTraceParent(ctx context.Context, traceParent string) context.Context {
	return context.WithValue(ctx, traceParentKey, traceParent)
}

// WithSw8 将 sw8 写入 context
func WithSw8(ctx context.Context, sw8 string) context.Context {
	return context.WithValue(ctx, traceSw8Key, sw8)
}

// TraceIDField 返回 zap.Field，如果 TraceID 为空则返回 zap.Skip()
func TraceIDField(ctx context.Context) zap.Field {
	traceID := FromContext(ctx)
	if traceID == "" {
		return zap.Skip()
	}
	return zap.String(TraceFieldName, traceID)
}

// WithLogger 将带 TraceID 的 logger 缓存到 context 中，避免链路中重复创建
func WithLogger(ctx context.Context, l *zap.Logger) context.Context {
	return context.WithValue(ctx, loggerKey{}, l)
}

// LoggerFromContext 从 context 中取出缓存的 logger，不存在则返回 nil
func LoggerFromContext(ctx context.Context) *zap.Logger {
	if ctx == nil {
		return nil
	}
	if l, ok := ctx.Value(loggerKey{}).(*zap.Logger); ok {
		return l
	}
	return nil
}

// TraceParentFromContext 从 context 中提取 traceparent
func TraceParentFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if value, ok := ctx.Value(traceParentKey).(string); ok {
		return value
	}
	return ""
}

// Sw8FromContext 从 context 中提取 sw8
func Sw8FromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if value, ok := ctx.Value(traceSw8Key).(string); ok {
		return value
	}
	return ""
}
