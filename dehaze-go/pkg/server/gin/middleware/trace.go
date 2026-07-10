package middleware

import (
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// Trace 为每个请求生成或提取 TraceID，写入 context 和响应头
func Trace() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 优先从请求头提取（支持上游服务传入）
		traceID, traceParent := trace.ExtractTraceIDFromHeaders(
			c.GetHeader(trace.HeaderName),
			c.GetHeader(trace.HeaderNameTraceParent),
		)
		if traceID == "" {
			traceID = trace.NewTraceID()
		}
		if traceParent == "" {
			traceParent = trace.NewTraceParent(traceID)
		}

		// 写入 c.Request.Context()（让下游 service/repo 通过标准 ctx 获取）
		ctx := trace.WithTraceID(c.Request.Context(), traceID)
		ctx = trace.WithTraceParent(ctx, traceParent)

		// 将带 TraceID 的 logger 缓存到 context，整条链路复用同一实例
		ctx = trace.WithLogger(ctx, zap.L().With(zap.String(trace.TraceFieldName, traceID)))

		c.Request = c.Request.WithContext(ctx)

		// 响应头回传
		c.Header(trace.HeaderName, traceID)
		if traceParent != "" {
			c.Header(trace.HeaderNameTraceParent, traceParent)
		}

		c.Next()
	}
}
