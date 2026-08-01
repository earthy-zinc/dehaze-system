package middleware

import (
	"bytes"
	"io"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// LogLayout 日志layout
type LogLayout struct {
	Time      time.Time
	Metadata  map[string]interface{} // 存储自定义原数据
	Path      string                 // 访问路径
	Query     string                 // 携带query
	Body      string                 // 携带body数据
	IP        string                 // ip地址
	UserAgent string                 // 代理
	Error     string                 // 错误
	Cost      time.Duration          // 花费时间
	Source    string                 // 来源
}

type Logger struct {
	// Filter 用户自定义过滤
	Filter func(c *gin.Context) bool
	// FilterKeyword 关键字过滤(key)
	FilterKeyword func(layout *LogLayout) bool
	// AuthProcess 鉴权处理
	AuthProcess func(c *gin.Context, layout *LogLayout)
	// 日志处理
	Print func(c *gin.Context, layout LogLayout)
	// Source 服务唯一标识
	Source string
}

func (l Logger) SetLoggerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery
		method := c.Request.Method
		var body []byte

		if l.Filter != nil && !l.Filter(c) {
			body, _ = c.GetRawData()
			// 将原body塞回去
			c.Request.Body = io.NopCloser(bytes.NewBuffer(body))
		}

		c.Next()

		cost := time.Since(start)
		status := c.Writer.Status()

		layout := LogLayout{
			Time:      time.Now(),
			Path:      path,
			Query:     query,
			IP:        c.ClientIP(),
			UserAgent: c.Request.UserAgent(),
			Error:     strings.TrimRight(c.Errors.ByType(gin.ErrorTypePrivate).String(), "\n"),
			Cost:      cost,
			Source:    l.Source,
			Metadata: map[string]interface{}{
				"method": method,
				"status": status,
			},
		}

		if l.Filter != nil && !l.Filter(c) {
			layout.Body = string(body)
		}
		if l.AuthProcess != nil {
			// 处理鉴权需要的信息
			l.AuthProcess(c, &layout)
		}
		if l.FilterKeyword != nil {
			// 自行判断key/value 脱敏等
			l.FilterKeyword(&layout)
		}
		// 自行处理日志
		l.Print(c, layout)
	}
}

func DefaultLogger() gin.HandlerFunc {
	return Logger{
		Print: func(c *gin.Context, layout LogLayout) {
			var status int
			if layout.Metadata != nil {
				if s, ok := layout.Metadata["status"]; ok {
					status, _ = s.(int)
				}
			}
			durationMs := float64(layout.Cost.Microseconds()) / 1000.0

			log := logger.WithContext(c.Request.Context())
			fields := []zap.Field{
				zap.Int("status", status),
				zap.Float64("duration_ms", durationMs),
			}
			if layout.Query != "" {
				fields = append(fields, zap.String("query", layout.Query))
			}

			// 每请求一条 INFO ACCESS 日志，method/path/ip/user_agent/trace_id 由 logger.WithContext 自动注入，
			// user_id 由认证层写入 context 后自动注入（未认证时为空）
			log.Info("ACCESS", fields...)
		},
		Source: "GVA",
	}.SetLoggerMiddleware()
}
