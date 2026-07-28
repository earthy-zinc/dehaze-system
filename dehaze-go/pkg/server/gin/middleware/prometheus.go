package middleware

import (
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	// httpRequestsTotal 统计 HTTP 请求总数
	httpRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "path", "status"},
	)

	// httpRequestDuration HTTP 请求耗时（命名对齐 Java Spring Boot 的 http_server_requests_seconds）
	httpRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_server_requests_seconds",
			Help:    "Duration of HTTP requests in seconds",
			Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
		},
		[]string{"method", "path"},
	)

	// httpRequestsInFlight 当前正在处理的请求数
	httpRequestsInFlight = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "http_requests_in_flight",
			Help: "Current number of HTTP requests being processed",
		},
		[]string{"method"},
	)
)

// Prometheus 返回 Prometheus 监控中间件
func Prometheus() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.FullPath()
		if path == "" {
			path = c.Request.URL.Path
		}

		// 排除 swagger、metrics、健康检查路径，避免污染监控指标
		if strings.HasPrefix(path, "/swagger") || path == "/metrics" || path == "/health" || path == "/ready" {
			c.Next()
			return
		}

		method := c.Request.Method

		// 记录正在处理的请求
		httpRequestsInFlight.WithLabelValues(method).Inc()
		defer httpRequestsInFlight.WithLabelValues(method).Dec()

		c.Next()

		// 记录请求总数和耗时
		status := strconv.Itoa(c.Writer.Status())
		httpRequestsTotal.WithLabelValues(method, path, status).Inc()
		duration := time.Since(start).Seconds()
		httpRequestDuration.WithLabelValues(method, path).Observe(duration)
	}
}

// MetricsHandler 返回 Prometheus 指标 HTTP Handler
func MetricsHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		promhttp.Handler().ServeHTTP(c.Writer, c.Request)
	}
}
