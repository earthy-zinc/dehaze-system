// Package metrics 提供业务 Prometheus 指标采集（预测/评估/任务）。
//
// 三端统一命名规范（dehaze_ 前缀）：
//   - dehaze_prediction_total{status}            预测请求计数（对齐 Java）
//   - dehaze_prediction_duration_seconds{status} 预测请求耗时
//   - dehaze_evaluation_total{status}            评估请求计数（对齐 Java）
//   - dehaze_evaluation_duration_seconds{status} 评估请求耗时
//   - dehaze_task_total{task_type,status}        异步任务计数（对齐 Java/Python）
//   - dehaze_task_duration_seconds{task_type,status} 异步任务耗时
//
// HTTP 请求指标见 pkg/server/gin/middleware/prometheus.go，
// 缓存指标见 pkg/cache/multilevel/metrics.go。
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	predictionTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dehaze_prediction_total",
			Help: "Total number of prediction requests",
		},
		[]string{"status"}, // success/failure
	)

	predictionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dehaze_prediction_duration_seconds",
			Help:    "Prediction request duration in seconds",
			Buckets: []float64{0.5, 1, 2, 5, 10, 30, 60, 120, 300},
		},
		[]string{"status"},
	)

	evaluationTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dehaze_evaluation_total",
			Help: "Total number of evaluation requests",
		},
		[]string{"status"}, // success/failure
	)

	evaluationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dehaze_evaluation_duration_seconds",
			Help:    "Evaluation request duration in seconds",
			Buckets: []float64{0.5, 1, 2, 5, 10, 30, 60, 120, 300},
		},
		[]string{"status"},
	)

	taskTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dehaze_task_total",
			Help: "Total number of async tasks",
		},
		[]string{"task_type", "status"}, // status: completed/failed/cancelled
	)

	taskDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dehaze_task_duration_seconds",
			Help:    "Async task duration in seconds",
			Buckets: []float64{1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600},
		},
		[]string{"task_type", "status"},
	)
)

// RecordPrediction 记录一次预测请求的终态与耗时（秒）
func RecordPrediction(status string, durationSeconds float64) {
	predictionTotal.WithLabelValues(status).Inc()
	predictionDuration.WithLabelValues(status).Observe(durationSeconds)
}

// RecordEvaluation 记录一次评估请求的终态与耗时（秒）
func RecordEvaluation(status string, durationSeconds float64) {
	evaluationTotal.WithLabelValues(status).Inc()
	evaluationDuration.WithLabelValues(status).Observe(durationSeconds)
}

// RecordTask 记录一个异步任务的终态与耗时（秒）
func RecordTask(taskType, status string, durationSeconds float64) {
	taskTotal.WithLabelValues(taskType, status).Inc()
	taskDuration.WithLabelValues(taskType, status).Observe(durationSeconds)
}
