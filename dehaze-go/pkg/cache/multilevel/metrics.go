package multilevel

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// 缓存命中率 Prometheus 指标
var (
	// cacheHitsTotal 缓存命中次数（按层级统计：L1/L2）
	cacheHitsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dehaze_cache_hits_total",
			Help: "Total number of cache hits by layer (L1/L2)",
		},
		[]string{"layer"},
	)

	// cacheMissesTotal 缓存未命中次数
	cacheMissesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dehaze_cache_misses_total",
			Help: "Total number of cache misses by layer (L1/L2)",
		},
		[]string{"layer"},
	)

	// cacheLoaderTotal 数据源加载次数（缓存完全未命中后回源）
	cacheLoaderTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dehaze_cache_loader_total",
			Help: "Total number of data loader invocations (cache fully missed)",
		},
		[]string{"result"}, // hit/miss/error
	)

	// cacheOperationDuration 缓存操作耗时
	cacheOperationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dehaze_cache_operation_duration_seconds",
			Help:    "Cache operation duration in seconds",
			Buckets: []float64{.0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5},
		},
		[]string{"operation", "layer"}, // operation: get/set/delete, layer: L1/L2/multi
	)
)

// recordHit 记录缓存命中
func recordHit(layer string) {
	cacheHitsTotal.WithLabelValues(layer).Inc()
}

// recordMiss 记录缓存未命中
func recordMiss(layer string) {
	cacheMissesTotal.WithLabelValues(layer).Inc()
}

// recordLoader 记录数据源加载
func recordLoader(result string) {
	cacheLoaderTotal.WithLabelValues(result).Inc()
}
