"""缓存命中率 Prometheus 指标

三端统一指标命名规范：
- dehaze_cache_hits_total{layer} - 缓存命中次数（L1/L2/L1_null/L2_null）
- dehaze_cache_misses_total{layer} - 缓存未命中次数（L1/L2/bloom/all）
- dehaze_cache_loader_total{result} - 数据源加载次数（hit/miss/error）
"""
from prometheus_client import Counter

# 缓存命中次数（按层级统计：L1/L2）
CACHE_HITS_TOTAL = Counter(
    "dehaze_cache_hits_total",
    "Total number of cache hits by layer (L1/L2)",
    ["layer"],
)

# 缓存未命中次数
CACHE_MISSES_TOTAL = Counter(
    "dehaze_cache_misses_total",
    "Total number of cache misses by layer (L1/L2)",
    ["layer"],
)

# 数据源加载次数（缓存完全未命中后回源）
CACHE_LOADER_TOTAL = Counter(
    "dehaze_cache_loader_total",
    "Total number of data loader invocations (cache fully missed)",
    ["result"],  # hit/miss/error
)


def record_hit(layer: str) -> None:
    """记录缓存命中"""
    CACHE_HITS_TOTAL.labels(layer=layer).inc()


def record_miss(layer: str) -> None:
    """记录缓存未命中"""
    CACHE_MISSES_TOTAL.labels(layer=layer).inc()


def record_loader(result: str) -> None:
    """记录数据源加载"""
    CACHE_LOADER_TOTAL.labels(result=result).inc()
