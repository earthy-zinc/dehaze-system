"""计费可观测 Prometheus 指标（与 cache_metrics / task_metrics 同风格：Counter + record_* 助手）

指标清单：
- dehaze_voice_unpriced_total{model_type} - 未配置价格、按 0 积分落账的实扣次数

`model_type` 取计费类型（`unit_billing` 共用的 asr / tts / image）：asr/tts 的计费类型
即模型类型，图像路径为 `image`。该计数是"未配置价格"告警与对账的入口——配了价却仍是 0
分（如单价档位缺失）会在此持续增长，而正常免费模型（显式 0 价）不计入。
"""

from prometheus_client import Counter

UNPRICED_TOTAL = Counter(
    "dehaze_voice_unpriced_total",
    "Total number of unit-billed charges recorded as 0 credits because no price is configured",
    ["model_type"],
)


def record_unpriced(model_type: str) -> None:
    """记录一次「未配置价格、计 0 积分」实扣"""
    UNPRICED_TOTAL.labels(model_type=model_type).inc()
