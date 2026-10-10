"""Prometheus 指标：HTTP 指标 + 推理指标 + GPU 指标。

推理指标随执行器迁入本服务；批量维度由编排侧业务指标聚合，本服务只记单请求。
"""

import asyncio
import contextlib
import logging
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

logger = logging.getLogger(__name__)

# endpoint 取路由路径（不含 query）：控制标签基数，避免查询串炸开时间序列
HTTP_REQUESTS = Counter("dehaze_http_requests_total", "HTTP 请求数", ["endpoint", "status"])
HTTP_DURATION = Histogram(
    "dehaze_http_request_duration_seconds",
    "HTTP 请求处理耗时（秒）",
    ["endpoint"],
    buckets=(0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0),
)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


INFERENCE_DURATION = Histogram(
    "dehaze_inference_duration_seconds",
    "Inference request duration in seconds",
    ["algorithm", "status"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "dehaze_inference_requests_total",
    "Total number of inference requests",
    ["algorithm", "status"],
)

INFERENCE_IMAGE_SIZE = Histogram(
    "dehaze_inference_image_size_pixels",
    "Inference input image size (width * height)",
    ["algorithm"],
    buckets=(256 * 256, 512 * 512, 1024 * 1024, 2048 * 2048, 4096 * 4096),
)

GPU_MEMORY_USED = Gauge(
    "dehaze_gpu_memory_used_bytes", "GPU memory used in bytes", ["device_id", "device_name"]
)
GPU_MEMORY_TOTAL = Gauge(
    "dehaze_gpu_memory_total_bytes", "GPU total memory in bytes", ["device_id", "device_name"]
)
GPU_MEMORY_FREE = Gauge(
    "dehaze_gpu_memory_free_bytes", "GPU memory free in bytes", ["device_id", "device_name"]
)
GPU_UTILIZATION = Gauge(
    "dehaze_gpu_utilization_percent",
    "GPU utilization percentage (0-100)",
    ["device_id", "device_name"],
)
GPU_MEMORY_UTILIZATION = Gauge(
    "dehaze_gpu_memory_utilization_percent",
    "GPU memory utilization percentage (0-100)",
    ["device_id", "device_name"],
)
GPU_TEMPERATURE = Gauge(
    "dehaze_gpu_temperature_celsius", "GPU temperature in celsius", ["device_id", "device_name"]
)
GPU_POWER_USAGE = Gauge(
    "dehaze_gpu_power_usage_watts", "GPU power usage in watts", ["device_id", "device_name"]
)


def record_inference_metrics(
    algorithm: str, duration_seconds: float, status: str, image_size: int | None = None
) -> None:
    INFERENCE_DURATION.labels(algorithm=algorithm, status=status).observe(duration_seconds)
    INFERENCE_REQUESTS_TOTAL.labels(algorithm=algorithm, status=status).inc()
    if image_size is not None:
        INFERENCE_IMAGE_SIZE.labels(algorithm=algorithm).observe(image_size)


class GPUMetricsCollector:
    """GPU 显存/利用率/温度/功耗采集（pynvml）；无 GPU 环境自动空转。"""

    def __init__(self, collect_interval: float = 5):
        self.collect_interval = collect_interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._pynvml_available = False
        self._device_count = 0
        self._pynvml: Any = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._pynvml_available = True
            self._device_count = pynvml.nvmlDeviceGetCount()
            logger.info("GPU 指标采集器初始化成功，检测到 %s 个 GPU 设备", self._device_count)
        except Exception as e:
            logger.warning("GPU 指标采集器初始化失败（无 GPU 环境？）: %s", e)

    async def start(self) -> None:
        if not self._pynvml_available:
            logger.info("GPU 不可用，跳过 GPU 指标采集")
            return
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info("GPU 指标采集器已启动，采集间隔: %ss", self.collect_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._pynvml_available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning("NVML 关闭失败: %s", e, exc_info=True)
        logger.info("GPU 指标采集器已停止")

    async def _collect_loop(self) -> None:
        while self._running:
            try:
                self._collect_once()
            except Exception as e:
                logger.error("GPU 指标采集异常: %s", e)
            await asyncio.sleep(self.collect_interval)

    def _collect_once(self) -> None:
        pynvml = self._pynvml
        for index in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                device_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode("utf-8")
                labels = {"device_id": str(index), "device_name": device_name}

                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                GPU_MEMORY_USED.labels(**labels).set(float(memory.used))
                GPU_MEMORY_TOTAL.labels(**labels).set(float(memory.total))
                GPU_MEMORY_FREE.labels(**labels).set(float(memory.free))

                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                GPU_UTILIZATION.labels(**labels).set(float(utilization.gpu))
                GPU_MEMORY_UTILIZATION.labels(**labels).set(float(utilization.memory))

                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    GPU_TEMPERATURE.labels(**labels).set(float(temperature))
                except Exception as e:
                    # 部分设备不支持温度采集，属预期，不应中断其他指标
                    logger.debug("GPU %s 温度指标跳过: %s", index, e)

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW -> W
                    GPU_POWER_USAGE.labels(**labels).set(float(power))
                except Exception as e:
                    # 部分设备不支持功耗采集，属预期，不应中断其他指标
                    logger.debug("GPU %s 功耗指标跳过: %s", index, e)
            except Exception as e:
                logger.warning("GPU %s 指标采集失败: %s", index, e)
