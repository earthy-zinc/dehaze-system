import asyncio
import logging
from typing import Optional

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

GPU_MEMORY_USED = Gauge(
    "dehaze_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["device_id", "device_name"],
)

GPU_MEMORY_TOTAL = Gauge(
    "dehaze_gpu_memory_total_bytes",
    "GPU total memory in bytes",
    ["device_id", "device_name"],
)

GPU_MEMORY_FREE = Gauge(
    "dehaze_gpu_memory_free_bytes",
    "GPU memory free in bytes",
    ["device_id", "device_name"],
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
    "dehaze_gpu_temperature_celsius",
    "GPU temperature in celsius",
    ["device_id", "device_name"],
)

GPU_POWER_USAGE = Gauge(
    "dehaze_gpu_power_usage_watts",
    "GPU power usage in watts",
    ["device_id", "device_name"],
)


class GPUMetricsCollector:
    def __init__(self, collect_interval: int = 5):
        self.collect_interval = collect_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._pynvml_available = False
        self._device_count = 0

        # 尝试初始化 pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._pynvml_available = True
            self._device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"GPU 指标采集器初始化成功，检测到 {self._device_count} 个 GPU 设备")
        except Exception as e:
            logger.warning(f"GPU 指标采集器初始化失败（非 GPU 环境？）: {e}")
            self._pynvml = None

    async def start(self) -> None:
        if not self._pynvml_available:
            logger.info("GPU 不可用，跳过 GPU 指标采集")
            return

        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info(f"GPU 指标采集器已启动，采集间隔: {self.collect_interval}s")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # 清理 pynvml
        if self._pynvml and self._pynvml_available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        logger.info("GPU 指标采集器已停止")

    async def _collect_loop(self) -> None:
        while self._running:
            try:
                self._collect_once()
            except Exception as e:
                logger.error(f"GPU 指标采集异常: {e}")

            await asyncio.sleep(self.collect_interval)

    def _collect_once(self) -> None:
        if not self._pynvml_available or self._pynvml is None:
            return

        pynvml = self._pynvml  # 类型收窄
        for i in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = pynvml.nvmlDeviceGetName(handle)
                # 处理 bytes 类型（pynvml 返回 bytes）
                if isinstance(device_name, bytes):
                    device_name = device_name.decode("utf-8")

                device_labels = {"device_id": str(
                    i), "device_name": device_name}

                # 显存信息
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                GPU_MEMORY_USED.labels(
                    **device_labels).set(float(mem_info.used))
                GPU_MEMORY_TOTAL.labels(
                    **device_labels).set(float(mem_info.total))
                GPU_MEMORY_FREE.labels(
                    **device_labels).set(float(mem_info.free))

                # GPU 利用率
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                GPU_UTILIZATION.labels(
                    **device_labels).set(float(util_info.gpu))
                GPU_MEMORY_UTILIZATION.labels(
                    **device_labels).set(float(util_info.memory))

                # 温度
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU)
                    GPU_TEMPERATURE.labels(**device_labels).set(float(temp))
                except Exception:
                    pass  # 部分设备可能不支持

                # 功耗
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(
                        handle) / 1000  # mW -> W
                    GPU_POWER_USAGE.labels(**device_labels).set(float(power))
                except Exception:
                    pass  # 部分设备可能不支持

            except Exception as e:
                logger.warning(f"GPU {i} 指标采集失败: {e}")


_collector: Optional[GPUMetricsCollector] = None


def get_gpu_metrics_collector(collect_interval: int = 5) -> GPUMetricsCollector:
    global _collector
    if _collector is None:
        _collector = GPUMetricsCollector(collect_interval=collect_interval)
    return _collector


async def collect_gpu_metrics(collect_interval: int = 5) -> Optional[GPUMetricsCollector]:
    """
    启动 GPU 指标采集（仅 Prometheus 启用时）

    Args:
        collect_interval: 采集间隔（秒）

    Returns:
        GPUMetricsCollector 实例，未启用时返回 None
    """
    from app.config import settings

    if not settings.PROMETHEUS_ENABLED:
        logger.info("Prometheus 未启用，跳过 GPU 指标采集")
        return None

    collector = get_gpu_metrics_collector(collect_interval)
    await collector.start()
    return collector
