"""能力发现：算法清单、指标集清单、权重可用性。

声明的是**运行时能力**；算法权重相对路径与算法标识仍由编排侧从业务目录解析后随请求下发，
服务侧不查库。
"""

import asyncio
import logging
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from algorithm.metrics import METRICS_CONFIG
from algorithm.model_loader import check_model_exists
from app.config import settings
from app.errors import ModalError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["discovery"])

# 算法目录内的 dehaze 定义探测：静态判定，避免能力发现导入 30 个算法模块
# （WPXNet/RIDCP 等导入时即触发 CUDA 算子 JIT 编译）
_DEHAZE_DEF = re.compile(r"^\s*def dehaze\s*\(", re.MULTILINE)


@lru_cache(maxsize=1)
def _algorithm_items() -> tuple[dict, ...]:
    """服务自带的算法实现清单；算法目录随发布固定，进程内缓存避免每次巡检都扫盘。"""
    import algorithm

    root = Path(algorithm.__file__).parent
    items = []
    for directory in sorted(p for p in root.iterdir() if (p / "run.py").is_file()):
        exports_dehaze = bool(_DEHAZE_DEF.search((directory / "run.py").read_text("utf-8")))
        items.append(
            {
                "id": f"algorithm.{directory.name}.run",
                "type": "inference",
                "available": exports_dehaze,
                "unavailable_reason": None if exports_dehaze else "算法模块未导出 dehaze()",
                "capabilities": {"task_types": ["dehazing"]},
                "limits": {
                    "concurrency": settings.INFERENCE_THREAD_POOL_SIZE,
                    "timeout_seconds": settings.INFERENCE_TIMEOUT_SECONDS,
                },
            }
        )
    return tuple(items)


@lru_cache(maxsize=1)
def _pyiqa_status() -> tuple[bool, str | None]:
    try:
        # pyright: ignore
        import pyiqa  # noqa: F401
    except ImportError as e:
        logger.warning("pyiqa 不可用，指标集标记为不可服务: %s", e)
        return False, f"pyiqa 不可用: {e}"
    return True, None


def _metric_items() -> list[dict]:
    available, reason = _pyiqa_status()
    return [
        {
            "id": name,
            "type": "metrics",
            "available": available,
            "unavailable_reason": reason,
            "capabilities": {
                "label": config["label"],
                "better": config["better"],
                "requires_clear": config["requires_clear"],
                "description": config["description"],
            },
            "limits": {"concurrency": settings.INFERENCE_THREAD_POOL_SIZE},
        }
        for name, config in METRICS_CONFIG.items()
    ]


@router.get("/models", summary="能力发现：算法与指标集清单")
async def list_models() -> dict:
    # image_edit 项（统一模型管理阶段 3）：规范 slug 为 id，工件映射内嵌，
    # 是编排侧模型驱动推理的清单事实源；下方 algorithm.* 项为旧编排链路
    # （算法 CRUD 权重校验）保留，阶段 3 编排切换后随算法管理一并退役
    from app import catalog as image_catalog

    return {"data": [*_algorithm_items(), *_metric_items(), *image_catalog.catalog_snapshot()]}


@router.get("/models/{model_id:path}", summary="单项详情：算法标识或权重相对路径")
async def get_model(model_id: str) -> dict:
    if model_id.startswith("algorithm."):
        for item in _algorithm_items():
            if item["id"] == model_id:
                return item
        raise ModalError("NOT_FOUND", f"算法不存在: {model_id}")

    # 编排侧算法 CRUD 的权重存在性校验改查本端点，size_bytes 用于回填算法体积。
    # type 固定 inference：权重是算法项的子资源，type 枚举不新增 weights。
    size_bytes = await asyncio.to_thread(check_model_exists, model_id)
    return {
        "id": model_id,
        "type": "inference",
        "available": size_bytes is not None,
        "unavailable_reason": None
        if size_bytes is not None
        else "权重不可用（本地无缓存且下载源不可达）",
        "size_bytes": size_bytes,
    }
