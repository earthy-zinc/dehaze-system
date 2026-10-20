"""图像编辑模型目录（阶段 3）

模型清单的**技术事实源**收敛到本服务：`app/artifacts.json` 以统一模型表的规范
slug（如 `aecrnet`/`c2pnet-its`）为主键，映射到运行工件（import_path/权重相对
路径）。编排侧（clarity-python）只按 model_id 调用 `/v1/images/edits`，不再从
DB 下发 import_path/weights——消除「DB 与目录双事实源」。

种子再生成口径（阶段 1 迁移 `2026-09-29_unified_model_management_stage1.sql` 的**历史**溯源，
`sys_algorithm` 已随算法管理退役，不再是运行时事实源）：
- slug = slugify(原 `sys_algorithm.import_path` 模块名)（algorithm.AECRNet.run → aecrnet）
- 同 slug base 存在多个不同 weights path 时 slug = base-suffix（C2PNet/ITS.pkl → c2pnet-its）
- 仅迁可运行行（path 非空且 type<>'TEST'）；enabled 由原 `sys_algorithm` 6 态映射
  （已发布/测试中/待审核→true，草稿/停用/已归档→false）
- 当前权威清单 = `sys_ai_model` 中 `model_type='image_edit'` 的 `model_id`（两者保持一致）

权重文件不在服务目录（懒下载，见 algorithm/model_loader.py），可用性由
`GET /v1/models/{slug}` 的 weights 探测承担，目录只声明静态映射。
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from app.errors import ModalError

logger = logging.getLogger(__name__)

_ARTIFACTS_FILE = Path(__file__).parent / "artifacts.json"


@lru_cache(maxsize=1)
def artifacts() -> dict:
    """slug → {import_path, weights_path, name, enabled}；进程内缓存（种子随发布固定）"""
    data = json.loads(_ARTIFACTS_FILE.read_text("utf-8"))
    logger.info("图像编辑工件目录加载: %s 个模型", len(data))
    return data


def resolve_edit(model_id: str) -> dict:
    """按规范 slug 解析运行工件；未知或停用模型拒绝推理。

    enabled=false 对应阶段 1 迁移时原算法表的草稿/停用/已归档行：目录保留映射
    供展示与可用性探测，但不承接推理（编排侧的启停另有模型表 status 承载，
    此处是双保险——目录级禁用同样拦截）。
    """
    item = artifacts().get(model_id)
    if item is None:
        raise ModalError("MODEL_NOT_FOUND", f"模型不存在: {model_id}")
    if not item["enabled"]:
        raise ModalError("MODEL_DISABLED", f"模型已停用: {model_id}")
    return item


def catalog_snapshot() -> list[dict]:
    """/v1/models 的图像编辑项（对齐 OpenAI 风格：id 为主键，详情内嵌）"""
    return [
        {
            "id": slug,
            "type": "image_edit",
            "available": item["enabled"],
            "unavailable_reason": None if item["enabled"] else "模型已停用",
            "owned_by": "local-modal",
            "artifacts": {
                "import_path": item["import_path"],
                "weights_path": item["weights_path"],
            },
            "display_name": item["name"],
        }
        for slug, item in sorted(artifacts().items())
    ]
