"""AI 模型注册表：模型 → 候选供应商路由解析

get_call_routes 从数据库读取模型配置构建「降级链候选路由序列」，是所有模态
（LLM/Embedding/TTS）调用前路由决策的单一实现。模型配置由 sys_ai_provider /
sys_ai_model 驱动，后续在此扩展为跨模态配置化路由（见架构文档）。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_model import SysAiModel
from app.repository.ai_model_repository import ai_model_repository

# 降级链最大深度：防配置环导致的无限递归
_FALLBACK_CHAIN_MAX_DEPTH = 5


def _model_route(model: SysAiModel) -> dict:
    """将模型实体序列化为候选路由项"""
    return {
        "model_pk": model.id,
        "model_id": model.model_id,
        "provider_id": model.provider_id,
        "model_config": {
            "max_output_tokens": model.max_output_tokens,
            "max_context_tokens": model.max_context_tokens,
            "supports_multimodal": model.supports_multimodal,
            "supports_tool_call": model.supports_tool_call,
            "supports_streaming": model.supports_streaming,
        },
    }


def _model_meets_caps(model: SysAiModel, required_caps: set[str]) -> bool:
    """校验模型能力是否满足全部要求（required_caps 中不存在的能力视为不要求）"""
    for cap in required_caps:
        if cap == "multimodal" and not model.supports_multimodal:
            return False
        if cap == "tool_call" and not model.supports_tool_call:
            return False
        if cap == "streaming" and not model.supports_streaming:
            return False
    return True


class ModelRegistry:
    """AI 模型注册表（单例）：按 model_id 解析降级链候选路由序列"""

    async def get_call_routes(
        self,
        db: AsyncSession,
        model_id: str,
        required_caps: set[str],
    ) -> list[dict]:
        """降级链候选路由序列：[{"model_pk","model_id","provider_id","model_config"}...]。

        顺序：该 model_id 全部启用行（当前/备用供应商）→ 降级链各级
        （fallback_model_id 逐级，能力匹配过滤 required_caps ⊆ 模型 supports_*，
        防环：已出现的 model_pk 跳过，链深上限 5）。
        """
        routes: list[dict] = []
        seen: set[int] = set()

        # 1. 当前 model_id 的全部启用行（同模型多供应商，保持优先级）
        current_rows = await ai_model_repository.list_enabled_by_model_id(db, model_id)
        for row in current_rows:
            seen.add(row.id)
            routes.append(_model_route(row))

        # 2. 沿降级链逐级扩展（按主键引用，能力匹配过滤）
        pending = current_rows
        depth = 0
        while pending and depth < _FALLBACK_CHAIN_MAX_DEPTH:
            next_targets = [
                m.fallback_model_id
                for m in pending
                if m.fallback_model_id is not None and m.fallback_model_id not in seen
            ]
            if not next_targets:
                break
            fallback_rows = await ai_model_repository.list_enabled_by_pks(db, next_targets)
            depth += 1
            matched: list[SysAiModel] = []
            for row in fallback_rows:
                if row.id in seen:
                    continue
                seen.add(row.id)
                if _model_meets_caps(row, required_caps):
                    matched.append(row)
                    routes.append(_model_route(row))
            pending = matched

        return routes


model_registry = ModelRegistry()
