"""AI 对话模块 - 中间产物服务

产物注册/失效联动、多模态视觉读取限额与读取、引用查询。visual_read 的
多模态调用组装走 llm_client.stream_chat，不侵入其韧性主链；返回 (text,
input_tokens) 供工具壳将多模态 token 归集到推理 ctx。
"""

from datetime import datetime

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_artifact import SysAiArtifact
from app.models.schema.ai_artifact import ArtifactResult
from app.models.schema.common import PageResult
from app.repository.ai_artifact_repository import ai_artifact_repository
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_model_repository import ai_model_repository
from app.repository.file_repository import file_repository
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository
from app.repository.pred_eval_log_repository import eval_log_repository, pred_log_repository
from app.service.ai.llm_client import llm_client
from app.service.storage.factory import get_storage_by_name

# 多模态当日计数的跨会话全局 Redis Key 前缀
_VISUAL_QUOTA_KEY_PREFIX = "ai:multimodal"
# 超限时降级文案（无需上下文变量，直接拼接）
_VISUAL_QUOTA_EXCEEDED = "视觉读取已达今日上限，基于指标判断："

# artifact.ref_type 可能引用的业务表（图片文件 ID 的解析方式）
_IMAGE_REF_TYPES = {"sys_file", "sys_pred_log", "sys_eval_log"}


class AiArtifactService:

    async def list_by_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        page: int,
        size: int,
    ) -> PageResult[ArtifactResult]:
        """查询会话产物列表（校验会话归属）"""
        conv = await ai_conversation_repository.get_by_id_and_user(
            db, conv_id, user_id
        )
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")

        artifacts, total = await ai_artifact_repository.list_by_conversation(
            db, conv_id, page, size
        )

        return PageResult(
            list=[ArtifactResult.model_validate(a) for a in artifacts],
            total=total
        )

    async def list_by_message(
        self,
        db: AsyncSession,
        msg_id: int,
        user_id: int,
    ) -> list[ArtifactResult]:
        """查询消息关联产物（校验归属）"""
        msg = await ai_message_repository.get_by_id_and_user(db, msg_id, user_id)

        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")

        artifacts = await ai_artifact_repository.list_by_message(db, msg_id)

        return [ArtifactResult.model_validate(a) for a in artifacts]

    async def register_artifact(
        self,
        db: AsyncSession,
        conv_id: int,
        msg_id: int,
        artifact_type: str,
        ref_type: str | None = None,
        ref_id: int | None = None,
        summary: dict | None = None,
    ) -> ArtifactResult:
        """注册产物（供工具调用产生结果时调用）"""
        artifact = SysAiArtifact(
            conversation_id=conv_id,
            message_id=msg_id,
            type=artifact_type,
            ref_type=ref_type,
            ref_id=ref_id,
            summary=summary,
        )
        artifact = await ai_artifact_repository.create(db, artifact)
        return ArtifactResult.model_validate(artifact)

    async def mark_invalid_for_file(self, db: AsyncSession, file_id: int) -> None:
        """文件删除时联动失效产物：直接引用 sys_file，及经预测/评估日志间接引用。"""
        await ai_artifact_repository.mark_invalid(db, "sys_file", file_id)

        pred_ids = await pred_log_repository.list_ids_by_file(db, file_id)
        for log_id in pred_ids:
            await ai_artifact_repository.mark_invalid(db, "sys_pred_log", log_id)

        eval_ids = await eval_log_repository.list_ids_by_file(db, file_id)
        for log_id in eval_ids:
            await ai_artifact_repository.mark_invalid(db, "sys_eval_log", log_id)

    async def list_by_ref(
        self,
        db: AsyncSession,
        ref_type: str,
        ref_id: int,
        user_id: int,
    ) -> list[ArtifactResult]:
        """按业务引用反查产物列表（校验会话归属）"""
        artifacts = await ai_artifact_repository.list_by_ref(db, ref_type, ref_id)
        return await self._filter_owned(db, artifacts, user_id)

    async def get_detail(
        self,
        db: AsyncSession,
        artifact_id: int,
        user_id: int,
    ) -> dict:
        """产物详情：记录 + 按需解析图片运行时 URL（供前端展示）。"""
        artifact = await ai_artifact_repository.get_by_id(db, artifact_id)
        if not artifact or artifact.is_invalid:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "产物不存在或已失效")
        conv = await ai_conversation_repository.get_by_id_and_user(
            db, artifact.conversation_id, user_id
        )
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "产物所属会话不存在")
        return {
            "artifact": ArtifactResult.model_validate(artifact),
            "imageUrl": await self._resolve_image_url(db, artifact),
        }

    async def _filter_owned(
        self,
        db: AsyncSession,
        artifacts: list[SysAiArtifact],
        user_id: int,
    ) -> list[ArtifactResult]:
        """过滤当前用户所属会话的产物"""
        result = []
        for a in artifacts:
            conv = await ai_conversation_repository.get_by_id_and_user(
                db, a.conversation_id, user_id
            )
            if conv:
                result.append(ArtifactResult.model_validate(a))
        return result

    async def get_message_artifact_refs(
        self,
        db: AsyncSession,
        message_ids: list[int],
    ) -> dict[int, list[dict]]:
        """批量取消息关联产物（跨成员契约，供上下文组装引用层）。

        Args:
            message_ids: 消息 ID 列表。

        Returns:
            {message_id: [{id, type, summary}]}，仅包含 is_invalid=0 的产物。
        """
        artifacts = await ai_artifact_repository.list_by_message_ids(db, message_ids)
        grouped: dict[int, list[dict]] = {}
        for a in artifacts:
            grouped.setdefault(a.message_id, []).append(
                {"id": a.id, "type": a.type, "summary": a.summary}
            )
        return grouped

    async def _get_visual_limit(self, db: AsyncSession, user_id: int) -> int:
        """读取用户等级对应的多模态视觉读取日限额。"""
        member = await member_repository.get_by_user_id(db, user_id)
        level_code = member.level_code if member else "level_0"
        benefit = await member_benefit_repository.get_by_level_code(db, level_code)
        return benefit.multimodal_limit if benefit and benefit.multimodal_limit else 0

    async def check_visual_quota(
        self,
        db: AsyncSession,
        redis: Redis,
        user_id: int,
    ) -> tuple[int, int]:
        """纯读当日多模态视觉读取额度（供前端展示，不承担并发判定）。

        Returns:
            (used, limit)：当日已用次数、日限额。并发安全判定由
            _consume_visual_quota（INCR 原子）承担。
        """
        limit = await self._get_visual_limit(db, user_id)
        key = self._visual_quota_key(user_id)
        used = int(await redis.get(key) or 0)
        return used, limit

    def _visual_quota_key(self, user_id: int) -> str:
        return f"{_VISUAL_QUOTA_KEY_PREFIX}:{user_id}:{datetime.now().strftime('%Y%m%d')}"

    async def _consume_visual_quota(
        self,
        redis: Redis,
        user_id: int,
        limit: int,
    ) -> bool:
        """原子消费一次多模态视觉读取额度（INCR 先行后判断）。

        语义：同用户并发请求下 INCR 是原子的，不可能多个请求同时读到同一
        new_count 而都放行——若 new_count > limit 则 DECR 回退并返回 False（该次
        拒绝），避免并发间隙绕过日上限。首次 INCR（new_count==1）设置 TTL 至
        次日零点，午夜过期自动重置。
        """
        key = self._visual_quota_key(user_id)
        new_count = int(await redis.incr(key))
        if new_count == 1:
            now = datetime.now()
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            ttl_seconds = int((tomorrow - now).total_seconds()) + 86400
            await redis.expire(key, ttl_seconds)
        if new_count > limit:
            await redis.decr(key)
            return False
        return True

    async def _resolve_image_url(self, db: AsyncSession, artifact: SysAiArtifact) -> str | None:
        """经 artifact 的 ref 链路解析出图片运行时 URL（URL 不落库，按需拼接）。"""
        file_id = None
        if artifact.ref_type == "sys_file":
            file_id = artifact.ref_id
        elif artifact.ref_type == "sys_pred_log":
            pred = await pred_log_repository.get_by_id(db, artifact.ref_id)
            file_id = pred.pred_file_id if pred else None
        elif artifact.ref_type == "sys_eval_log":
            eval_log = await eval_log_repository.get_by_id(db, artifact.ref_id)
            file_id = eval_log.pred_file_id if eval_log else None
        if not file_id:
            return None
        file_info = await file_repository.get_by_id(db, file_id)
        if not file_info:
            return None
        return get_storage_by_name(file_info.storage).get_url(file_info.object_name)

    async def _pick_multimodal_model(self, db: AsyncSession, model_id: str | None) -> str | None:
        """选择多模态模型：优先当前会话模型，否则取任一启用多模态模型。"""
        if model_id:
            current = await ai_model_repository.get_by_model_id(db, model_id)
            if current and current.supports_multimodal:
                return model_id
        enabled = await ai_model_repository.list_enabled(db)
        for model in enabled:
            if model.supports_multimodal:
                return model.model_id
        return None

    async def visual_read(
        self,
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        artifact_id: int,
        model_id: str | None = None,
    ) -> tuple[str, int]:
        """多模态视觉读取产物图片（含限额判定）。

        超限时降级返回产物摘要文本；未超则 INCR 计数并调多模态模型读取图片，
        返回 (视觉理解结果文本, 多模态 input_tokens)。input_tokens 由工具壳
        归集到推理 ctx，随本次推理一并计入 Token 消耗。
        """
        artifact = await ai_artifact_repository.get_by_id(db, artifact_id)
        if not artifact or artifact.is_invalid:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "产物不存在或已失效")
        conv = await ai_conversation_repository.get_by_id_and_user(
            db, artifact.conversation_id, user_id
        )
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "产物所属会话不存在")

        image_url = await self._resolve_image_url(db, artifact)
        if not image_url:
            # 图片引用缺失（如预测文件尚未生成）时也走降级文本
            return _VISUAL_QUOTA_EXCEEDED + str(artifact.summary or ""), 0

        multimodal_model_id = await self._pick_multimodal_model(db, model_id)
        if not multimodal_model_id:
            return _VISUAL_QUOTA_EXCEEDED + str(artifact.summary or ""), 0

        # 原子消费额度（INCR 先行）：超限该次拒绝且已回退计数；并发不可绕过
        limit = await self._get_visual_limit(db, user_id)
        if not await self._consume_visual_quota(redis, user_id, limit):
            return _VISUAL_QUOTA_EXCEEDED + str(artifact.summary or ""), 0

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请评估这张图像的整体效果，并简要说明可以改进的地方。",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        text_parts: list[str] = []
        usage: dict = {}
        async for chunk in llm_client.stream_chat(db, redis, multimodal_model_id, messages):
            if chunk.type == "text_delta":
                text_parts.append(chunk.content)
            elif chunk.type == "done" and chunk.usage:
                usage = chunk.usage
        text = "".join(text_parts).strip()
        if not text:
            return _VISUAL_QUOTA_EXCEEDED + str(artifact.summary or ""), 0
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        return text, int(input_tokens)


ai_artifact_service = AiArtifactService()
