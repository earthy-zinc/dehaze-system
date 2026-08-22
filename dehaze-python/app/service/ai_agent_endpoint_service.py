"""外部 A2A 端点管理服务（Endpoints）

管理外部 A2A Agent 注册端点：
- 端点 CRUD（name / agent_card_url / base_url / auth_type / credential / status）
- 注册时拉取并缓存 Agent Card（SSRF 防护 + 格式校验）
- 凭证仅存 AES 密文，日志不打印明文

安全：credential 只落密文；Agent Card 拉取目标仅 https 且禁内网。
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.crypto.aes_cipher import encrypt
from app.models.entity.sys_ai_agent_endpoint import SysAiAgentEndpoint
from app.repository.ai_agent_endpoint_repository import (
    ai_agent_endpoint_repository,
)
from app.service.ai.a2a_client import A2AClientError, a2a_client
from app.utils.ssrf import is_safe_url

logger = logging.getLogger(__name__)


class AiAgentEndpointService:
    @staticmethod
    async def create_endpoint(db: AsyncSession, form) -> SysAiAgentEndpoint:
        base_url = form.base_url.rstrip("/")
        if not await is_safe_url(base_url) or not await is_safe_url(form.agent_card_url):
            raise BusinessException(
                ResultCode.PARAM_ERROR, "base_url/agent_card_url 仅支持 https 且禁止内网地址"
            )

        if await ai_agent_endpoint_repository.get_by_base_url(db, base_url):
            raise BusinessException(ResultCode.DATA_EXISTS, "该端点地址已注册")

        credential_cipher = encrypt(form.credential) if form.credential else None
        endpoint = SysAiAgentEndpoint(
            name=form.name,
            agent_card_url=form.agent_card_url,
            base_url=base_url,
            auth_type=form.auth_type,
            credential=credential_cipher,
            status=form.status,
        )
        endpoint = await ai_agent_endpoint_repository.create(db, endpoint)
        # 注册成功后拉取并校验 Agent Card（失败不阻断创建，仅告警）
        await AiAgentEndpointService._refresh_agent_card(db, endpoint.id)
        return endpoint

    @staticmethod
    async def update_endpoint(db: AsyncSession, endpoint_id: int, form) -> SysAiAgentEndpoint:
        endpoint = await ai_agent_endpoint_repository.get_by_id(db, endpoint_id)
        if not endpoint:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "端点不存在")

        data = form.model_dump(exclude_unset=True)
        if "base_url" in data:
            data["base_url"] = data["base_url"].rstrip("/")
            if not await is_safe_url(data["base_url"]):
                raise BusinessException(
                    ResultCode.PARAM_ERROR, "base_url 仅支持 https 且禁止内网地址"
                )
        if "agent_card_url" in data and not await is_safe_url(data["agent_card_url"]):
            raise BusinessException(
                ResultCode.PARAM_ERROR, "agent_card_url 仅支持 https 且禁止内网地址"
            )
        if "credential" in data and data["credential"]:
            data["credential"] = encrypt(data["credential"])
        for field, value in data.items():
            setattr(endpoint, field, value)
        await db.flush()
        await db.refresh(endpoint)
        await AiAgentEndpointService._refresh_agent_card(db, endpoint_id)
        return endpoint

    @staticmethod
    async def delete_endpoint(db: AsyncSession, endpoint_id: int) -> None:
        endpoint = await ai_agent_endpoint_repository.get_by_id(db, endpoint_id)
        if not endpoint:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "端点不存在")
        await ai_agent_endpoint_repository.soft_delete_by_ids(db, [endpoint_id])

    @staticmethod
    async def list_endpoints(
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
    ) -> tuple[list[SysAiAgentEndpoint], int]:
        return await ai_agent_endpoint_repository.paginate_endpoints(
            db, page, size, keyword, status
        )

    # ── Agent Card 拉取缓存 ────────────────────────────────────

    @staticmethod
    async def _refresh_agent_card(db: AsyncSession, endpoint_id: int) -> None:
        """拉取并缓存 Agent Card 到端点记录（拉取失败仅告警）。"""
        endpoint = await ai_agent_endpoint_repository.get_by_id(db, endpoint_id)
        if not endpoint or not endpoint.agent_card_url:
            return
        try:
            card = await a2a_client.fetch_agent_card(endpoint.agent_card_url)
            endpoint.agent_card = card
            await db.flush()
        except A2AClientError as e:
            logger.warning("端点 %s Agent Card 拉取失败: %s", endpoint_id, e)

    @staticmethod
    async def refresh_agent_card(db: AsyncSession, endpoint_id: int) -> dict[str, Any]:
        """手动刷新 Agent Card（校验失败抛错，便于前端感知）。"""
        endpoint = await ai_agent_endpoint_repository.get_by_id(db, endpoint_id)
        if not endpoint:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "端点不存在")
        if not endpoint.agent_card_url:
            raise BusinessException(ResultCode.PARAM_ERROR, "端点未配置 agent_card_url")
        card = await a2a_client.fetch_agent_card(endpoint.agent_card_url)
        endpoint.agent_card = card
        await db.flush()
        await db.refresh(endpoint)
        return card


ai_agent_endpoint_service = AiAgentEndpointService()
