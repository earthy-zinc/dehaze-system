"""内置本地模型/推理服务的数据初始化（幂等，由 `python -m app.cli init-builtin` 显式调用）

确保供应商、服务密钥、内置模型记录存在，使本地能力可用（推理由独立服务承担，
此处只维护统一模型注册表）。**不在进程启动时执行**：配置权威源是数据库
（初始化数据见 config/sql/data/），本模块只作为显式初始化入口的实现，且
绝不覆盖用户对已有行的修改（首次插入后即由 DB 与管理员接管）：

- LLM（qwen3-0.6b）：状态启用，供对话路由（平台默认模型来自 `sys_dict`
  `ai_model.default_model`，用户偏好优先，见 service/ai/service/model_defaults.py）使用；
- Embedding（qwen3-embedding-0.6b）：状态停用，仅作统一模型注册表记录，
  不进入对话模型列表；向量调用由 embedding_client 按 sys_ai_provider
  api_base_url 直连 local provider 的 /v1/embeddings 端点，不依赖该记录。
- Rerank（qwen3-reranker-0.6b）：状态停用，仅登记模型目录；重排调用由
  rerank_service 按 api_base_url 直连 /v1/rerank 端点。模型文件为可选懒下载
  （首次重排请求触发），启用与否不影响启动。

阶段 1 起，内置语音（clarity-voice）与图像（clarity-modal）服务也注册为统一模型
供应商（local-voice / local-modal），供迁移入库的 asr/tts/image_* 模型行按类型做
连通性探测（GET {api_base_url}/models）使用。其模型/音色已在阶段 2 迁入统一注册表
（sys_ai_model，model_id 如 sensevoice/paraformer/huayan），运行时工件由
clarity-voice 服务端目录解析，不落模型行。

服务密钥（明文即 `INTERNAL_SERVICE_KEY`）按供应商各登记一行：sys_ai_provider_key
唯一键为 (provider_id, key_hash)，同一把部署级密钥可在 local / local-voice /
local-modal 三行共存；播种幂等，密钥轮换时自动停用同供应商下的旧启用行。
"""

import logging
from datetime import datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.infrastructure.crypto.aes_cipher import encrypt, hash_key
from app.models.entity import SysAiModel, SysAiModelPrice, SysAiProvider, SysAiProviderKey
from app.repository.ai_model_price_repository import ai_model_price_repository
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_key_repository import ai_provider_key_repository
from app.repository.ai_provider_repository import ai_provider_repository

logger = logging.getLogger(__name__)

LOCAL_PROVIDER_CODE = "local"
LOCAL_VOICE_PROVIDER_CODE = "local-voice"
LOCAL_MODAL_PROVIDER_CODE = "local-modal"
LOCAL_MODEL_ID = "qwen3-0.6b"
LOCAL_EMBEDDING_MODEL_ID = "qwen3-embedding-0.6b"
LOCAL_RERANK_MODEL_ID = "qwen3-reranker-0.6b"

# 非 token 模型（asr/tts/image_edit）的**首次播种单价**（设计 §2.6）：asr 1 积分/秒、
# tts 0.01 积分/字符 = 100 积分/万字符（与阶段 2 前的语音计费字典种子等值，该字典已废除）；
# image_edit 为本地低积分标定 1 积分/张。
# 阶段 2 起 `sys_ai_model_price` 是唯一事实源，运营调价改价格版本而非改代码/字典。
ASR_CREDITS_PER_SECOND = Decimal("1")
TTS_CREDITS_PER_CHAR = Decimal("0.01")
# TTS 计费按「万字符」计量：万字符单价 = 每字符单价 × 10000
TTS_CHARS_PER_BILLING_UNIT = 10000
TTS_CREDITS_PER_TEN_THOUSAND_CHARS = TTS_CREDITS_PER_CHAR * TTS_CHARS_PER_BILLING_UNIT
LOCAL_IMAGE_EDIT_CREDITS_PER_IMAGE = Decimal("1")


async def ensure_local_models(db: AsyncSession) -> None:
    """幂等播种 local provider / key / 模型；已存在则按需补齐缺失部分"""
    base_url = f"{settings.LOCAL_LLM_BASE_URL.rstrip('/')}/v1"
    provider = await _ensure_provider(
        db,
        provider_code=LOCAL_PROVIDER_CODE,
        display_name="内置本地模型",
        base_url=base_url,
        protocol_type="openai_compat",
    )
    await _ensure_service_key(db, provider)

    # 对话 LLM：状态启用，供模型路由与降级链使用
    if await ai_model_repository.get_by_model_and_provider(db, LOCAL_MODEL_ID, provider.id) is None:
        db.add(
            SysAiModel(
                provider_id=provider.id,
                model_id=LOCAL_MODEL_ID,
                display_name="Qwen3-0.6B（内置本地）",
                # 与 LOCAL_LLM_CTX_SIZE（本地推理上下文）同源：模型真实上下文 40960，
                # CPU 推理保守取值；写死 8192 会让新环境的上下文预算口径与既有环境不一致
                max_context_tokens=settings.LOCAL_LLM_CTX_SIZE,
                max_output_tokens=2048,
                supports_multimodal=0,
                # 本地推理服务（clarity-llm）不解析工具协议（见其 openai_compat 路由注释），
                # 无法产出原生 tool_calls；标 0 与事实一致，工具走文本协议注入（tool_dialect）
                supports_tool_call=0,
                supports_streaming=1,
                supports_prompt_cache=0,
                supports_structured_output=0,
                prompt_cache_prefix_len=0,
                status=1,
                vip_level=0,
            )
        )
        logger.info("播种 qwen3-0.6b 模型（对话兜底）")
    # 价格幂等补齐须独立于模型记录：模型在早期版本已播种（无价格逻辑）的环境，
    # 否则永远缺价——计费结算抛"未配置用户售价"，direct 轮次整条 trace 丢失
    await _ensure_model_price(db, LOCAL_MODEL_ID, provider.id)

    # Embedding：状态停用，仅登记模型目录（避免出现在对话模型列表）；
    # 实际向量调用由 embedding_client 按 local provider 的 api_base_url 直连。
    # model_type 须显式为 embedding（schema 默认 chat，会污染模型类型筛选与目录展示）
    if (
        await ai_model_repository.get_by_model_and_provider(
            db, LOCAL_EMBEDDING_MODEL_ID, provider.id
        )
        is None
    ):
        db.add(
            SysAiModel(
                provider_id=provider.id,
                model_id=LOCAL_EMBEDDING_MODEL_ID,
                model_type="embedding",
                dimension=1024,
                display_name="Qwen3-Embedding-0.6B（内置本地，向量）",
                max_context_tokens=4096,
                max_output_tokens=0,
                supports_multimodal=0,
                supports_tool_call=0,
                supports_streaming=0,
                supports_prompt_cache=0,
                supports_structured_output=0,
                prompt_cache_prefix_len=0,
                status=0,
                vip_level=0,
            )
        )
        logger.info("播种 qwen3-embedding-0.6b 模型（向量注册表记录）")
    await _ensure_model_price(db, LOCAL_EMBEDDING_MODEL_ID, provider.id)
    # Rerank：状态停用，仅登记模型目录（同 embedding 语义）；实际重排调用由
    # rerank_service 按 local provider 的 api_base_url 直连 /v1/rerank 端点。
    # model_type 须显式为 rerank（schema 默认 chat，会污染模型类型筛选与目录展示）
    if (
        await ai_model_repository.get_by_model_and_provider(db, LOCAL_RERANK_MODEL_ID, provider.id)
        is None
    ):
        db.add(
            SysAiModel(
                provider_id=provider.id,
                model_id=LOCAL_RERANK_MODEL_ID,
                model_type="rerank",
                display_name="Qwen3-Reranker-0.6B（内置本地，重排）",
                max_context_tokens=settings.LOCAL_LLM_CTX_SIZE,
                max_output_tokens=0,
                supports_multimodal=0,
                supports_tool_call=0,
                supports_streaming=0,
                supports_prompt_cache=0,
                supports_structured_output=0,
                prompt_cache_prefix_len=0,
                status=0,
                vip_level=0,
            )
        )
        logger.info("播种 qwen3-reranker-0.6b 模型（重排注册表记录）")
    await _ensure_model_price(db, LOCAL_RERANK_MODEL_ID, provider.id)


def _builtin_infra_providers() -> list[dict[str, str]]:
    """内置 voice/modal 推理服务的供应商规格（api_base_url = 服务基址 + /v1，与 local 同口径）"""
    return [
        {
            "provider_code": LOCAL_VOICE_PROVIDER_CODE,
            "display_name": "内置本地语音",
            "base_url": f"{settings.VOICE_ENGINE_BASE_URL.rstrip('/')}/v1",
            "protocol_type": "openai_audio",
        },
        {
            "provider_code": LOCAL_MODAL_PROVIDER_CODE,
            "display_name": "内置本地图像",
            "base_url": f"{settings.MODAL_ENGINE_BASE_URL.rstrip('/')}/v1",
            "protocol_type": "openai_image",
        },
    ]


async def ensure_local_infra_providers(db: AsyncSession) -> None:
    """幂等初始化内置 voice/modal 供应商及其服务密钥。

    阶段 1 的按类型连通性探测（asr/tts/image_* 走 GET {api_base_url}/models）与
    asr/tts/image_* 模型行都依赖这两个 provider 及其 Key；Key 缺失会让探测 401，
    被误判为「不可用」。

    `VOICE_LOCAL_ENABLED=False`（本部署不提供内置语音引擎）时：不初始化 local-voice，
    并禁用已有行——开关表达部署意图，DB 行状态随之收敛（配置改动仍在 DB 侧可见）。
    """
    for spec in _builtin_infra_providers():
        if spec["provider_code"] == LOCAL_VOICE_PROVIDER_CODE and not settings.VOICE_LOCAL_ENABLED:
            await _disable_provider(db, LOCAL_VOICE_PROVIDER_CODE)
            continue
        provider = await _ensure_provider(db, **spec)
        await _ensure_service_key(db, provider)


async def _disable_provider(db: AsyncSession, provider_code: str) -> None:
    """禁用供应商行（不存在则无操作）；已禁用时幂等返回"""
    provider = await ai_provider_repository.get_by_provider_code(db, provider_code)
    if provider is not None and provider.status != 0:
        provider.status = 0
        await db.flush()
        logger.info("内置供应商 %s 已禁用（部署开关关闭）", provider_code)


async def _ensure_provider(
    db: AsyncSession,
    *,
    provider_code: str,
    display_name: str,
    base_url: str,
    protocol_type: str,
) -> SysAiProvider:
    """幂等确保供应商行存在；已存在则原样返回。

    api_base_url 只在首次插入时写入：运行期改地址应由管理端改 DB，代码不做隐式同步
    （历史实现在此处把用户改过的地址覆盖回环境变量派生值，已移除）。
    """
    provider = await ai_provider_repository.get_by_provider_code(db, provider_code)
    if provider is None:
        provider = SysAiProvider(
            provider_code=provider_code,
            display_name=display_name,
            api_base_url=base_url,
            protocol_type=protocol_type,
            auth_type="bearer",
            status=1,
        )
        db.add(provider)
        await db.flush()
        logger.info("初始化 %s provider（id=%s）", provider_code, provider.id)
    return provider


async def _ensure_service_key(db: AsyncSession, provider: SysAiProvider) -> None:
    """播种供应商的服务密钥（明文即 INTERNAL_SERVICE_KEY）。

    查重按供应商维度（唯一键 (provider_id, key_hash)）：同一把部署级密钥在
    local / local-voice / local-modal 各登记一行。密钥变更自愈——停用同供应商下
    其余启用行（旧 INTERNAL_SERVICE_KEY 已被服务侧拒绝 401，保留会让 Key 轮换
    命中旧行）；停用而非删除（留痕可审计）。
    """
    key_plaintext = settings.INTERNAL_SERVICE_KEY
    key_hash = hash_key(key_plaintext)
    rows = await ai_provider_key_repository.list_by_provider(db, provider.id)
    if any(row.key_hash == key_hash for row in rows):
        return
    for row in rows:
        if row.status == 1:
            row.status = 0
    db.add(
        SysAiProviderKey(
            provider_id=provider.id,
            name="内置服务密钥（INTERNAL_SERVICE_KEY）",
            key_hash=key_hash,
            key_prefix="local",
            key_cipher=encrypt(key_plaintext),
            status=1,
            priority=1,
            weight=1,
        )
    )
    await db.flush()
    logger.info("播种 %s provider 服务密钥（变更时自动停用旧密钥行）", provider.provider_code)


async def _ensure_model_price(
    db: AsyncSession,
    model_id: str,
    provider_id: int,
    *,
    billing_unit: str = "million_tokens",
    usage_kind: str = "token",
    unit_price: Decimal = Decimal("0"),
) -> None:
    """幂等播种模型的一个用户售价版本（token 类全 0 占位 / 非 token 类按计量单位单价）。

    - token 类（billing_unit=million_tokens）：input/cached/output × idle/peak 全 0 价，
      不扣积分处理（见 AI模型管理 §2.12），使本地兜底模型结算不抛"未配置售价"；
    - 非 token 类（asr/tts/image_edit）：单行 `usage_kind='unit'` 档，不设峰谷/分段。
    本阶段只负责播种，非 token 的价格解析（忽略时段/分段）属阶段 2/3 计费改造。
    """
    if (
        await ai_model_price_repository.get_effective_version(
            db, model_id, provider_id, datetime.now()
        )
        is not None
    ):
        return
    # MySQL DATETIME(0) 对微秒四舍五入到秒：xx.7s 会被进位成下一秒，紧随其后的
    # 生效判定（effective_from <= now）反而查不到刚插入的行，幂等检查落空。
    # 落地前截掉微秒，保证生效时间不晚于任何后续查询时刻。
    effective_from = datetime.now().replace(microsecond=0)
    price = await ai_model_price_repository.create(
        db,
        SysAiModelPrice(
            model_id=model_id,
            provider_id=provider_id,
            price_version=1,
            unit="credits_per_million",
            billing_unit=billing_unit,
            effective_from=effective_from,
            status=1,
        ),
    )
    if usage_kind == "unit":
        details = [
            {
                "token_type": "unit",
                "usage_kind": "unit",
                "time_slot": "idle",
                "min_tokens": 0,
                "max_tokens": None,
                "unit_price": unit_price,
            }
        ]
    else:
        details = [
            {
                "token_type": t,
                "usage_kind": "token",
                "time_slot": s,
                "min_tokens": 0,
                "max_tokens": None,
                "unit_price": 0,
            }
            for t in ("input", "cached", "output")
            for s in ("idle", "peak")
        ]
    await ai_model_price_repository.create_details(db, price.id, details)
    logger.info(
        "播种 %s 价格版本（billing_unit=%s, usage_kind=%s, 单价=%s）",
        model_id,
        billing_unit,
        usage_kind,
        unit_price,
    )


async def ensure_local_infra_prices(db: AsyncSession) -> None:
    """幂等播种内置 asr/tts/image_edit 模型的用户售价（非 token 计量）。

    模型按 `provider + model_type` 动态查询（不硬编码 model_id，image_edit 行由迁移写入
    modal provider 下的规范 slug）；单价为首次播种常量（值 = 阶段 2 前的语音计费字典种子：
    asr 1 积分/秒、tts 0.01 积分/字符 = 100 积分/万字符、image_edit 1 积分/张，设计 §2.6）。
    阶段 2 起 `sys_ai_model_price` 是唯一事实源，运营调价改价格版本，不再改字典。
    """
    voice = await ai_provider_repository.get_by_provider_code(db, LOCAL_VOICE_PROVIDER_CODE)
    modal = await ai_provider_repository.get_by_provider_code(db, LOCAL_MODAL_PROVIDER_CODE)
    if voice is None or modal is None:
        # provider 未播种（如播种失败）时无价格可挂，交由下次启动补齐
        return

    for model in await ai_model_repository.list_by_provider_and_type(db, voice.id, "asr"):
        await _ensure_model_price(
            db,
            model.model_id,
            voice.id,
            billing_unit="second",
            usage_kind="unit",
            unit_price=ASR_CREDITS_PER_SECOND,
        )

    for model in await ai_model_repository.list_by_provider_and_type(db, voice.id, "tts"):
        await _ensure_model_price(
            db,
            model.model_id,
            voice.id,
            billing_unit="ten_thousand_chars",
            usage_kind="unit",
            unit_price=TTS_CREDITS_PER_TEN_THOUSAND_CHARS,
        )

    for model in await ai_model_repository.list_by_provider_and_type(db, modal.id, "image_edit"):
        await _ensure_model_price(
            db,
            model.model_id,
            modal.id,
            billing_unit="image",
            usage_kind="unit",
            unit_price=LOCAL_IMAGE_EDIT_CREDITS_PER_IMAGE,
        )
