"""KB embedding/rerank 计费（kb_billing_service）与检索/文档链路接入

口径（AI计费管理 后端实现 §KB 计费）：
- 后扣费模式（voice 同款）：调用前 ensure（欠费熔断+配额+余额 fail-closed）、
  调用成功后按实际 token 实扣；调用失败不扣
- 计量：文档向量化=分块精确 token_count 求和；检索查询/重排序=_approx_tokens
  （与 chat 预扣同口径）
- bill_type=embedding/rerank，token 数记 input_tokens；local 模型完全跳过不落记录
- 未配置用户售价 warn 回退 0（免费，防误扣）；实扣不足扣至 0 并标记欠费
- 检索链路计费拒绝（QUOTA_INSUFFICIENT）透传而非降级为空结果
"""

from contextlib import asynccontextmanager
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.repository.ai_billing_repository import AiBillingRepository
from app.repository.ai_credit_log_repository import AiCreditLogRepository
from app.repository.ai_provider_repository import AiProviderRepository
from app.service.ai_model_price_service import AiModelPriceService
from app.service.billing.balance_service import BalanceService
from app.service.billing.estimate_service import _approx_tokens
from app.service.billing.quota_service import QuotaService
from app.service.kb import kb_billing_service as kb_billing_module
from app.service.kb import search_service as search_module
from app.service.kb.kb_billing_service import KbBillingService, kb_billing_service
from app.service.kb.search_service import _run_retrieval, search_service

pytestmark = pytest.mark.requires_db

# 直连测试的 db 占位：检索链路 db 被仓储 mock 覆盖，用真实会话实例满足契约
_DB = AsyncSession()
_QUOTA_CODE = ResultCode.QUOTA_INSUFFICIENT


@asynccontextmanager
async def _fake_session():
    yield _DB


class _BillingRepoStub(AiBillingRepository):
    """测试替身：仅实现 create_billing（委托注入的协程）。"""

    def __init__(self, create_billing):
        self._create_billing = create_billing

    async def create_billing(self, db, **kwargs):
        return await self._create_billing(db, **kwargs)


class _CreditLogRepoStub(AiCreditLogRepository):
    """测试替身：仅实现 create_log（委托注入的协程）。"""

    def __init__(self, create_log):
        self._create_log = create_log

    async def create_log(self, db, **kwargs):
        return await self._create_log(db, **kwargs)


class _ProviderRepoStub(AiProviderRepository):
    """测试替身：仅实现 get_by_provider_code（委托注入的协程）。"""

    def __init__(self, get_provider):
        self._get_provider = get_provider

    async def get_by_provider_code(self, db, provider_code, include_deleted=False):
        return await self._get_provider(db, provider_code)


class _PriceServiceStub(AiModelPriceService):
    """测试替身：仅实现 calculate（委托注入的协程）。"""

    def __init__(self, calculate):
        self._calculate = calculate

    async def calculate(
        self, db, model_id, provider_id, at_time, input_tokens, cached_tokens, output_tokens
    ):
        return await self._calculate(
            db, model_id, provider_id, at_time, input_tokens, cached_tokens, output_tokens
        )


def _install(monkeypatch, *, price=None):
    price = price or {"credits": 42, "credits_saved": 0, "configured": True}
    captured = {
        "create": None,
        "logs": [],
        "quota_deduct": [],
        "balance_deduct": [],
        "calc": None,
    }

    async def _create_billing(db, **kwargs):
        captured["create"] = kwargs
        return SimpleNamespace(id=9)

    async def _create_log(db, **kwargs):
        captured["logs"].append(kwargs)

    async def _deduct_quota(uid, credits):
        captured["quota_deduct"].append((uid, credits))

    async def _deduct_balance(db, uid, credits):
        captured["balance_deduct"].append((uid, credits))

    async def _get_balance(db, uid):
        return 900

    async def _check_balance(db, uid, estimated):
        return True

    async def _calculate(db, model_id, provider_id, at, it, ct, ot):
        captured["calc"] = {"model": model_id, "provider_id": provider_id, "tokens": it}
        return dict(price)

    async def _get_provider(db, code):
        return SimpleNamespace(id=7)

    svc = KbBillingService(
        ai_billing_repository=_BillingRepoStub(_create_billing),
        ai_credit_log_repository=_CreditLogRepoStub(_create_log),
        ai_provider_repository=_ProviderRepoStub(_get_provider),
        ai_model_price_service=_PriceServiceStub(_calculate),
        balance_service=cast(  # 替身：含 AsyncMock 方法（用例会重赋），无法子类化
            BalanceService,
            SimpleNamespace(
                is_arrears=AsyncMock(return_value=False),
                check_balance=AsyncMock(side_effect=_check_balance),
                deduct=_deduct_balance,
                get_balance=_get_balance,
            ),
        ),
        quota_service=cast(  # 替身：仅实现 check_quota/deduct（AsyncMock 特征），无法子类化
            QuotaService,
            SimpleNamespace(
                check_quota=AsyncMock(return_value=True),
                deduct=_deduct_quota,
            ),
        ),
    )
    monkeypatch.setattr(kb_billing_module, "get_db_session", _fake_session)
    return svc, captured


class TestCharge:
    async def test_charge_embedding_records_and_deducts(self, monkeypatch):
        svc, cap = _install(monkeypatch)
        credits = await svc.charge_embedding(10, "openai", "text-embedding-3-small", 1000)

        assert credits == 42
        record = cap["create"]
        assert record["user_id"] == 10
        assert record["model"] == "text-embedding-3-small"
        assert record["bill_type"] == "embedding"
        assert record["input_tokens"] == 1000
        assert record["credits"] == 42
        assert record["quota_consumed"] == 42
        assert record["pre_deduct"] == 0
        assert record["provider_id"] == 7
        assert cap["quota_deduct"] == [(10, 42)]
        assert cap["balance_deduct"] == [(10, 42)]
        assert len(cap["logs"]) == 1
        assert cap["logs"][0]["amount"] == Decimal(-42)
        assert cap["logs"][0]["source"] == "consume"
        assert "向量化" in cap["logs"][0]["reason"]

    async def test_charge_rerank_type_and_reason(self, monkeypatch):
        svc, cap = _install(monkeypatch)
        credits = await svc.charge_rerank(10, "cohere", "rerank-v3", 500)

        assert credits == 42
        assert cap["create"]["bill_type"] == "rerank"
        assert "重排序" in cap["logs"][0]["reason"]

    async def test_local_provider_skipped_entirely(self, monkeypatch):
        """内置本地模型零成本：不落计费记录、不扣减、不写流水"""
        svc, cap = _install(monkeypatch)
        credits = await svc.charge_embedding(10, "local", "qwen3-embedding-0.6b", 500)

        assert credits == 0
        assert cap["create"] is None
        assert cap["quota_deduct"] == []
        assert cap["balance_deduct"] == []
        assert cap["logs"] == []

    async def test_unconfigured_price_falls_back_zero(self, monkeypatch):
        """未配置用户售价：warn 回退 0（免费），仍落归因记录但不扣减不写流水"""
        svc, cap = _install(
            monkeypatch, price={"credits": 0, "credits_saved": 0, "configured": False}
        )
        credits = await svc.charge_embedding(10, "openai", "no-price-model", 800)

        assert credits == 0
        assert cap["create"]["credits"] == 0
        assert cap["create"]["input_tokens"] == 800
        assert cap["quota_deduct"] == []
        assert cap["balance_deduct"] == []
        assert cap["logs"] == []

    async def test_zero_price_configured_no_deduct(self, monkeypatch):
        """全 0 价（免费模型明确配置）：落记录不扣减（与本地 LLM 全 0 价口径一致）"""
        svc, cap = _install(
            monkeypatch, price={"credits": 0, "credits_saved": 0, "configured": True}
        )
        assert await svc.charge_embedding(10, "openai", "free-model", 800) == 0
        assert cap["quota_deduct"] == []
        assert cap["logs"] == []


class TestEnsure:
    async def test_arrears_rejected(self, monkeypatch):
        svc, _ = _install(monkeypatch)
        svc.balance_service.is_arrears = AsyncMock(return_value=True)
        with pytest.raises(BusinessException) as excinfo:
            await svc.ensure(10, "openai", "m", 100)
        assert excinfo.value.code == _QUOTA_CODE

    async def test_quota_fail_closed_rejected(self, monkeypatch):
        """权益缺失/停用（check_quota fail-closed）拒绝，与 chat 口径一致"""
        svc, _ = _install(monkeypatch)
        svc.quota_service.check_quota = AsyncMock(return_value=False)
        with pytest.raises(BusinessException) as excinfo:
            await svc.ensure(10, "openai", "m", 100)
        assert excinfo.value.code == _QUOTA_CODE

    async def test_insufficient_balance_rejected(self, monkeypatch):
        svc, _ = _install(monkeypatch)
        svc.balance_service.check_balance = AsyncMock(return_value=False)
        with pytest.raises(BusinessException) as excinfo:
            await svc.ensure(10, "openai", "m", 100)
        assert excinfo.value.code == _QUOTA_CODE

    async def test_sufficient_passes(self, monkeypatch):
        svc, _cap = _install(monkeypatch)
        await svc.ensure(10, "openai", "text-embedding-3-small", 100)

    async def test_local_still_checked(self, monkeypatch):
        """local 模型不计费但预校验照常执行（欠费/权益缺失与 chat 同口径拦截）"""
        svc, _ = _install(monkeypatch)
        svc.quota_service.check_quota = AsyncMock(return_value=False)
        with pytest.raises(BusinessException) as excinfo:
            await svc.ensure(10, "local", "qwen3-embedding-0.6b", 100)
        assert excinfo.value.code == _QUOTA_CODE


# ── 检索链路接入（search_service → kb_billing_service） ──


def _kb(
    *,
    kb_id: int = 1,
    search_strategy: str = "vector",
    enable_rerank: int = 0,
    rerank_model: str | None = None,
) -> SysKnowledgeBase:
    return SysKnowledgeBase(
        id=kb_id,
        name=f"kb{kb_id}",
        description=None,
        visibility="public",
        create_by=100,
        status=1,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        chunking_strategy="semantic",
        search_strategy=search_strategy,
        chunk_size=800,
        chunk_overlap=80,
        top_k=5,
        score_threshold=0.0,
        enable_rerank=enable_rerank,
        rerank_model=rerank_model,
        hybrid_weight=0.7,
        document_count=0,
        chunk_count=0,
        total_tokens=0,
    )


def _es_doc(*, doc_id: int, chunk_id: int, content: str, relevance: float) -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "doc_title": f"《文档{doc_id}》",
        "chunk_index": 0,
        "content": content,
        "relevance": relevance,
        "metadata": {},
        "content_vector": [0.1] * 8,
    }


class TestSearchBilling:
    def _billing_mocks(self, monkeypatch):
        ensure = AsyncMock()
        charge_embed = AsyncMock()
        charge_rerank = AsyncMock()
        monkeypatch.setattr(kb_billing_service, "ensure", ensure)
        monkeypatch.setattr(kb_billing_service, "charge_embedding", charge_embed)
        monkeypatch.setattr(kb_billing_service, "charge_rerank", charge_rerank)
        return ensure, charge_embed, charge_rerank

    def _kb_repo(self, monkeypatch, kb):
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        kb_repo.list_public.return_value = [kb]
        monkeypatch.setattr(search_module, "knowledge_base_repository", kb_repo)

    async def test_query_embedding_charged_with_approx_tokens(self, monkeypatch, mock_redis):
        self._kb_repo(monkeypatch, _kb())
        ensure, charge_embed, charge_rerank = self._billing_mocks(monkeypatch)
        embed_mock = AsyncMock(return_value=[0.1] * 8)
        with (
            patch("app.service.kb.search_service.embed_text", embed_mock),
            patch(
                "app.service.kb.search_service.kb_chunk_index.vector_search",
                AsyncMock(
                    return_value=[_es_doc(doc_id=1, chunk_id=1, content="命中", relevance=0.9)]
                ),
            ),
        ):
            await search_service.search(_DB, mock_redis, 100, "量子纠缠", knowledge_base_ids=[1])

        expected = _approx_tokens("量子纠缠")
        ensure.assert_awaited_once_with(100, "openai", "text-embedding-3-small", expected)
        charge_embed.assert_awaited_once_with(100, "openai", "text-embedding-3-small", expected)
        charge_rerank.assert_not_awaited()

    async def test_internal_retrieval_without_user_no_billing(self, monkeypatch):
        """user_id=None（对话内部注入）：不计费"""
        kb = _kb()
        ensure, charge_embed, _ = self._billing_mocks(monkeypatch)
        with patch("app.service.kb.search_service.embed_text", AsyncMock(return_value=[0.1] * 8)):
            await _run_retrieval([kb], "量子纠缠", [], 5, False, None)
        ensure.assert_not_awaited()
        charge_embed.assert_not_awaited()

    async def test_cache_hit_no_double_charge(self, monkeypatch, mock_redis):
        self._kb_repo(monkeypatch, _kb())
        _ensure, charge_embed, _ = self._billing_mocks(monkeypatch)
        with (
            patch("app.service.kb.search_service.embed_text", AsyncMock(return_value=[0.1] * 8)),
            patch(
                "app.service.kb.search_service.kb_chunk_index.vector_search",
                AsyncMock(
                    return_value=[_es_doc(doc_id=1, chunk_id=1, content="命中", relevance=0.9)]
                ),
            ),
        ):
            await search_service.search(_DB, mock_redis, 100, "缓存计费", knowledge_base_ids=[1])
            await search_service.search(_DB, mock_redis, 100, "缓存计费", knowledge_base_ids=[1])
        assert charge_embed.await_count == 1

    async def test_rerank_charged_with_query_plus_candidates(self, monkeypatch, mock_redis):
        """rerank 计量 = query + 全部候选文档的输入 token；keyword 策略无 embedding 计费"""
        docs = [
            _es_doc(doc_id=1, chunk_id=1, content="候选一：全文检索命中段落", relevance=0.9),
            _es_doc(doc_id=2, chunk_id=2, content="候选二：语义重排优先段落", relevance=0.8),
        ]
        self._kb_repo(
            monkeypatch, _kb(search_strategy="keyword", enable_rerank=1, rerank_model="rerank-v3")
        )
        ensure, charge_embed, charge_rerank = self._billing_mocks(monkeypatch)
        query = "检索重排"
        with (
            patch(
                "app.service.kb.search_service.kb_chunk_index.keyword_search",
                AsyncMock(return_value=docs),
            ),
            patch(
                "app.service.kb.search_service.rerank",
                AsyncMock(return_value=[{"index": 1}, {"index": 0}]),
            ),
        ):
            await search_service.search(
                _DB,
                mock_redis,
                100,
                query,
                knowledge_base_ids=[1],
                top_k=2,
            )

        expected = _approx_tokens(query) + sum(_approx_tokens(d["content"]) for d in docs)
        charge_embed.assert_not_awaited()
        charge_rerank.assert_awaited_once_with(100, "openai", "rerank-v3", expected)
        ensure.assert_awaited_once_with(100, "openai", "rerank-v3", expected)

    async def test_quota_rejection_propagates_not_degraded(self, monkeypatch, mock_redis):
        """计费拒绝透传为业务异常，而非降级为空结果（防欠费用户误判库无内容）"""
        self._kb_repo(monkeypatch, _kb())
        ensure = AsyncMock(side_effect=BusinessException(ResultCode.QUOTA_INSUFFICIENT, "配额不足"))
        monkeypatch.setattr(kb_billing_service, "ensure", ensure)
        with (
            patch("app.service.kb.search_service.embed_text", AsyncMock(return_value=[0.1] * 8)),
            pytest.raises(BusinessException) as excinfo,
        ):
            await search_service.search(_DB, mock_redis, 100, "配额检索", knowledge_base_ids=[1])
        assert excinfo.value.code == ResultCode.QUOTA_INSUFFICIENT
