from __future__ import annotations

import json
import socket
from collections.abc import Callable
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.models.entity.sys_knowledge_document import SysKnowledgeDocument
from app.service.kb.document_service import DocumentService, _clean_text
from app.service.kb.kb_billing_service import kb_billing_service
from app.service.kb.knowledge_base_service import knowledge_base_service
from tests.stubs.fakes import MemberBenefitRepo

pytestmark = pytest.mark.requires_db

# 测试替身：仓储已 mock，db/redis 仅传参占位（用例在触达缓存/事务前即断言或抛错）
_DB: AsyncSession = AsyncMock(spec=AsyncSession)
_REDIS: Redis = AsyncMock(spec=Redis)


CODE_UNAUTHORIZED = ResultCode.ACCESS_UNAUTHORIZED.code
CODE_BUSINESS = ResultCode.BUSINESS_ERROR.code
CODE_PARAM = ResultCode.PARAM_ERROR.code

KS = "app.service.kb.knowledge_base_service"


@pytest.fixture(autouse=True)
def _no_kb_billing(monkeypatch):
    """本文件测文档/知识库流水线本身，不测计费：跳过 KB 计费校验与扣费
    （计费不变量见 test_kb_billing.py；计费拒绝路径为真实行为，勿在本文件断言）"""
    monkeypatch.setattr(kb_billing_service, "ensure", AsyncMock())
    monkeypatch.setattr(kb_billing_service, "charge_embedding", AsyncMock())


def _ctx(user_id: int, *, admin: bool = False) -> SimpleNamespace:
    return SimpleNamespace(id=user_id, username="u", is_admin=admin)


def _kb(
    *,
    kb_id: int = 1,
    name: str = "测试库",
    visibility: str = "private",
    create_by: int = 100,
    status: int = 1,
) -> SysKnowledgeBase:
    return SysKnowledgeBase(
        id=kb_id,
        name=name,
        description=None,
        visibility=visibility,
        create_by=create_by,
        status=status,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        chunking_strategy="semantic",
        search_strategy="hybrid",
        chunk_size=800,
        chunk_overlap=80,
        top_k=5,
        score_threshold=0.5,
        enable_rerank=0,
        rerank_model=None,
        hybrid_weight=0.5,
        document_count=0,
        chunk_count=0,
        total_tokens=0,
    )


def _doc(
    *,
    doc_id: int = 1,
    kb_id: int = 1,
    file_id: int | None = 10,
    status: str = "completed",
    version: int = 1,
    content: str | None = "正文",
) -> SysKnowledgeDocument:
    return SysKnowledgeDocument(
        id=doc_id,
        knowledge_base_id=kb_id,
        file_id=file_id,
        title="文档",
        processing_status=status,
        version=version,
        content=content,
        parsing_strategy="auto",
    )


def _create_data(**over):
    base = {
        "name": "测试库",
        "visibility": "private",
        "embedding_model": "text-embedding-3-small",
        "chunking_strategy": "semantic",
    }
    base.update(over)
    return base


def _registry_patches(
    *,
    models: list | None = None,
    model_id: str = "text-embedding-3-small",
    provider_id: int = 1,
    dimension: int | None = 1536,
    providers: dict | None = None,
):
    """构造模型注册表桩：list_enabled_models 返回启用 embedding 模型；
    供应商按 id 映射 code（对齐种子 openai=1/local=77）
    """
    if models is None:
        models = [
            SimpleNamespace(
                model_id=model_id,
                model_type="embedding",
                provider_id=provider_id,
                dimension=dimension,
            )
        ]
    if providers is None:
        providers = {"openai": 1, "local": 77}
    code_by_id = {pid: code for code, pid in providers.items()}
    fake_model_service = SimpleNamespace(list_enabled_models=AsyncMock(return_value=models))
    fake_provider_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            side_effect=lambda db, pid: (
                SimpleNamespace(id=pid, provider_code=code_by_id[pid])
                if pid in code_by_id
                else None
            )
        )
    )
    return (
        patch(f"{KS}.ai_model_service", fake_model_service),
        patch(f"{KS}.ai_provider_repository", fake_provider_repo),
    )


def _enter(patches):
    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


class _FakeChunk:
    def __init__(self, idx: int, content: str, tokens: int):
        self.metadata = {"chunk_index": idx}
        self.content = content
        self.token_count = tokens
        self.id = 100 + idx
        self.create_time = None
        # 父子分块小节归属（写入侧 MySQL/ES 直传）
        self.section_index = 0
        self.section_path = None


class _FakeEmbedding(ModuleType):
    """测试替身：真实 embedding_client 契约（ModuleType）子类，仅实现 embed_texts。"""

    def __init__(
        self, side_effect: Callable[[], BaseException] | BaseException | None = None
    ) -> None:
        super().__init__("_fake_embedding")
        self._side_effect = side_effect

    async def embed_texts(self, provider, model, texts, batch):
        if self._side_effect is not None:
            if callable(self._side_effect):
                raise self._side_effect()
            raise self._side_effect
        return [[0.1] * 8 for _ in texts]


def _default_chunks():
    return [
        _FakeChunk(0, "第一章：去雾平台整体架构与部署要求", 24),
        _FakeChunk(1, "第二章：快速上手指南与常见问题排查", 26),
    ]


_DIRTY_DOC = (
    "\ufeff# 去雾平台用户手册\r\n"
    "版本说明：本文档适用于V2.0。\u200b请先阅读【快速开始】章节，\n"
    "再按需调用接口，例如 POST /api/v1/knowledge-bases。\r\n"
    "\r\n"
    "\r\n"
    "注意事项：\n"
    "<script>alert('x')</script>禁止上传含敏感信息的文件。\n"
    "\n\n\n"
    "（支持）请联系 admin@dehaze.local，电话 400-0000-0000。"
)


def _pipeline_env(
    *,
    content: str = "原始正文",
    chunks: list | None = None,
    embed_side_effect=None,
    es_return: bool = True,
    stats_cas_side_effect=None,
):
    kb_repo = AsyncMock()
    kb_repo.get_by_id.return_value = _kb(create_by=100)
    if stats_cas_side_effect is not None:
        kb_repo.update_stats_cas = AsyncMock(side_effect=stats_cas_side_effect)
    else:
        kb_repo.update_stats_cas = AsyncMock(return_value=True)
    doc_repo = AsyncMock()
    doc = SysKnowledgeDocument(
        id=1,
        knowledge_base_id=1,
        file_id=10,
        title="去雾平台用户手册",
        processing_status="pending",
        version=1,
        content=content,
        parsing_strategy="auto",
    )
    doc_repo.get_by_id.return_value = doc
    chunk_repo = AsyncMock()
    ce = MagicMock()
    ce.chunk_text.return_value = chunks if chunks is not None else _default_chunks()
    emb = _FakeEmbedding(side_effect=embed_side_effect)
    bulk_mock = AsyncMock(return_value=es_return)
    svc = DocumentService(
        knowledge_base_repository=kb_repo,
        knowledge_document_repository=doc_repo,
        knowledge_chunk_repository=chunk_repo,
        chunking_engine=ce,
        embedding_client=emb,
    )
    ds = "app.service.kb.document_service"
    patches = (
        patch(f"{ds}.bulk_index_chunks", bulk_mock),
        patch(f"{ds}._push_ws", AsyncMock()),
    )
    refs = {
        "kb_repo": kb_repo,
        "chunk_repo": chunk_repo,
        "bulk_mock": bulk_mock,
        "doc": doc,
        "svc": svc,
    }
    return patches, refs


class TestKBPermissionMatrix:
    async def test_edit_private_kb_of_others_raises(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(visibility="private", create_by=200)
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.update(_DB, _REDIS, 1, {"name": "篡改"}, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_delete_private_kb_of_others_raises(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(visibility="private", create_by=200)
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.delete(_DB, _REDIS, 1, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_get_private_kb_detail_of_others_raises(self):
        redis = AsyncMock()
        redis.get.return_value = None
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(visibility="private", create_by=200)
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.get_detail(_DB, redis, 1, 100)
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_public_kb_manageable_only_by_admin(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id = AsyncMock(return_value=_kb(visibility="public", create_by=200))
        kb_repo.get_by_name_and_owner = AsyncMock(return_value=None)
        kb_repo.update = AsyncMock()
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.update(_DB, AsyncMock(), 1, {"name": "x"}, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED
            await knowledge_base_service.update(
                _DB, AsyncMock(), 1, {"name": "x"}, _ctx(999, admin=True)
            )
            kb_repo.update.assert_awaited_once()

    async def test_public_kb_cache_hit_skips_vo_build(self):
        redis = AsyncMock()
        redis.get.return_value = json.dumps(
            {"id": 3, "name": "平台公共库", "visibility": "public", "documentCount": 12},
            ensure_ascii=False,
        )
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(kb_id=3, visibility="public", create_by=200)
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            result = await knowledge_base_service.get_detail(_DB, redis, 3, 100)
            assert result["id"] == 3
            assert result["name"] == "平台公共库"

    async def test_private_kb_cache_hit_still_denied_for_others(self):
        # 缓存为库级共享键：owner 首次访问写入缓存后，他人访问必须仍被可见性校验拦截
        redis = AsyncMock()
        redis.get.return_value = json.dumps(
            {"id": 5, "name": "他人私有库", "visibility": "private"},
            ensure_ascii=False,
        )
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(kb_id=5, visibility="private", create_by=200)
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.get_detail(_DB, redis, 5, 100)
            assert excinfo.value.code.code == CODE_UNAUTHORIZED


class TestKBQuotaBoundary:
    def _create_with_limit(self, current: int, level: str = "level_0"):
        kb_repo = AsyncMock()
        kb_repo.count_private_by_owner = AsyncMock(return_value=current)
        kb_repo.get_by_name_and_owner = AsyncMock(return_value=None)
        kb_repo.create = AsyncMock(return_value=_kb(kb_id=9))
        member_repo = MemberBenefitRepo(member=SimpleNamespace(level_code=level))
        patches = (
            patch(f"{KS}.knowledge_base_repository", kb_repo),
            patch(f"{KS}.member_repository", member_repo),
            patch(f"{KS}.ensure_kb_index", AsyncMock(return_value=True)),
            *_registry_patches(),
        )
        return kb_repo, patches

    async def test_normal_user_at_limit_rejected(self):
        _, patches = self._create_with_limit(current=3)
        with _enter(patches):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.create(_DB, _REDIS, _create_data(), _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS
            assert "升级" in excinfo.value.message

    async def test_normal_user_just_below_limit_succeeds(self):
        _, patches = self._create_with_limit(current=2)
        with _enter(patches):
            kb_id = await knowledge_base_service.create(_DB, AsyncMock(), _create_data(), _ctx(100))
            assert kb_id == 9

    async def test_vip_higher_limit_allows_more(self):
        _, patches = self._create_with_limit(current=5, level="level_2")
        with _enter(patches):
            kb_id = await knowledge_base_service.create(_DB, AsyncMock(), _create_data(), _ctx(100))
            assert kb_id == 9

    async def test_public_kb_not_counted_in_quota(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_name_and_owner = AsyncMock(return_value=None)
        kb_repo.create = AsyncMock(return_value=_kb(kb_id=11, visibility="public"))
        patches = (
            patch(f"{KS}.knowledge_base_repository", kb_repo),
            patch(f"{KS}.ensure_kb_index", AsyncMock(return_value=True)),
            *_registry_patches(),
        )
        with _enter(patches):
            result = await knowledge_base_service.create(
                _DB, AsyncMock(), _create_data(visibility="public"), _ctx(100, admin=True)
            )
            assert result == 11
            kb_repo.count_private_by_owner.assert_not_called()
            kb_repo.create.assert_awaited_once()


class TestEmbeddingModelRegistry:
    """创建知识库时 embedding 模型/供应商/维度以 sys_ai_model 注册表为准（T-KB-009d）"""

    def _create_env(self, *, current: int = 0):
        kb_repo = AsyncMock()
        kb_repo.count_private_by_owner = AsyncMock(return_value=current)
        kb_repo.get_by_name_and_owner = AsyncMock(return_value=None)
        kb_repo.create = AsyncMock(return_value=_kb(kb_id=21))
        ensure_index = AsyncMock(return_value=True)
        patches = (
            patch(f"{KS}.knowledge_base_repository", kb_repo),
            patch(
                f"{KS}.member_repository",
                MemberBenefitRepo(member=SimpleNamespace(level_code="level_0")),
            ),
            patch(f"{KS}.ensure_kb_index", ensure_index),
            *_registry_patches(),
        )
        return patches, ensure_index

    async def test_registry_model_accepted_with_registry_dimension(self):
        # 注册表内模型通过；ES 索引维度取注册表 dimension（bge-m3=1024）
        patches, ensure_index = self._create_env()
        registry = _registry_patches(model_id="bge-m3", provider_id=77, dimension=1024)
        with _enter(patches + registry):
            kb_id = await knowledge_base_service.create(
                _DB,
                AsyncMock(),
                _create_data(embedding_model="bge-m3"),
                _ctx(100),
            )
            assert kb_id == 21
            ensure_index.assert_awaited_once_with(21, 1024)

    async def test_provider_derived_from_registry(self):
        # 供应商由注册表模型所属行推导写入 KB 记录（前端不传、无 openai 默认值）
        patches, _ = self._create_env()
        kb_repo = AsyncMock()
        kb_repo.count_private_by_owner = AsyncMock(return_value=0)
        kb_repo.get_by_name_and_owner = AsyncMock(return_value=None)
        kb_repo.create = AsyncMock(return_value=_kb(kb_id=22))
        captured = {}
        kb_repo.create.side_effect = lambda db, kb: captured.update(kb=kb) or _kb(kb_id=22)
        patches = (
            patch(f"{KS}.knowledge_base_repository", kb_repo),
            patch(
                f"{KS}.member_repository",
                MemberBenefitRepo(member=SimpleNamespace(level_code="level_0")),
            ),
            patch(f"{KS}.ensure_kb_index", AsyncMock(return_value=True)),
            *_registry_patches(model_id="bge-m3", provider_id=77),
        )
        with _enter(patches):
            await knowledge_base_service.create(
                _DB, AsyncMock(), _create_data(embedding_model="bge-m3"), _ctx(100)
            )
            assert captured["kb"].embedding_provider == "local"

    async def test_unknown_model_rejected(self):
        patches, _ = self._create_env()
        registry = _registry_patches(models=[])  # 注册表为空 = 模型不存在
        with _enter(patches + registry):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.create(
                    _DB, AsyncMock(), _create_data(embedding_model="not-in-registry"), _ctx(100)
                )
            assert excinfo.value.code.code == CODE_PARAM

    async def test_disabled_model_rejected(self):
        # list_enabled_models 仅返回启用模型，禁用模型不出现在列表 → 同样拒绝
        patches, _ = self._create_env()
        registry = _registry_patches(
            models=[
                SimpleNamespace(
                    model_id="other-m3", model_type="embedding", provider_id=77, dimension=1024
                )
            ]
        )
        with _enter(patches + registry):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.create(
                    _DB, AsyncMock(), _create_data(embedding_model="bge-m3"), _ctx(100)
                )
            assert excinfo.value.code.code == CODE_PARAM

    async def test_provider_missing_in_registry_rejected(self):
        # 注册表模型行的供应商不存在（数据异常）→ 拒绝
        patches, _ = self._create_env()
        registry = _registry_patches(model_id="text-embedding-3-small", provider_id=999)
        with _enter(patches + registry):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.create(_DB, AsyncMock(), _create_data(), _ctx(100))
            assert excinfo.value.code.code == CODE_PARAM

    async def test_model_without_dimension_rejected(self):
        patches, _ = self._create_env()
        registry = _registry_patches(dimension=None)
        with _enter(patches + registry):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.create(_DB, AsyncMock(), _create_data(), _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS


class TestDocPermissionMatrix:
    @staticmethod
    def _svc(kb: SysKnowledgeBase, doc_repo=None, file_svc=None):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = kb
        return DocumentService(
            knowledge_base_repository=kb_repo,
            knowledge_document_repository=doc_repo or AsyncMock(),
            file_service=file_svc or AsyncMock(),
        )

    async def test_upload_to_others_private_kb_denied(self):
        svc = self._svc(_kb(visibility="private", create_by=200))
        with pytest.raises(BusinessException) as excinfo:
            await svc.upload(_DB, _REDIS, 1, 1, None, _ctx(100))
        assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_list_docs_in_others_private_kb_denied(self):
        svc = self._svc(_kb(visibility="private", create_by=200))
        with pytest.raises(BusinessException) as excinfo:
            await svc.get_page(_DB, 1, None, 1, 20, _ctx(100))
        assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_delete_doc_in_others_private_kb_denied(self):
        doc_repo = AsyncMock()
        doc_repo.get_by_id.return_value = _doc(kb_id=1)
        svc = self._svc(_kb(visibility="private", create_by=200), doc_repo=doc_repo)
        with pytest.raises(BusinessException) as excinfo:
            await svc.delete(_DB, _REDIS, 7, _ctx(100))
        assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_reprocess_doc_in_others_private_kb_denied(self):
        doc_repo = AsyncMock()
        doc_repo.get_by_id.return_value = _doc(kb_id=1, status="failed")
        svc = self._svc(_kb(visibility="private", create_by=200), doc_repo=doc_repo)
        with pytest.raises(BusinessException) as excinfo:
            await svc.reprocess(_DB, _REDIS, 7, _ctx(100))
        assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_public_kb_docs_readable_by_others(self):
        doc_repo = AsyncMock()
        doc_repo.paginate_by_kb.return_value = ([], 0)
        svc = self._svc(_kb(visibility="public", create_by=200), doc_repo=doc_repo)
        result = await svc.get_page(_DB, 1, None, 1, 20, _ctx(100))
        assert "list" in result


class TestDocStatusMachine:
    def _build(self, doc: SysKnowledgeDocument, kb: SysKnowledgeBase | None = None):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = kb or _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_id.return_value = doc
        chunk_repo = AsyncMock()
        svc = DocumentService(
            knowledge_base_repository=kb_repo,
            knowledge_document_repository=doc_repo,
            knowledge_chunk_repository=chunk_repo,
        )
        return svc, kb_repo, doc_repo, chunk_repo

    async def test_delete_processing_doc_denied(self):
        svc, _kb_repo, doc_repo, _ = self._build(_doc(status="processing"))
        with patch("app.service.kb.document_service.delete_doc_chunks", AsyncMock()) as del_es:
            with pytest.raises(BusinessException) as excinfo:
                await svc.delete(_DB, _REDIS, 7, _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS
            doc_repo.soft_delete_by_ids.assert_not_called()
            del_es.assert_not_called()

    async def test_update_processing_doc_denied(self):
        svc, _kb_repo, _doc_repo, _ = self._build(_doc(status="processing"))
        with pytest.raises(BusinessException) as excinfo:
            await svc.update_document(_DB, _REDIS, 7, None, "新内容", _ctx(100))
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_reprocess_completed_doc_denied(self):
        svc, _kb_repo, _doc_repo, _ = self._build(_doc(status="completed"))
        with pytest.raises(BusinessException) as excinfo:
            await svc.reprocess(_DB, _REDIS, 7, _ctx(100))
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_reprocess_pending_doc_denied(self):
        svc, _kb_repo, _doc_repo, _ = self._build(_doc(status="pending"))
        with pytest.raises(BusinessException) as excinfo:
            await svc.reprocess(_DB, _REDIS, 7, _ctx(100))
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_reprocess_failed_doc_clears_chunks_and_es(self, db):
        svc, _kb_repo, _doc_repo, chunk_repo = self._build(_doc(status="failed"))
        redis = AsyncMock()
        with patch("app.service.kb.document_service.delete_doc_chunks", AsyncMock()) as del_es:
            result = await svc.reprocess(db, redis, 7, _ctx(100))
            assert result["document_id"] == 7
            assert result["kb_id"] == 1
            chunk_repo.delete_by_document.assert_awaited_once_with(db, 7)
            del_es.assert_awaited_once_with(1, 7)

    async def test_update_document_version_increments_and_clears_chunks(self, db):
        svc, _kb_repo, _doc_repo, chunk_repo = self._build(_doc(version=3, status="completed"))
        redis = AsyncMock()
        with patch("app.service.kb.document_service.delete_doc_chunks", AsyncMock()) as del_es:
            result = await svc.update_document(db, redis, 7, None, "新版本内容", _ctx(100))
            assert result["version"] == 4
            chunk_repo.delete_by_document.assert_awaited_once_with(db, 7)
            del_es.assert_awaited_once_with(1, 7)


class TestCleanText:
    def test_html_crlf_whitespace_normalized(self):
        raw = "  <b>加粗</b>  第一行\r\n第二行\r第三行\n\n\n\n第四行  \t 尾部  "
        cleaned = _clean_text(raw)
        assert cleaned == "加粗 第一行\n第二行\n第三行\n\n第四行 尾部"
        assert "\r" not in cleaned
        assert "<b>" not in cleaned

    def test_control_chars_removed(self):
        assert _clean_text("a\x00b\x1f c") == "ab c"

    def test_bom_zero_width_fullhalf_punct_content_preserved(self):
        raw = (
            "\ufeff## 去雾平台手册\r\n"
            "注意\u200b：混合, 全角、半角；标点。\n"
            "超长无分隔行" + "A" * 3000 + "\n\n\n结尾"
        )
        cleaned = _clean_text(raw)
        assert "\ufeff" not in cleaned
        assert "\u200b" not in cleaned
        assert "去雾平台手册" in cleaned
        assert "混合" in cleaned
        assert "结尾" in cleaned
        assert "\r" not in cleaned
        assert "\n\n\n" not in cleaned

    def test_long_unbroken_line_survives(self):
        line = "abcdefghABCDEFGH0123456789" * 400
        assert _clean_text(line) == line

    def test_emoji_preserved_no_control_residue(self):
        raw = "🌫️ 去雾算法😊：把雾图🚀还原为清晰图\n\t  \r\n尾部\u200b空白"
        cleaned = _clean_text(raw)
        # emoji 作为可见字符保留，零宽/控制/NBSP 字符被剥离
        assert "🌫️" in cleaned
        assert "😊" in cleaned
        assert "🚀" in cleaned
        assert "\u200b" not in cleaned  # 零宽空格
        assert "\r" not in cleaned
        # 不间断空格 \xa0 作为不可见字符被清除（避免污染 embedding/ES 分词）
        assert "\u00a0" not in _clean_text("a\u00a0b")
        # 无控制字符残留（不含零宽/控制字符）
        assert all(ord(c) >= 32 or c == "\n" for c in cleaned)

    def test_combined_dirty_corpus_normalized(self):
        # 全半角混杂 + CRLF + BOM + 零宽 + emoji + 连续空白 组合对抗
        raw = (
            "\ufeff【标题】去雾平台Ｖ２．０\r\n"
            "混合，全角、半角；标点。emoji🌟测试\u200b零宽\n"
            "连续空白    多\r\n\r\n\r\n结尾空白  \t "
        )
        cleaned = _clean_text(raw)
        assert "\ufeff" not in cleaned
        assert "\u200b" not in cleaned
        assert "\r" not in cleaned
        assert "【标题】" in cleaned
        assert "去雾平台" in cleaned
        assert "🌟" in cleaned
        # 行尾/行内多余空白压缩，连续空行折叠为单个
        assert "\n\n\n" not in cleaned
        assert not cleaned.endswith(" ")


class TestCreateTextBoundary:
    @staticmethod
    def _svc(count: int = 0):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.count_by_kb.return_value = count
        svc = DocumentService(
            knowledge_base_repository=kb_repo,
            knowledge_document_repository=doc_repo,
        )
        return svc, doc_repo

    async def test_empty_content_rejected(self):
        # 空内容（仅空白）入口直接拒绝，不入库
        svc, doc_repo = self._svc()
        redis = AsyncMock()
        for payload in ["", "   ", "\n\t "]:
            with pytest.raises(BusinessException) as excinfo:
                await svc.create_text(_DB, redis, 1, "空正文", payload, _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS
            assert "内容不能为空" in excinfo.value.message
        doc_repo.create.assert_not_called()

    async def test_overlong_content_rejected(self):
        # 字符数超过上限（>1_000_000）入口拒绝，防极端输入撑爆单库
        from app.service.kb.document_service import KB_MAX_TEXT_CHARS

        long_content = "去" * (KB_MAX_TEXT_CHARS + 1)
        svc, doc_repo = self._svc()
        redis = AsyncMock()
        with pytest.raises(BusinessException) as excinfo:
            await svc.create_text(_DB, redis, 1, "超长正文", long_content, _ctx(100))
        assert excinfo.value.code.code == CODE_BUSINESS
        assert "内容长度超过上限" in excinfo.value.message
        doc_repo.create.assert_not_called()

    async def test_create_text_private_kb_of_others_rejected(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(visibility="private", create_by=200)
        svc = DocumentService(
            knowledge_base_repository=kb_repo, knowledge_document_repository=AsyncMock()
        )
        redis = AsyncMock()
        with pytest.raises(BusinessException) as excinfo:
            await svc.create_text(_DB, redis, 1, "越权", "正文", _ctx(100))
        assert excinfo.value.code.code == CODE_UNAUTHORIZED


class TestDocIdempotency:
    def _build(self, existing_doc: SysKnowledgeDocument | None = None):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_file_id.return_value = existing_doc
        doc_repo.count_by_kb.return_value = 0
        doc_repo.create.return_value = _doc(doc_id=8, kb_id=2)
        chunk_repo = AsyncMock()
        svc = DocumentService(
            knowledge_base_repository=kb_repo,
            knowledge_document_repository=doc_repo,
            knowledge_chunk_repository=chunk_repo,
        )
        return svc, kb_repo, doc_repo, chunk_repo

    async def test_duplicate_file_in_same_kb_denied(self):
        svc, _kb_repo, doc_repo, _ = self._build(existing_doc=_doc())
        redis = AsyncMock()
        with pytest.raises(BusinessException) as excinfo:
            await svc.upload(_DB, redis, 1, 42, None, _ctx(100))
        assert excinfo.value.code.code == CODE_BUSINESS
        doc_repo.create.assert_not_called()

    async def test_same_file_in_different_kb_allowed(self):
        svc, _kb_repo, _doc_repo, _ = self._build(existing_doc=None)
        redis = AsyncMock()
        file_service = AsyncMock()
        file_service.get_file_by_id = AsyncMock(
            return_value=SimpleNamespace(name="a.pdf", create_by=100)
        )
        svc.file_service = file_service
        result = await svc.upload(_DB, redis, 2, 42, None, _ctx(100))
        assert result["document_id"] == 8


class TestDocCountQuota:
    @staticmethod
    def _svc(count: int):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_file_id.return_value = None
        doc_repo.count_by_kb.return_value = count
        svc = DocumentService(
            knowledge_base_repository=kb_repo,
            knowledge_document_repository=doc_repo,
        )
        return svc, doc_repo

    async def test_at_limit_500_rejected(self):
        svc, _ = self._svc(500)
        with pytest.raises(BusinessException) as excinfo:
            await svc.upload(_DB, _REDIS, 1, 1, None, _ctx(100))
        assert excinfo.value.code.code == CODE_BUSINESS
        assert "500" in excinfo.value.message

    async def test_just_below_limit_allowed(self):
        svc, doc_repo = self._svc(499)
        doc_repo.create.return_value = _doc(doc_id=99)
        redis = AsyncMock()
        file_service = AsyncMock()
        file_service.get_file_by_id = AsyncMock(
            return_value=SimpleNamespace(name="a.pdf", create_by=100)
        )
        svc.file_service = file_service
        result = await svc.upload(_DB, redis, 1, 1, None, _ctx(100))
        assert result["document_id"] == 99


class TestDocPipelineFailure:
    async def test_chunk_over_limit_raises_no_stats(self, db):
        big_chunks = [_FakeChunk(i, f"第{i}段内容", 12) for i in range(10001)]
        patches, refs = _pipeline_env(chunks=big_chunks)
        with _enter(patches):
            with pytest.raises(BusinessException) as excinfo:
                await refs["svc"]._process_document(1, 1, 100)
            assert excinfo.value.code.code == CODE_BUSINESS
            assert "上限" in excinfo.value.message
        refs["kb_repo"].update_stats_cas.assert_not_called()

    async def test_embedding_failure_exhausts_retry(self, db):
        from app.config import settings

        call_count = {"n": 0}

        def _raise(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("embedding 服务不可用")

        patches, refs = _pipeline_env(embed_side_effect=_raise)
        with _enter(patches), pytest.raises(RuntimeError):
            await refs["svc"]._process_document(1, 1, 100)
        assert call_count["n"] == 1 + settings.KB_ASYNC_MAX_RETRY
        refs["kb_repo"].update_stats_cas.assert_not_called()

    async def test_es_bulk_failure_exhausts_retry(self, db):
        from app.config import settings

        patches, refs = _pipeline_env(es_return=False)
        with _enter(patches), pytest.raises(RuntimeError):
            await refs["svc"]._process_document(1, 1, 100)
        assert refs["bulk_mock"].call_count == 1 + settings.KB_ASYNC_MAX_RETRY
        refs["kb_repo"].update_stats_cas.assert_not_called()

    async def test_success_path_cleans_dirty_text_writes_chunks_and_stats(self, db):
        patches, refs = _pipeline_env(content=_DIRTY_DOC)
        with _enter(patches):
            await refs["svc"]._process_document(1, 1, 100)
        doc = refs["doc"]
        assert doc.processing_status == "completed"
        assert doc.chunk_count == 2
        assert "<script>" not in doc.content
        assert "\r" not in doc.content
        assert "\n\n\n" not in doc.content
        assert "去雾平台用户手册" in doc.content
        assert "400-0000-0000" in doc.content
        refs["chunk_repo"].create_all.assert_awaited_once()
        refs["kb_repo"].update_stats_cas.assert_awaited_once()


class TestKBCasRetry:
    async def test_cas_conflict_then_success(self, db):
        patches, refs = _pipeline_env(stats_cas_side_effect=[False, False, True])
        with _enter(patches):
            await refs["svc"]._process_document(1, 1, 100)
        assert refs["kb_repo"].update_stats_cas.call_count == 3
        assert refs["doc"].processing_status == "completed"


class TestKBAdminListView:
    """view=admin 管理端视角：全量含私有库、普通用户可见性隔离"""

    async def test_admin_view_returns_all_including_private(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.paginate_all.return_value = (
            [_kb(kb_id=1, visibility="private", create_by=100), _kb(kb_id=2, visibility="public")],
            2,
        )
        with (
            patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo),
        ):
            result = await knowledge_base_service.get_page(
                _DB, mock_redis, 100, None, 1, 10, view="admin"
            )
        kb_repo.paginate_all.assert_awaited_once_with(_DB, None, 1, 10)
        kb_repo.paginate_visible.assert_not_called()
        assert result["total"] == 2
        assert [item["id"] for item in result["list"]] == [1, 2]
        # 私有库进入 admin 列表
        assert result["list"][0]["visibility"] == "private"

    async def test_admin_view_uses_global_cache_key(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.paginate_all.return_value = ([_kb(kb_id=1)], 1)
        with (
            patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo),
        ):
            await knowledge_base_service.get_page(_DB, mock_redis, 100, None, 1, 10, view="admin")
        assert await mock_redis.exists("kb:list:admin")
        assert not await mock_redis.exists("kb:list:100")

    async def test_normal_view_still_filters_by_visibility(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.paginate_visible.return_value = ([_kb(kb_id=1, visibility="public")], 1)
        with (
            patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo),
        ):
            result = await knowledge_base_service.get_page(_DB, mock_redis, 100, None, 1, 10)
        kb_repo.paginate_visible.assert_awaited_once_with(_DB, 100, None, 1, 10)
        kb_repo.paginate_all.assert_not_called()
        assert result["total"] == 1


class TestKBIndexStats:
    """索引状态：正常/索引不存在降级/知识库不存在"""

    async def test_returns_size_doc_count_and_warning(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        from app.config import settings

        under = settings.KB_INDEX_WARNING_THRESHOLD - 1
        with (
            patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.knowledge_base_service.get_index_stats") as stats,
        ):
            stats.return_value = {"index_size": under, "index_doc_count": 5}
            result = await knowledge_base_service.get_index_stats(_DB, 1)
        assert result == {
            "index_size": under,
            "index_doc_count": 5,
            "threshold_warning": False,
        }

    async def test_at_or_above_threshold_warns(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        from app.config import settings

        threshold = settings.KB_INDEX_WARNING_THRESHOLD
        with (
            patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.knowledge_base_service.get_index_stats") as stats,
        ):
            stats.return_value = {"index_size": threshold, "index_doc_count": 0}
            result = await knowledge_base_service.get_index_stats(_DB, 1)
        assert result["threshold_warning"] is True

    async def test_missing_index_degrades_to_zero(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        with (
            patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.knowledge_base_service.get_index_stats") as stats,
        ):
            stats.return_value = {"index_size": 0, "index_doc_count": 0}
            result = await knowledge_base_service.get_index_stats(_DB, 1)
        assert result == {"index_size": 0, "index_doc_count": 0, "threshold_warning": False}

    async def test_nonexistent_kb_raises_not_found(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = None
        with (
            patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo),
            pytest.raises(BusinessException) as excinfo,
        ):
            await knowledge_base_service.get_index_stats(_DB, 999)
        assert excinfo.value.code.code == ResultCode.RESOURCE_NOT_FOUND.code


class TestFileOwnership:
    """文档关联文件归属校验（B0407 口径）：非 admin 仅本人文件，admin 全量放行"""

    @staticmethod
    def _svc_with_file(create_by: int, kb_create_by: int = 100):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=kb_create_by)
        doc_repo = AsyncMock()
        doc_repo.get_by_file_id.return_value = None
        doc_repo.count_by_kb.return_value = 0
        doc_repo.create.return_value = _doc(doc_id=21)
        file_svc = AsyncMock()
        file_svc.get_file_by_id = AsyncMock(
            return_value=SimpleNamespace(name="a.pdf", create_by=create_by)
        )
        svc = DocumentService(
            knowledge_base_repository=kb_repo,
            knowledge_document_repository=doc_repo,
            file_service=file_svc,
        )
        return svc, doc_repo

    async def test_upload_other_users_file_denied(self):
        svc, doc_repo = self._svc_with_file(create_by=200)
        with pytest.raises(BusinessException) as excinfo:
            await svc.upload(_DB, AsyncMock(), 1, 42, None, _ctx(100))
        assert excinfo.value.code.code == ResultCode.FILE_ACCESS_DENIED.code
        doc_repo.create.assert_not_called()

    async def test_upload_own_file_allowed(self):
        svc, _ = self._svc_with_file(create_by=100)
        result = await svc.upload(_DB, AsyncMock(), 1, 42, None, _ctx(100))
        assert result["document_id"] == 21

    async def test_upload_other_users_file_by_admin_allowed(self):
        # admin 拥有的私有库引用他人上传的文件：文件归属校验放行（私有库归属校验另行生效）
        svc, _ = self._svc_with_file(create_by=200, kb_create_by=999)
        result = await svc.upload(_DB, AsyncMock(), 1, 42, None, _ctx(999, admin=True))
        assert result["document_id"] == 21

    async def test_update_document_other_users_file_denied(self):
        # 版本更新换新文件时同样校验归属
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_id.return_value = _doc(status="completed", file_id=10)
        doc_repo.get_by_file_id.return_value = None
        file_svc = AsyncMock()
        file_svc.get_file_by_id = AsyncMock(
            return_value=SimpleNamespace(name="b.pdf", create_by=200)
        )
        svc = DocumentService(
            knowledge_base_repository=kb_repo,
            knowledge_document_repository=doc_repo,
            file_service=file_svc,
        )
        with pytest.raises(BusinessException) as excinfo:
            await svc.update_document(_DB, AsyncMock(), 7, 43, None, _ctx(100))
        assert excinfo.value.code.code == ResultCode.FILE_ACCESS_DENIED.code


class TestImportUrlSsrf:
    """导入网页 SSRF 拦截：解析后 IP 命中私网/环回/链路本地段即拒绝"""

    @staticmethod
    async def _check(url: str, *, fake_infos=None, gaierror=False):
        from app.service.kb.document_service import _ensure_public_url

        if fake_infos is None and not gaierror:
            await _ensure_public_url(url)
            return
        with patch("app.service.kb.document_service.asyncio.get_running_loop") as get_loop:
            loop = MagicMock()
            if gaierror:
                loop.getaddrinfo = AsyncMock(side_effect=socket.gaierror(1, "not known"))
            else:
                loop.getaddrinfo = AsyncMock(return_value=fake_infos)
            get_loop.return_value = loop
            await _ensure_public_url(url)

    async def test_loopback_ip_rejected(self):
        with pytest.raises(BusinessException) as excinfo:
            await self._check("http://127.0.0.1:8991/admin")
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_private_ip_rejected(self):
        with pytest.raises(BusinessException) as excinfo:
            await self._check("http://192.168.1.10/x")
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_link_local_metadata_ip_rejected(self):
        with pytest.raises(BusinessException) as excinfo:
            await self._check("http://169.254.169.254/latest/meta-data")
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_dns_resolving_to_private_ip_rejected(self):
        # 字符串 host 看不出内网（如 localtest.me→127.0.0.1），解析后校验必须拦截
        fake_infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 80))]
        with pytest.raises(BusinessException) as excinfo:
            await self._check("http://internal.example.com/x", fake_infos=fake_infos)
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_dns_resolving_to_loopback_rejected(self):
        fake_infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        with pytest.raises(BusinessException) as excinfo:
            await self._check("http://localtest.me/x", fake_infos=fake_infos)
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_unresolvable_host_rejected(self):
        with pytest.raises(BusinessException) as excinfo:
            await self._check("http://no-such-host.invalid/x", gaierror=True)
        assert excinfo.value.code.code == CODE_BUSINESS

    async def test_missing_hostname_rejected(self):
        with pytest.raises(BusinessException) as excinfo:
            await self._check("http:///path")
        assert excinfo.value.code.code == ResultCode.PARAM_ERROR.code

    async def test_public_ip_allowed(self):
        fake_infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        await self._check("http://93.184.216.34/x", fake_infos=fake_infos)


class TestDeleteCleansChunkRows:
    """删除文档/知识库时 MySQL 分块行、ES 索引、统计三者一致清理"""

    async def test_delete_document_removes_chunk_rows(self, db):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_id.return_value = _doc(status="completed")
        chunk_repo = AsyncMock()
        chunk_repo.count_by_document.return_value = 3
        svc = DocumentService(
            knowledge_base_repository=kb_repo,
            knowledge_document_repository=doc_repo,
            knowledge_chunk_repository=chunk_repo,
        )
        # _sum_document_tokens 走 SQL 聚合，此处 mock 掉只关注分块行清理
        with (
            patch("app.service.kb.document_service.delete_doc_chunks", AsyncMock()) as del_es,
            patch.object(svc, "_sum_document_tokens", AsyncMock(return_value=3)),
        ):
            await svc.delete(_DB, AsyncMock(), 7, _ctx(100))
            del_es.assert_awaited_once_with(1, 7)
        chunk_repo.delete_by_document.assert_awaited_once_with(_DB, 7)

    async def test_delete_kb_removes_chunk_rows_of_all_docs(self, db):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.list_ids_by_kb.return_value = [11, 12]
        chunk_repo = AsyncMock()
        ks = "app.service.kb.knowledge_base_service"
        with (
            patch(f"{ks}.knowledge_base_repository", kb_repo),
            patch(f"{ks}.knowledge_document_repository", doc_repo),
            patch(f"{ks}.knowledge_chunk_repository", chunk_repo),
            patch(f"{ks}.delete_kb_index", AsyncMock()) as del_index,
        ):
            await knowledge_base_service.delete(_DB, AsyncMock(), 1, _ctx(100))
        kb_repo.soft_delete_by_ids.assert_awaited_once()
        doc_repo.soft_delete_by_ids.assert_awaited_once()
        chunk_repo.delete_by_documents.assert_awaited_once_with(_DB, [11, 12])
        del_index.assert_awaited_once_with(1)
