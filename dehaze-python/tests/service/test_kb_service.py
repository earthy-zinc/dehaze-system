from __future__ import annotations

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.models.entity.sys_knowledge_document import SysKnowledgeDocument
from app.service.kb.document_service import document_service, _clean_text
from app.service.kb.knowledge_base_service import knowledge_base_service
from tests.stubs import MemberBenefitRepo, NullDBSession

CODE_UNAUTHORIZED = ResultCode.ACCESS_UNAUTHORIZED.code
CODE_BUSINESS = ResultCode.BUSINESS_ERROR.code


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


def _fake_db():
    return NullDBSession()


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


class _FakeEmbedding:
    def __init__(self, side_effect=None):
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
    DS = "app.service.kb.document_service"
    patches = (
        patch(f"{DS}.get_db_session", _fake_db),
        patch(f"{DS}.knowledge_base_repository", kb_repo),
        patch(f"{DS}.knowledge_document_repository", doc_repo),
        patch(f"{DS}.knowledge_chunk_repository", chunk_repo),
        patch(f"{DS}.chunking_engine", ce),
        patch(f"{DS}.embedding_service", emb),
        patch(f"{DS}.bulk_index_chunks", bulk_mock),
        patch(f"{DS}._push_ws", AsyncMock()),
    )
    refs = {
        "kb_repo": kb_repo,
        "chunk_repo": chunk_repo,
        "bulk_mock": bulk_mock,
        "doc": doc,
    }
    return patches, refs


class TestKBPermissionMatrix:
    async def test_edit_private_kb_of_others_raises(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(visibility="private", create_by=200)
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.update(None, None, 1, {"name": "篡改"}, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_delete_private_kb_of_others_raises(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(visibility="private", create_by=200)
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.delete(None, None, 1, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_get_private_kb_detail_of_others_raises(self):
        redis = AsyncMock()
        redis.get.return_value = None
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(visibility="private", create_by=200)
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.get_detail(None, redis, 1, 100)
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_public_kb_manageable_only_by_admin(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id = AsyncMock(return_value=_kb(visibility="public", create_by=200))
        kb_repo.get_by_name_and_owner = AsyncMock(return_value=None)
        kb_repo.update = AsyncMock()
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.update(None, AsyncMock(), 1, {"name": "x"}, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED
            await knowledge_base_service.update(None, AsyncMock(), 1, {"name": "x"}, _ctx(999, admin=True))
            kb_repo.update.assert_awaited_once()

    async def test_public_kb_readable_by_anyone(self):
        redis = AsyncMock()
        redis.get.return_value = json.dumps(
            {"id": 3, "name": "平台公共库", "visibility": "public", "documentCount": 12},
            ensure_ascii=False,
        )
        with patch("app.service.kb.knowledge_base_service.knowledge_base_repository") as kb_repo:
            result = await knowledge_base_service.get_detail(None, redis, 3, 100)
            assert result["id"] == 3
            assert result["name"] == "平台公共库"
            kb_repo.get_by_id.assert_not_called()


class TestKBQuotaBoundary:
    def _create_with_limit(self, current: int, level: str = "level_0"):
        kb_repo = AsyncMock()
        kb_repo.count_private_by_owner = AsyncMock(return_value=current)
        kb_repo.get_by_name_and_owner = AsyncMock(return_value=None)
        kb_repo.create = AsyncMock(return_value=_kb(kb_id=9))
        member_repo = MemberBenefitRepo(member=SimpleNamespace(level_code=level))
        KS = "app.service.kb.knowledge_base_service"
        patches = (
            patch(f"{KS}.knowledge_base_repository", kb_repo),
            patch(f"{KS}.member_repository", member_repo),
            patch(f"{KS}.ensure_kb_index", AsyncMock(return_value=True)),
            patch(f"{KS}.get_embedding_dim", return_value=1536),
        )
        return kb_repo, patches

    async def test_normal_user_at_limit_rejected(self):
        _, patches = self._create_with_limit(current=3)
        with _enter(patches):
            with pytest.raises(BusinessException) as excinfo:
                await knowledge_base_service.create(None, None, _create_data(), _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS
            assert "升级" in excinfo.value.message

    async def test_normal_user_just_below_limit_succeeds(self):
        _, patches = self._create_with_limit(current=2)
        with _enter(patches):
            kb_id = await knowledge_base_service.create(None, AsyncMock(), _create_data(), _ctx(100))
            assert kb_id == 9

    async def test_vip_higher_limit_allows_more(self):
        _, patches = self._create_with_limit(current=5, level="level_2")
        with _enter(patches):
            kb_id = await knowledge_base_service.create(None, AsyncMock(), _create_data(), _ctx(100))
            assert kb_id == 9

    async def test_public_kb_not_counted_in_quota(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_name_and_owner = AsyncMock(return_value=None)
        kb_repo.create = AsyncMock(return_value=_kb(kb_id=11, visibility="public"))
        KS = "app.service.kb.knowledge_base_service"
        patches = (
            patch(f"{KS}.knowledge_base_repository", kb_repo),
            patch(f"{KS}.ensure_kb_index", AsyncMock(return_value=True)),
            patch(f"{KS}.get_embedding_dim", return_value=1536),
        )
        with _enter(patches):
            result = await knowledge_base_service.create(
                None, AsyncMock(), _create_data(visibility="public"), _ctx(100, admin=True)
            )
            assert result == 11
            kb_repo.count_private_by_owner.assert_not_called()
            kb_repo.create.assert_awaited_once()


class TestDocPermissionMatrix:
    @staticmethod
    def _patch_kb_repo(kb: SysKnowledgeBase):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = kb
        return kb_repo

    async def test_upload_to_others_private_kb_denied(self):
        kb_repo = self._patch_kb_repo(_kb(visibility="private", create_by=200))
        with patch("app.service.kb.document_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await document_service.upload(None, None, 1, 1, None, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_list_docs_in_others_private_kb_denied(self):
        kb_repo = self._patch_kb_repo(_kb(visibility="private", create_by=200))
        with patch("app.service.kb.document_service.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await document_service.get_page(None, 1, None, 1, 20, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_delete_doc_in_others_private_kb_denied(self):
        kb_repo = self._patch_kb_repo(_kb(visibility="private", create_by=200))
        doc_repo = AsyncMock()
        doc_repo.get_by_id.return_value = _doc(kb_id=1)
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
        ):
            with pytest.raises(BusinessException) as excinfo:
                await document_service.delete(None, None, 7, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_reprocess_doc_in_others_private_kb_denied(self):
        kb_repo = self._patch_kb_repo(_kb(visibility="private", create_by=200))
        doc_repo = AsyncMock()
        doc_repo.get_by_id.return_value = _doc(kb_id=1, status="failed")
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
        ):
            with pytest.raises(BusinessException) as excinfo:
                await document_service.reprocess(None, None, 7, _ctx(100))
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_public_kb_docs_readable_by_others(self):
        kb_repo = self._patch_kb_repo(_kb(visibility="public", create_by=200))
        doc_repo = AsyncMock()
        doc_repo.paginate_by_kb.return_value = ([], 0)
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
        ):
            result = await document_service.get_page(None, 1, None, 1, 20, _ctx(100))
            assert "list" in result


class TestDocStatusMachine:
    def _build(self, doc: SysKnowledgeDocument, kb: SysKnowledgeBase | None = None):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = kb or _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_id.return_value = doc
        chunk_repo = AsyncMock()
        svc = document_service
        return svc, kb_repo, doc_repo, chunk_repo

    async def test_delete_processing_doc_denied(self):
        svc, kb_repo, doc_repo, _ = self._build(_doc(status="processing"))
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
            patch("app.service.kb.document_service.delete_doc_chunks", AsyncMock()) as del_es,
        ):
            with pytest.raises(BusinessException) as excinfo:
                await svc.delete(None, None, 7, _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS
            doc_repo.soft_delete_by_ids.assert_not_called()
            del_es.assert_not_called()

    async def test_update_processing_doc_denied(self):
        svc, kb_repo, doc_repo, _ = self._build(_doc(status="processing"))
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
        ):
            with pytest.raises(BusinessException) as excinfo:
                await svc.update_document(None, None, 7, None, "新内容", _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS

    async def test_reprocess_completed_doc_denied(self):
        svc, kb_repo, doc_repo, _ = self._build(_doc(status="completed"))
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
        ):
            with pytest.raises(BusinessException) as excinfo:
                await svc.reprocess(None, None, 7, _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS

    async def test_reprocess_pending_doc_denied(self):
        svc, kb_repo, doc_repo, _ = self._build(_doc(status="pending"))
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
        ):
            with pytest.raises(BusinessException) as excinfo:
                await svc.reprocess(None, None, 7, _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS

    async def test_reprocess_failed_doc_clears_chunks_and_es(self):
        svc, kb_repo, doc_repo, chunk_repo = self._build(_doc(status="failed"))
        db = _fake_db()
        redis = AsyncMock()
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
            patch("app.service.kb.document_service.knowledge_chunk_repository", chunk_repo),
            patch("app.service.kb.document_service.delete_doc_chunks", AsyncMock()) as del_es,
        ):
            result = await svc.reprocess(db, redis, 7, _ctx(100))
            assert result["document_id"] == 7
            assert result["kb_id"] == 1
            chunk_repo.delete_by_document.assert_awaited_once_with(db, 7)
            del_es.assert_awaited_once_with(1, 7)

    async def test_update_document_version_increments_and_clears_chunks(self):
        svc, kb_repo, doc_repo, chunk_repo = self._build(_doc(version=3, status="completed"))
        db = _fake_db()
        redis = AsyncMock()
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
            patch("app.service.kb.document_service.knowledge_chunk_repository", chunk_repo),
            patch("app.service.kb.document_service.delete_doc_chunks", AsyncMock()) as del_es,
        ):
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


class TestDocIdempotency:
    def _build(self, existing_doc: SysKnowledgeDocument | None = None):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_file_id.return_value = existing_doc
        doc_repo.count_by_kb.return_value = 0
        doc_repo.create.return_value = _doc(doc_id=8, kb_id=2)
        chunk_repo = AsyncMock()
        svc = document_service
        return svc, kb_repo, doc_repo, chunk_repo

    async def test_duplicate_file_in_same_kb_denied(self):
        svc, kb_repo, doc_repo, _ = self._build(existing_doc=_doc())
        redis = AsyncMock()
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
        ):
            with pytest.raises(BusinessException) as excinfo:
                await svc.upload(None, redis, 1, 42, None, _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS
            doc_repo.create.assert_not_called()

    async def test_same_file_in_different_kb_allowed(self):
        svc, kb_repo, doc_repo, _ = self._build(existing_doc=None)
        redis = AsyncMock()
        fs = patch("app.service.kb.document_service.file_service")
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
            fs as fs_mock,
        ):
            fs_mock.get_file_by_id = AsyncMock(return_value=SimpleNamespace(name="a.pdf"))
            result = await svc.upload(None, redis, 2, 42, None, _ctx(100))
            assert result["document_id"] == 8


class TestDocCountQuota:
    async def test_at_limit_500_rejected(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_file_id.return_value = None
        doc_repo.count_by_kb.return_value = 500
        svc = document_service
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
        ):
            with pytest.raises(BusinessException) as excinfo:
                await svc.upload(None, None, 1, 1, None, _ctx(100))
            assert excinfo.value.code.code == CODE_BUSINESS
            assert "500" in excinfo.value.message

    async def test_just_below_limit_allowed(self):
        kb_repo = AsyncMock()
        kb_repo.get_by_id.return_value = _kb(create_by=100)
        doc_repo = AsyncMock()
        doc_repo.get_by_file_id.return_value = None
        doc_repo.count_by_kb.return_value = 499
        doc_repo.create.return_value = _doc(doc_id=99)
        svc = document_service
        redis = AsyncMock()
        fs = patch("app.service.kb.document_service.file_service")
        with (
            patch("app.service.kb.document_service.knowledge_base_repository", kb_repo),
            patch("app.service.kb.document_service.knowledge_document_repository", doc_repo),
            fs as fs_mock,
        ):
            fs_mock.get_file_by_id = AsyncMock(return_value=SimpleNamespace(name="a.pdf"))
            result = await svc.upload(None, redis, 1, 1, None, _ctx(100))
            assert result["document_id"] == 99


class TestDocPipelineFailure:
    async def test_chunk_over_limit_raises_no_stats(self):
        big_chunks = [_FakeChunk(i, f"第{i}段内容", 12) for i in range(10001)]
        patches, refs = _pipeline_env(chunks=big_chunks)
        with _enter(patches):
            with pytest.raises(BusinessException) as excinfo:
                await document_service._process_document(1, 1, 100)
            assert excinfo.value.code.code == CODE_BUSINESS
            assert "上限" in excinfo.value.message
        refs["kb_repo"].update_stats_cas.assert_not_called()

    async def test_embedding_failure_exhausts_retry(self):
        from app.config import settings

        call_count = {"n": 0}

        def _raise(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("embedding 服务不可用")

        patches, refs = _pipeline_env(embed_side_effect=_raise)
        with _enter(patches):
            with pytest.raises(RuntimeError):
                await document_service._process_document(1, 1, 100)
        assert call_count["n"] == 1 + settings.KB_ASYNC_MAX_RETRY
        refs["kb_repo"].update_stats_cas.assert_not_called()

    async def test_es_bulk_failure_exhausts_retry(self):
        from app.config import settings

        patches, refs = _pipeline_env(es_return=False)
        with _enter(patches):
            with pytest.raises(RuntimeError):
                await document_service._process_document(1, 1, 100)
        assert refs["bulk_mock"].call_count == 1 + settings.KB_ASYNC_MAX_RETRY
        refs["kb_repo"].update_stats_cas.assert_not_called()

    async def test_success_path_cleans_dirty_text_writes_chunks_and_stats(self):
        patches, refs = _pipeline_env(content=_DIRTY_DOC)
        with _enter(patches):
            await document_service._process_document(1, 1, 100)
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
    async def test_cas_conflict_then_success(self):
        patches, refs = _pipeline_env(stats_cas_side_effect=[False, False, True])
        with _enter(patches):
            await document_service._process_document(1, 1, 100)
        assert refs["kb_repo"].update_stats_cas.call_count == 3
        assert refs["doc"].processing_status == "completed"
