"""AI 知识库文档 Service：文档上传/导入/版本管理 + 异步处理流水线。

对齐《后端实现-文档管理.md》§3/§5/§7 与《后端实现-架构与公共.md》§4/§8/§9。
- 文档创建(source=upload/url/manual)后由 Router 通过 FastAPI BackgroundTasks 提交异步任务，
  响应发送后执行（请求事务必然已提交，避免后台任务查不到未提交记录）
- 异步流水线：pending→processing→completed/failed，任一步骤失败重试 KB_ASYNC_MAX_RETRY 次
- 版本更新：version+1，旧分块清除（MySQL+ES），重建索引
- 处理状态变更通过 WebSocket 向知识库 owner 推送（复用 websocket_service.manager）
"""

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx
from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.infrastructure.es.kb_chunk_index import (
    bulk_index_chunks,
    delete_doc_chunks,
)
from app.models.base import set_current_user_id
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.models.entity.sys_knowledge_chunk import SysKnowledgeChunk
from app.models.entity.sys_knowledge_document import SysKnowledgeDocument
from app.repository.knowledge_base_repository import knowledge_base_repository
from app.repository.knowledge_chunk_repository import knowledge_chunk_repository
from app.repository.knowledge_document_repository import knowledge_document_repository
from app.service.file_service import file_service
from app.service.kb import chunking_engine, document_parser, embedding_service
from app.service.kb.knowledge_base_service import _check_manage_permission
from app.service.storage.factory import get_storage_by_name

logger = logging.getLogger(__name__)

# 单库文档数/分块数上限（架构文档 §8，需求规格 §2.1.3）
KB_MAX_DOCUMENTS = 500
KB_MAX_CHUNKS = 10000

# 解析为 CPU/IO 密集操作，在线程池中运行避免阻塞事件循环
_parse_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb-parse")

# 状态机：仅 failed 允许重处理；pending/processing 拒绝并发重处理
_PROTECTED_STATUSES = ("pending", "processing")


def _validate_url(url: str) -> None:
    """校验 URL 合法性（http/https）。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise BusinessException(ResultCode.PARAM_ERROR, "URL 格式不合法，仅支持 http/https")


def _validate_file_type(filename: str | None) -> None:
    """上传时同步校验文件格式白名单（复用 parser 的支持列表，非法格式立即拒绝）。"""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in document_parser.SUPPORTED_EXTENSIONS:
        raise BusinessException(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "不支持的文件格式")


async def _push_ws(owner_id: int, payload: dict) -> None:
    """向知识库 owner 推送文档处理状态变更（WS 不可用静默降级，不影响主流程）。"""
    try:
        from app.service.websocket_service import manager

        await manager.send_personal(owner_id, payload)
    except Exception as e:  # noqa: BLE001 - WS 推送失败不影响处理结果
        logger.debug("WS 推送失败: %s", e)


class DocumentService:
    """文档服务（异步版本）。

    创建/重处理/版本更新方法返回调度所需信息（document_id/kb_id/owner_id），
    由 Router 端点通过 background_tasks.add_task(self._process_document_guarded, ...)
    接线异步流水线。服务内部不自行 create_task，避免请求事务未提交导致的竞态。
    """

    # ==================== 创建 ====================

    async def upload(self, 
        db: AsyncSession, redis: Redis, kb_id: int, file_id: int, title: str | None, user
    ) -> dict:
        """上传文档（file_id 关联已上传文件）。返回调度所需信息。"""
        kb = await self._validate_kb(db, kb_id, user)

        # file_id 幂等去重：同库同文件已存在则拒绝重复创建
        existing = await knowledge_document_repository.get_by_file_id(db, kb_id, file_id)
        if existing:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "该文件已存在于知识库中")

        # 单库文档数配额
        doc_count = await knowledge_document_repository.count_by_kb(db, kb_id)
        if doc_count >= KB_MAX_DOCUMENTS:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"单库文档数已达上限({KB_MAX_DOCUMENTS})"
            )

        # 文件必须存在，且上传时即校验文件格式白名单
        file_info = await file_service.get_file_by_id(db, file_id)
        if not file_info:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文件不存在")
        _validate_file_type(file_info.name)

        document = SysKnowledgeDocument(
            knowledge_base_id=kb_id,
            file_id=file_id,
            title=title or file_info.name,
            source="upload",
            parsing_strategy="auto",
            processing_status="pending",
        )
        created = await knowledge_document_repository.create(db, document)

        owner_id = kb.create_by or user.id
        await redis.delete(f"kb:list:{owner_id}", f"kb:config:{kb_id}")
        return {"document_id": created.id, "kb_id": kb_id, "owner_id": owner_id}

    async def batch_upload(self, 
        db: AsyncSession, redis: Redis, kb_id: int, file_ids: list[int], user
    ) -> list[dict]:
        """批量上传文档：逐个走 upload 逻辑，单个失败不影响其余，返回逐条结果。"""
        results = []
        for file_id in file_ids:
            try:
                result = await self.upload(db, redis, kb_id, file_id, None, user)
                results.append({"fileId": file_id, "success": True, **result})
            except BusinessException as e:
                results.append(
                    {"fileId": file_id, "success": False, "code": e.code, "message": e.msg}
                )
        return results

    async def import_url(self, 
        db: AsyncSession, redis: Redis, kb_id: int, url: str, title: str | None, user
    ) -> dict:
        """导入网页为文档：抓取失败抛 A0500 不创建记录；异步任务从分块开始（跳过解析）。"""
        kb = await self._validate_kb(db, kb_id, user)
        _validate_url(url)

        # 抓取网页正文（httpx），失败则拒绝创建
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                cleaned = document_parser.parse_html(resp.text)
        except Exception as e:
            logger.warning("网页抓取失败 url=%s: %s", url, e)
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, "网页抓取失败，无法创建文档"
            ) from None
        if not cleaned.strip():
            raise BusinessException(ResultCode.BUSINESS_ERROR, "网页无有效正文内容")

        doc_count = await knowledge_document_repository.count_by_kb(db, kb_id)
        if doc_count >= KB_MAX_DOCUMENTS:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"单库文档数已达上限({KB_MAX_DOCUMENTS})"
            )

        document = SysKnowledgeDocument(
            knowledge_base_id=kb_id,
            title=title or url,
            source="url",
            parsing_strategy="auto",
            processing_status="pending",
            content=cleaned,
        )
        created = await knowledge_document_repository.create(db, document)

        owner_id = kb.create_by or user.id
        await redis.delete(f"kb:list:{owner_id}", f"kb:config:{kb_id}")
        return {"document_id": created.id, "kb_id": kb_id, "owner_id": owner_id}

    async def create_text(self, 
        db: AsyncSession, redis: Redis, kb_id: int, title: str, content: str, user
    ) -> dict:
        """自定义文本创建文档：content 直接入库，异步任务跳过解析。"""
        kb = await self._validate_kb(db, kb_id, user)

        doc_count = await knowledge_document_repository.count_by_kb(db, kb_id)
        if doc_count >= KB_MAX_DOCUMENTS:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"单库文档数已达上限({KB_MAX_DOCUMENTS})"
            )

        document = SysKnowledgeDocument(
            knowledge_base_id=kb_id,
            title=title,
            source="manual",
            parsing_strategy="auto",
            processing_status="pending",
            content=content,
        )
        created = await knowledge_document_repository.create(db, document)

        owner_id = kb.create_by or user.id
        await redis.delete(f"kb:list:{owner_id}", f"kb:config:{kb_id}")
        return {"document_id": created.id, "kb_id": kb_id, "owner_id": owner_id}

    async def _validate_kb(self, db: AsyncSession, kb_id: int, user) -> SysKnowledgeBase:
        """校验知识库存在且启用、并校验文档管理权限。"""
        kb = await knowledge_base_repository.get_by_id(db, kb_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        if kb.status == 0:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "知识库已禁用")
        await _check_manage_permission(db, kb, user)
        return kb

    # ==================== 查询 ====================

    async def get_page(self, 
        db: AsyncSession, kb_id: int, processing_status: str | None, page: int, size: int, user
    ) -> dict:
        """文档列表（校验知识库可见性）。"""
        kb = await knowledge_base_repository.get_by_id(db, kb_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        if kb.visibility == "private" and kb.create_by != user.id:
            raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权查看他人私有知识库")

        items, total = await knowledge_document_repository.paginate_by_kb(
            db, kb_id, processing_status, page, size
        )
        from app.models.schema.knowledge_base import KnowledgeDocumentVO

        # 列表不返回大字段 content，详情单独返回（避免列表载荷过大）
        return {
            "list": [
                KnowledgeDocumentVO.model_validate(i).model_dump(
                    mode="json", by_alias=True, exclude={"content"}
                )
                for i in items
            ],
            "total": total,
        }

    async def get_detail(self, db: AsyncSession, document_id: int, user) -> dict:
        """文档详情（含解析后 content）。"""
        doc = await knowledge_document_repository.get_by_id(db, document_id)
        if not doc:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文档不存在")
        await self._check_doc_readable(db, doc, user)

        from app.models.schema.knowledge_base import KnowledgeDocumentVO

        return KnowledgeDocumentVO.model_validate(doc).model_dump(mode="json", by_alias=True)

    async def _check_doc_readable(self, db: AsyncSession, doc: SysKnowledgeDocument, user) -> None:
        """文档可见性：所属私有库仅 owner 可读。"""
        kb = await knowledge_base_repository.get_by_id(db, doc.knowledge_base_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        if kb.visibility == "private" and kb.create_by != user.id:
            raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权查看他人私有知识库的文档")

    # ==================== 删除 / 重处理 / 版本更新 ====================

    async def delete(self, db: AsyncSession, redis: Redis, document_id: int, user) -> None:
        """删除文档：软删文档 + ES 清除分块 + 统计 CAS 递减。处理中文档拒绝删除。"""
        doc = await knowledge_document_repository.get_by_id(db, document_id)
        if not doc:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文档不存在")
        kb = await knowledge_base_repository.get_by_id(db, doc.knowledge_base_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        await _check_manage_permission(db, kb, user)

        if doc.processing_status in _PROTECTED_STATUSES:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "文档处理中，暂不能删除")

        chunk_count = await knowledge_chunk_repository.count_by_document(db, document_id)
        token_total = await self._sum_document_tokens(db, document_id)

        await knowledge_document_repository.soft_delete_by_ids(db, [document_id])
        await delete_doc_chunks(kb.id, document_id)

        # 统计 CAS 递减（document_count-1, chunk_count-N, total_tokens-N）
        await self._update_kb_stats_cas(db, kb.id, -1, -chunk_count, -token_total)
        await redis.delete(f"kb:list:{kb.create_by}", f"kb:detail:{kb.id}", f"kb:config:{kb.id}")

    async def reprocess(self, db: AsyncSession, redis: Redis, document_id: int, user) -> dict:
        """重新处理文档：仅 failed 允许；先删旧分块(MySQL+ES)，重置 pending，返回调度信息。"""
        doc = await knowledge_document_repository.get_by_id(db, document_id)
        if not doc:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文档不存在")
        kb = await knowledge_base_repository.get_by_id(db, doc.knowledge_base_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        await _check_manage_permission(db, kb, user)

        if doc.processing_status != "failed":
            raise BusinessException(ResultCode.BUSINESS_ERROR, "仅处理失败的文档允许重新处理")

        # 删除旧分块记录与 ES 分块
        await knowledge_chunk_repository.delete_by_document(db, document_id)
        await delete_doc_chunks(kb.id, document_id)

        doc.processing_status = "pending"
        doc.error = None
        await db.flush()

        owner_id = kb.create_by or user.id
        await redis.delete(f"kb:config:{kb.id}")
        return {"document_id": document_id, "kb_id": kb.id, "owner_id": owner_id}

    async def update_document(self, 
        db: AsyncSession,
        redis: Redis,
        document_id: int,
        file_id: int | None,
        content: str | None,
        user,
    ) -> dict:
        """文档版本更新：重新上传 file_id 或更新文本 → version+1，清除旧分块，重建索引。"""
        doc = await knowledge_document_repository.get_by_id(db, document_id)
        if not doc:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文档不存在")
        kb = await knowledge_base_repository.get_by_id(db, doc.knowledge_base_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        await _check_manage_permission(db, kb, user)

        if doc.processing_status in _PROTECTED_STATUSES:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "文档处理中，暂不能更新")

        new_file_id = doc.file_id
        if file_id is not None:
            # 同库同文件已存在（且不是本文档自身）→ 拒绝
            duplicate = await knowledge_document_repository.get_by_file_id(
                db, kb.id, file_id
            )
            if duplicate and duplicate.id != document_id:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "该文件已存在于知识库中")
            file_info = await file_service.get_file_by_id(db, file_id)
            if not file_info:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文件不存在")
            _validate_file_type(file_info.name)
            new_file_id = file_id

        if new_file_id is None and content is None:
            raise BusinessException(ResultCode.PARAM_ERROR, "需提供新文件或更新文本")

        # 清除旧分块（MySQL + ES）
        await knowledge_chunk_repository.delete_by_document(db, document_id)
        await delete_doc_chunks(kb.id, document_id)

        # version+1 并重置状态；纯文本更新且未换文件时，file_id 置空保持语义一致
        doc.version += 1
        doc.file_id = new_file_id
        if content is not None and file_id is None:
            doc.file_id = None
        doc.processing_status = "pending"
        doc.error = None
        if content is not None:
            doc.content = content
        await db.flush()

        owner_id = kb.create_by or user.id
        await redis.delete(f"kb:config:{kb.id}")
        return {
            "document_id": document_id,
            "kb_id": kb.id,
            "owner_id": owner_id,
            "version": doc.version,
        }

    # ==================== 分块管理 ====================

    async def list_chunks(self, db: AsyncSession, document_id: int, page: int, size: int, user) -> dict:
        """文档分块列表。"""
        doc = await knowledge_document_repository.get_by_id(db, document_id)
        if not doc:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文档不存在")
        await self._check_doc_readable(db, doc, user)

        from app.models.schema.knowledge_base import KnowledgeChunkVO

        stmt = (
            select(SysKnowledgeChunk)
            .where(SysKnowledgeChunk.document_id == document_id)
            .order_by(SysKnowledgeChunk.chunk_index.asc())
        )
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0
        result = await db.execute(stmt.offset((page - 1) * size).limit(size))
        items = list(result.scalars().all())
        return {
            "list": [
                KnowledgeChunkVO.model_validate(i).model_dump(mode="json", by_alias=True)
                for i in items
            ],
            "total": total,
        }

    async def preview_chunks(self, 
        file_id: int, chunking_strategy: str, chunk_size: int, chunk_overlap: int
    ) -> list[dict]:
        """分块预览：下载→解析→分块，不向量化不写索引，返回 [{content, token_count, index}]。"""
        file_bytes, filename = await self._download_file(file_id)
        parsed = await asyncio.get_running_loop().run_in_executor(
            _parse_executor,
            document_parser.parse_document,
            file_bytes,
            filename,
            "auto",
        )
        chunks = chunking_engine.chunk_text(
            parsed.content, chunking_strategy, chunk_size, chunk_overlap
        )
        return [
            {
                "index": c.metadata.get("chunk_index", i),
                "content": c.content,
                "token_count": c.token_count,
            }
            for i, c in enumerate(chunks)
        ]

    # ==================== 异步流水线 ====================

    async def _process_document_guarded(self, document_id: int, kb_id: int, owner_id: int) -> None:
        """异步流水线入口：兜底捕获异常，避免后台任务未处理异常导致告警。"""
        try:
            await self._process_document(document_id, kb_id, owner_id)
        except Exception as e:  # noqa: BLE001 - 流水线兜底，置 failed
            logger.exception("文档 %s 处理异常", document_id)
            try:
                async with get_db_session() as db:
                    doc = await knowledge_document_repository.get_by_id(db, document_id)
                    if doc and doc.processing_status != "completed":
                        doc.processing_status = "failed"
                        doc.error = f"文档处理失败: {e}"
                        await db.flush()
            except Exception:
                logger.exception("文档 %s 失败状态回写异常", document_id)
            await _push_ws(
                owner_id,
                {"type": "kb_doc_status", "documentId": document_id, "status": "failed"},
            )

    async def _process_document(self, document_id: int, kb_id: int, owner_id: int) -> None:
        """文档处理流水线（八步）：解析→清洗→分块→入库→向量化→ES→统计→完成。

        Step 5 分块数超过 KB_MAX_CHUNKS 时拒绝并告警；任一步骤失败重试 KB_ASYNC_MAX_RETRY 次。
        """
        # 后台任务无请求上下文，注入 owner 作为审计人
        set_current_user_id(owner_id)

        async with get_db_session() as db:
            doc = await knowledge_document_repository.get_by_id(db, document_id)
            if not doc:
                return
            kb = await knowledge_base_repository.get_by_id(db, kb_id)
            if not kb:
                return
            # Step 1: 状态 → processing
            doc.processing_status = "processing"
            await db.flush()
            await _push_ws(
                owner_id,
                {"type": "kb_doc_status", "documentId": document_id, "status": "processing"},
            )

        # Step 2-3: 获取内容并解析（manual/url 直接用 content，跳过解析）
        content = doc.content
        if content is None:
            file_bytes, filename = await self._download_file(doc.file_id)
            parsed = await asyncio.get_running_loop().run_in_executor(
                _parse_executor,
                document_parser.parse_document,
                file_bytes,
                filename,
                doc.parsing_strategy,
            )
            content = parsed.content

        # Step 4: 清洗（去除 HTML 标签/多余空白/不可见字符，统一换行）
        cleaned = _clean_text(content)

        # Step 5-6: 分块 + 写 MySQL + 向量化（重试封装）
        retry = settings.KB_ASYNC_MAX_RETRY
        chunks = chunking_engine.chunk_text(
            cleaned, kb.chunking_strategy, kb.chunk_size, kb.chunk_overlap
        )
        if len(chunks) > KB_MAX_CHUNKS:
            logger.warning(
                "文档 %s 分块数 %s 超过上限 %s，拒绝索引", document_id, len(chunks), KB_MAX_CHUNKS
            )
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"分块数({len(chunks)})超过上限({KB_MAX_CHUNKS})"
            )
        total_tokens = sum(c.token_count for c in chunks)

        # 分块写 MySQL（含 chunk_index 元数据）
        async with get_db_session() as db:
            await knowledge_chunk_repository.delete_by_document(db, document_id)
            chunk_entities = []
            for c in chunks:
                metadata = dict(c.metadata)
                chunk_entities.append(
                    SysKnowledgeChunk(
                        document_id=document_id,
                        knowledge_base_id=kb_id,
                        chunk_index=metadata.get("chunk_index", 0),
                        content=c.content,
                        token_count=c.token_count,
                        metadata_=metadata,
                    )
                )
            if chunk_entities:
                await knowledge_chunk_repository.create_all(db, chunk_entities)

            # 更新文档 content / 统计，并捕获标题/版本供 ES 文档使用
            doc = await knowledge_document_repository.get_by_id(db, document_id)
            if not doc:
                return
            doc.content = cleaned
            doc.chunk_count = len(chunk_entities)
            doc.total_tokens = total_tokens
            doc_title = doc.title
            doc_version = doc.version
            await db.flush()

        # Step 7: 向量化 + ES 索引（失败重试 KB_ASYNC_MAX_RETRY 次）
        while retry >= 0:
            try:
                texts = [c.content for c in chunks]
                vectors = await embedding_service.embed_texts(
                    kb.embedding_provider, kb.embedding_model, texts, settings.KB_INDEX_BATCH_SIZE
                )
                es_docs = []
                for chunk_ent, c, vec in zip(chunk_entities, chunks, vectors, strict=True):
                    metadata = dict(c.metadata)
                    create_time = chunk_ent.create_time
                    es_docs.append(
                        {
                            "content_vector": vec,
                            "content": c.content,
                            "doc_title": doc_title,
                            "doc_id": document_id,
                            "chunk_id": chunk_ent.id,
                            "chunk_index": c.metadata.get("chunk_index", 0),
                            "version": doc_version,
                            "metadata": metadata,
                            "tags": metadata.get("tags", []),
                            "create_time": create_time.isoformat()
                            if create_time is not None
                            else "",
                        }
                    )
                if await bulk_index_chunks(kb_id, es_docs):
                    break
                raise RuntimeError("ES 批量写入失败")
            except Exception:
                retry -= 1
                if retry < 0:
                    raise

        # Step 8: CAS 更新知识库统计 + 文档 completed
        async with get_db_session() as db:
            doc = await knowledge_document_repository.get_by_id(db, document_id)
            if not doc:
                return
            doc.processing_status = "completed"
            doc.error = None
            await db.flush()
            await self._update_kb_stats_cas(
                db, kb_id, 1, len(chunk_entities), total_tokens
            )
            await _push_ws(
                owner_id,
                {"type": "kb_doc_status", "documentId": document_id, "status": "completed"},
            )

    # ==================== 工具方法 ====================

    async def _download_file(self, file_id: int) -> tuple[bytes, str]:
        """通过 file_id → sys_file → 存储读取文件字节，返回 (bytes, 文件名)。"""
        async with get_db_session() as db:
            file_info = await file_service.get_file_by_id(db, file_id)
            if not file_info:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文件不存在")
            object_name = file_info.object_name
            filename = file_info.name
            storage = file_info.storage or settings.FILE_STORAGE_TYPE

        storage_service = get_storage_by_name(storage)
        bucket = settings.MINIO_BUCKET_NAME
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                None, lambda: storage_service.download(bucket, object_name)
            )
        except Exception as e:
            logger.warning("文件下载失败 file_id=%s: %s", file_id, e)
            raise BusinessException(ResultCode.FILE_NOT_FOUND, "文件下载失败") from e
        return data, filename

    async def _sum_document_tokens(self, db: AsyncSession, document_id: int) -> int:
        """汇总文档下分块 token 总数（SQL 聚合，避免全量加载大字段）。"""
        stmt = select(func.sum(SysKnowledgeChunk.token_count)).where(
            SysKnowledgeChunk.document_id == document_id
        )
        return (await db.execute(stmt)).scalar() or 0

    async def _update_kb_stats_cas(self, 
        db: AsyncSession, kb_id: int, document_delta: int, chunk_delta: int, token_delta: int
    ) -> None:
        """CAS 更新知识库统计，冲突时重试（CAS 返回 False 表示并发覆盖，重读后重试）。"""
        for _ in range(3):
            if await knowledge_base_repository.update_stats_cas(
                db, kb_id, document_delta, chunk_delta, token_delta
            ):
                return
            await db.flush()
        logger.warning("知识库 %s 统计 CAS 更新 3 次仍冲突，放弃本次累加", kb_id)


def _clean_text(text: str) -> str:
    """文本清洗：去除 HTML 标签/多余空白/不可见字符（含 BOM/零宽字符），统一换行。"""
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\u2060\ufeff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


document_service = DocumentService()
