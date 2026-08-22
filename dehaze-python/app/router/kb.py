"""AI 知识库模块路由

基础路径: /api/v1/kb
覆盖知识库管理、文档管理、分块管理端点（检索端点由 T4 成员追加到本文件）。
对齐《API接口.md》§2.1/2.2/2.3，权限标识 kb:manage / kb:document:manage。
"""

from fastapi import APIRouter, BackgroundTasks, Body, Depends, Path, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators.permission import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.knowledge_base import (
    ChunkPreviewForm,
    DocumentBatchUploadForm,
    DocumentImportUrlForm,
    DocumentTextCreateForm,
    DocumentUpdateForm,
    DocumentUploadForm,
    KnowledgeBaseCreateForm,
    KnowledgeBasePageQuery,
    KnowledgeBaseUpdateForm,
    KnowledgeDocumentPageQuery,
    RetrieveTestForm,
    SearchForm,
)
from app.service.kb.document_service import DocumentService
from app.service.kb.knowledge_base_service import KnowledgeBaseService
from app.service.kb.search_service import search_service


def _filters_to_dict(f) -> dict | None:
    """将检索过滤 Form 字段映射为 build_filters 的 snake_case 参数 dict"""
    if not f:
        return None
    return {
        "doc_type": f.docType,
        "tags": f.tags,
        "start_time": f.startTime,
        "end_time": f.endTime,
        "algorithm_id": f.algorithmId,
        "entities": f.entities,
        "relations": f.relations,
    }

router = APIRouter(
    prefix="/api/v1/kb",
    tags=["AI知识库"],
    dependencies=[Depends(get_current_user)],
)


# ==================== 知识库管理（F-KB-001） ====================


@router.post("", summary="创建知识库")
@require_permission("kb:manage")
async def create_knowledge_base(
    body: KnowledgeBaseCreateForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    kb_id = await KnowledgeBaseService.create(db, redis, body.model_dump(), user)
    return success({"id": kb_id})


@router.get("", summary="知识库列表")
async def list_knowledge_bases(
    query: KnowledgeBasePageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await KnowledgeBaseService.get_page(
        db, redis, user.id, query.keyword, query.pageNum, query.pageSize
    )
    return success(result)


@router.get("/{kb_id}", summary="知识库详情")
async def get_knowledge_base(
    kb_id: int = Path(..., description="知识库ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await KnowledgeBaseService.get_detail(db, redis, kb_id, user.id)
    return success(result)


@router.put("/{kb_id}", summary="编辑知识库")
@require_permission("kb:manage")
async def update_knowledge_base(
    kb_id: int = Path(..., description="知识库ID"),
    body: KnowledgeBaseUpdateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await KnowledgeBaseService.update(db, redis, kb_id, body.model_dump(exclude_none=True), user)
    # 返回更新后的完整知识库 VO（含配置与统计），对齐 SDK KnowledgeBaseVO 契约
    result = await KnowledgeBaseService.get_detail(db, redis, kb_id, user.id)
    return success(result)


@router.delete("/{kb_id}", summary="删除知识库")
@require_permission("kb:manage")
async def delete_knowledge_base(
    kb_id: int = Path(..., description="知识库ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await KnowledgeBaseService.delete(db, redis, kb_id, user)
    return success()


# ==================== 文档管理（F-KB-002） ====================


@router.post("/{kb_id}/documents", summary="上传文档")
@require_permission("kb:document:manage")
async def upload_document(
    kb_id: int = Path(..., description="知识库ID"),
    body: DocumentUploadForm = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    info = await DocumentService.upload(db, redis, kb_id, body.fileId, body.title, user)
    background_tasks.add_task(
        DocumentService._process_document_guarded,
        info["document_id"],
        info["kb_id"],
        info["owner_id"],
    )
    return success({"id": info["document_id"], "processingStatus": "pending"})


@router.post("/{kb_id}/documents/batch", summary="批量上传文档")
@require_permission("kb:document:manage")
async def batch_upload_documents(
    kb_id: int = Path(..., description="知识库ID"),
    body: DocumentBatchUploadForm = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    results = await DocumentService.batch_upload(db, redis, kb_id, body.fileIds, user)
    response = []
    for item in results:
        if item.get("success"):
            background_tasks.add_task(
                DocumentService._process_document_guarded,
                item["document_id"],
                item["kb_id"],
                item["owner_id"],
            )
            response.append(
                {
                    "fileId": item["fileId"],
                    "success": True,
                    "id": item["document_id"],
                    "processingStatus": "pending",
                }
            )
        else:
            response.append(
                {
                    "fileId": item["fileId"],
                    "success": False,
                    "code": item["code"],
                    "message": item["message"],
                }
            )
    return success(response)


@router.post("/{kb_id}/documents/import-url", summary="导入网页为文档")
@require_permission("kb:document:manage")
async def import_url_document(
    kb_id: int = Path(..., description="知识库ID"),
    body: DocumentImportUrlForm = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    info = await DocumentService.import_url(db, redis, kb_id, body.url, body.title, user)
    background_tasks.add_task(
        DocumentService._process_document_guarded,
        info["document_id"],
        info["kb_id"],
        info["owner_id"],
    )
    return success({"id": info["document_id"], "processingStatus": "pending"})


@router.post("/{kb_id}/documents/text", summary="自定义文本创建文档")
@require_permission("kb:document:manage")
async def create_text_document(
    kb_id: int = Path(..., description="知识库ID"),
    body: DocumentTextCreateForm = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    info = await DocumentService.create_text(db, redis, kb_id, body.title, body.content, user)
    background_tasks.add_task(
        DocumentService._process_document_guarded,
        info["document_id"],
        info["kb_id"],
        info["owner_id"],
    )
    return success({"id": info["document_id"], "processingStatus": "pending"})


@router.get("/{kb_id}/documents", summary="知识库文档列表")
async def list_documents(
    kb_id: int = Path(..., description="知识库ID"),
    query: KnowledgeDocumentPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await DocumentService.get_page(
        db, kb_id, query.processingStatus, query.pageNum, query.pageSize, user
    )
    return success(result)


@router.get("/documents/{document_id}", summary="文档详情")
async def get_document(
    document_id: int = Path(..., description="文档ID"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await DocumentService.get_detail(db, document_id, user)
    return success(result)


@router.put("/documents/{document_id}", summary="文档版本更新")
@require_permission("kb:document:manage")
async def update_document(
    document_id: int = Path(..., description="文档ID"),
    body: DocumentUpdateForm = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    info = await DocumentService.update_document(
        db, redis, document_id, body.fileId, body.content, user
    )
    background_tasks.add_task(
        DocumentService._process_document_guarded,
        info["document_id"],
        info["kb_id"],
        info["owner_id"],
    )
    return success(
        {
            "id": info["document_id"],
            "processingStatus": "pending",
            "version": info["version"],
        }
    )


@router.delete("/documents/{document_id}", summary="删除文档")
@require_permission("kb:document:manage")
async def delete_document(
    document_id: int = Path(..., description="文档ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await DocumentService.delete(db, redis, document_id, user)
    return success()


@router.post("/documents/{document_id}/reprocess", summary="重新处理文档")
@require_permission("kb:document:manage")
async def reprocess_document(
    document_id: int = Path(..., description="文档ID"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    info = await DocumentService.reprocess(db, redis, document_id, user)
    background_tasks.add_task(
        DocumentService._process_document_guarded,
        info["document_id"],
        info["kb_id"],
        info["owner_id"],
    )
    return success({"id": info["document_id"], "processingStatus": "pending"})


# ==================== 分块管理（F-KB-003） ====================


@router.post("/documents/chunks/preview", summary="分块预览")
@require_permission("kb:document:manage")
async def preview_chunks(
    body: ChunkPreviewForm = Body(...),
    user: UserContext = Depends(get_current_user),
):
    result = await DocumentService.preview_chunks(
        body.fileId, body.chunking_strategy, body.chunk_size, body.chunk_overlap
    )
    return success(result)


@router.get("/documents/{document_id}/chunks", summary="文档分块列表")
async def list_document_chunks(
    document_id: int = Path(..., description="文档ID"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await DocumentService.list_chunks(db, document_id, pageNum, pageSize, user)
    return success(result)


# ==================== 检索（F-KB-004） ====================


@router.post("/search", summary="知识库检索（多库/元数据过滤/Rerank/MMR）")
async def search_knowledge_base(
    body: SearchForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """知识库多库检索。无权限码，可见性在 service 层过滤（私有库仅 owner）。"""
    result = await search_service.search(
        db,
        redis,
        user.id,
        body.query,
        knowledge_base_ids=body.knowledgeBaseIds,
        top_k=body.topK,
        filters=_filters_to_dict(body.filters),
        enable_mmr=body.enableMMR,
    )
    return success(result)


@router.post("/{kb_id}/retrieve/test", summary="检索测试（召回调试工具）")
@require_permission("kb:manage")
async def retrieve_test(
    kb_id: int = Path(..., description="知识库ID"),
    body: RetrieveTestForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """知识库检索测试/调试：限定单库，返回召回段落 + 分数（含命中 chunk 完整溯源信息）。"""
    result = await search_service.search(
        db,
        redis,
        user.id,
        body.query,
        knowledge_base_ids=[kb_id],
        top_k=body.topK,
        enable_mmr=body.enableMMR,
    )
    return success(result)
