"""算法选择服务测试（真实 MySQL 测试库 + SAVEPOINT 回滚）。

覆盖：选择树 leaf 字段与空分类过滤、搜索空关键词口径、详情字段完整性（含
path/version/status/ratingCount）、测试接口图片输入二选一校验、对比数量 2-3
与预测异常隔离、推荐 topN 边界。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.algorithm_repository import AlgorithmStatus, algorithm_repository
from app.service.algorithm_select_service import algorithm_select_service

pytestmark = pytest.mark.requires_db


async def _create_published(db, name: str, parent_id: int = 0) -> SysAlgorithm:
    algo = SysAlgorithm(
        parent_id=parent_id, type="TEST", name=name, status=AlgorithmStatus.PUBLISHED
    )
    return await algorithm_repository.create(db, algo)


async def _create_draft(db, name: str, parent_id: int = 0) -> SysAlgorithm:
    algo = SysAlgorithm(parent_id=parent_id, type="TEST", name=name, status=AlgorithmStatus.DRAFT)
    return await algorithm_repository.create(db, algo)


# ===== 选择树 =====


async def test_get_algorithm_tree_only_published_with_leaf_field(db):
    """选择树仅含已发布算法，节点使用 leaf 字段标记（三端统一 leaf）"""
    category = await _create_published(db, "树-分类")
    published = await _create_published(db, "树-已发布", parent_id=category.id)
    await _create_draft(db, "树-草稿")

    tree = await algorithm_select_service.get_algorithm_tree(db)

    all_ids = []
    for node in tree:
        assert "leaf" in node
        assert "isLeaf" not in node
        assert isinstance(node["leaf"], bool)
        assert "parentId" in node

        def collect(n):
            all_ids.append(n["id"])
            for child in n.get("children") or []:
                collect(child)

        collect(node)
    assert published.id in all_ids
    assert "树-草稿" not in [n["name"] for n in tree]


async def test_get_algorithm_tree_full_return_with_root_leaf(db):
    """全量返回：根下无子节点的已发布算法同样展示（leaf=true），不做空分类过滤"""
    root_leaf = await _create_published(db, "根下叶子算法")
    category = await _create_published(db, "分类节点")
    child = await _create_published(db, "分类内算法", parent_id=category.id)

    tree = await algorithm_select_service.get_algorithm_tree(db)
    top_ids = {n["id"] for n in tree}

    assert root_leaf.id in top_ids
    assert category.id in top_ids
    category_node = next(n for n in tree if n["id"] == category.id)
    assert category_node["leaf"] is False
    assert [c["id"] for c in category_node["children"]] == [child.id]
    assert category_node["children"][0]["leaf"] is True
    assert category_node["children"][0]["parentId"] == category.id
    root_leaf_node = next(n for n in tree if n["id"] == root_leaf.id)
    assert root_leaf_node["leaf"] is True
    assert "children" in root_leaf_node


# ===== 搜索 =====


async def test_search_algorithms_blank_keyword_returns_empty(db):
    """空关键词直接返回空列表（文档 §7 常量口径，对齐 Java）"""
    await _create_published(db, "搜索-算法甲")

    assert await algorithm_select_service.search_algorithms(db, None) == []
    assert await algorithm_select_service.search_algorithms(db, "   ") == []


async def test_search_algorithms_keyword_match(db):
    published = await _create_published(db, "搜索-独特关键词XYZ")

    results = await algorithm_select_service.search_algorithms(db, "独特关键词XYZ")

    assert any(r["id"] == published.id for r in results)


# ===== 详情 =====


async def test_get_algorithm_detail_full_fields(db):
    """详情含完整契约字段（path/version/status/ratingCount 对齐 Java AlgorithmDetailVO）"""
    algo = await _create_published(db, "详情-字段完整")
    algo.version = "v1.2.3"
    algo.path = "models/test.pk"
    await db.flush()

    detail = await algorithm_select_service.get_algorithm_detail(db, algo.id)

    assert detail["id"] == algo.id
    assert detail["version"] == "v1.2.3"
    assert detail["path"] == "models/test.pk"
    assert detail["status"] == AlgorithmStatus.PUBLISHED
    assert "ratingCount" in detail
    assert detail["usageCount"] >= 0
    assert detail["sampleImages"] == []


async def test_get_algorithm_detail_unpublished_rejected(db):
    algo = await _create_draft(db, "详情-未发布")

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_select_service.get_algorithm_detail(db, algo.id)

    assert exc_info.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 测试接口 =====


async def test_test_algorithm_requires_image_source(db):
    """imageUrl 与 fileId 均为空时拒绝（A0400）"""
    algo = await _create_published(db, "测试-缺图片输入")

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_select_service.test_algorithm(
            db, algo.id, image_url=None, file_id=None, user_id=1
        )

    assert exc_info.value.code == ResultCode.PARAM_ERROR


async def test_test_algorithm_rejects_unsupported_extension(db):
    algo = await _create_published(db, "测试-非法扩展名")

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_select_service.test_algorithm(
            db, algo.id, image_url="https://example.com/malware.exe", user_id=1
        )

    assert exc_info.value.code == ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH


async def test_test_algorithm_publishes_prediction_result_contract(db):
    """透传预测结果契约字段（logId/status/resultUrl/time）；
    内部字段由 router 层 response_model 过滤"""
    algo = await _create_published(db, "测试-契约透传")

    from app.service.prediction import prediction_service as prediction_module

    original = prediction_module.prediction_service.predict
    prediction_module.prediction_service.predict = AsyncMock(
        return_value={
            "logId": 123,
            "status": 2,
            "resultUrl": "https://storage/pred.png",
            "resultMd5": "a" * 32,
            "time": 88,
        }
    )
    try:
        result = await algorithm_select_service.test_algorithm(
            db, algo.id, image_url="https://example.com/in.png", user_id=1
        )
    finally:
        prediction_module.prediction_service.predict = original

    assert result["logId"] == 123
    assert result["status"] == 2
    assert result["resultUrl"] == "https://storage/pred.png"
    assert result["time"] == 88


# ===== 对比 =====


async def test_compare_size_bounds(db):
    """对比数量需在 2-3 个之间"""
    algo = await _create_published(db, "对比-数量")
    with pytest.raises(BusinessException, match="2-3"):
        await algorithm_select_service.compare(db, [algo.id], image_url="https://example.com/a.png")
    with pytest.raises(BusinessException, match="2-3"):
        await algorithm_select_service.compare(
            db, [algo.id, algo.id, algo.id, algo.id], image_url="https://example.com/a.png"
        )


async def test_compare_requires_image_source(db):
    """缺少图片输入（imageUrl/fileId 均为空）拒绝"""
    a = await _create_published(db, "对比-缺图甲")
    b = await _create_published(db, "对比-缺图乙")

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_select_service.compare(db, [a.id, b.id])

    assert exc_info.value.code == ResultCode.PARAM_ERROR


async def test_compare_rejects_unpublished_algorithm(db):
    """对比含未发布算法整体拒绝（A0401）"""
    published = await _create_published(db, "对比-已发布")
    draft = await _create_draft(db, "对比-草稿")

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_select_service.compare(
            db, [published.id, draft.id], image_url="https://example.com/a.png"
        )

    assert exc_info.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_compare_isolates_prediction_failure(db, monkeypatch):
    """单算法预测失败异常隔离置空，不影响整体对比"""
    a = await _create_published(db, "对比-隔离甲")
    b = await _create_published(db, "对比-隔离乙")

    from app.service.prediction import prediction_service as prediction_module

    call_count = 0

    async def flaky_predict(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("network down")
        return {"logId": 1, "status": 2, "resultUrl": "https://storage/ok.png", "time": 5}

    monkeypatch.setattr(prediction_module.prediction_service, "predict", flaky_predict)

    result = await algorithm_select_service.compare(
        db, [a.id, b.id], image_url="https://example.com/a.png"
    )

    assert call_count == 2
    assert result[0]["algorithmId"] == a.id
    assert result[0]["resultUrl"] is None
    assert result[0]["time"] is None
    assert result[1]["resultUrl"] == "https://storage/ok.png"
    assert result[1]["time"] == 5


# ===== 推荐匹配 =====


async def test_recommend_top_n_bounds(db):
    """topN 超出 1-10 范围拒绝"""
    with pytest.raises(BusinessException, match="topN"):
        await algorithm_select_service.recommend(db, keyword="去雾", top_n=0)
    with pytest.raises(BusinessException, match="topN"):
        await algorithm_select_service.recommend(db, keyword="去雾", top_n=11)


async def test_recommend_empty_input_returns_empty_result(db):
    """空入参兜底返回 total=0（HTTP 200 语义）"""
    result = await algorithm_select_service.recommend(db)

    assert result == {"total": 0, "items": []}


async def test_recommend_keyword_match_scores(db):
    """关键词命中按匹配度排序输出（用唯一 token 避免与种子算法竞争）"""
    unique_kw = f"去雾专精{uuid4().hex[:8]}"
    hit = await _create_published(db, f"推荐-{unique_kw}")
    miss = await _create_published(db, f"推荐-完全无关{uuid4().hex[:8]}")

    result = await algorithm_select_service.recommend(db, keyword=unique_kw, top_n=3)

    ids = [item["algorithmId"] for item in result["items"]]
    assert hit.id in ids
    assert miss.id not in ids
    assert result["items"][0]["algorithmName"] == hit.name
