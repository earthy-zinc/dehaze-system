import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.algorithm_repository import AlgorithmStatus, algorithm_repository
from app.service.algorithm_select_service import AlgorithmSelectService


def _algo(aid, name, type_="dehaze", parent_id=0, status=AlgorithmStatus.PUBLISHED, desc=""):
    class _A:
        pass

    a = _A()
    a.id = aid
    a.name = name
    a.type = type_
    a.parent_id = parent_id
    a.status = status
    a.description = desc
    return a


def _stub_list_published(algos):
    async def _fn(*args, **kwargs):
        return algos

    return _fn


async def test_keyword_match_returns_top_n(monkeypatch):
    algos = [
        _algo(1, "夜景去雾算法", desc="用于夜景去雾"),
        _algo(2, "去雾增强", desc="通用去雾"),
        _algo(3, "去雨算法", desc="derain"),
        _algo(4, "图像超分", desc="super resolution"),
    ]
    monkeypatch.setattr(algorithm_repository, "list_published", _stub_list_published(algos))

    result = await AlgorithmSelectService.recommend(None, keyword="去雾", top_n=3)
    assert result["total"] == 2
    assert len(result["items"]) == 2
    assert {i["algorithmName"] for i in result["items"]} == {"夜景去雾算法", "去雾增强"}
    assert all(i["matchScore"] > 0 for i in result["items"])


async def test_task_type_filter(monkeypatch):
    algos = [
        _algo(1, "夜景去雾", type_="dehaze"),
        _algo(2, "去雨", type_="derain"),
    ]
    monkeypatch.setattr(algorithm_repository, "list_published", _stub_list_published(algos))

    result = await AlgorithmSelectService.recommend(None, keyword="夜景", task_type="dehaze", top_n=3)
    assert result["total"] == 1
    assert result["items"][0]["algorithmId"] == 1


async def test_empty_result_returns_200_shape(monkeypatch):
    monkeypatch.setattr(algorithm_repository, "list_published", _stub_list_published([]))
    result = await AlgorithmSelectService.recommend(None, keyword="不存在的关键词", top_n=3)
    assert result["total"] == 0
    assert result["items"] == []


async def test_no_keyword_no_sample_no_task_returns_empty(monkeypatch):
    monkeypatch.setattr(algorithm_repository, "list_published", _stub_list_published([_algo(1, "夜景去雾")]))
    result = await AlgorithmSelectService.recommend(None)
    assert result["total"] == 0
    assert result["items"] == []


async def test_topn_out_of_range_raises(monkeypatch):
    monkeypatch.setattr(algorithm_repository, "list_published", _stub_list_published([_algo(1, "夜景去雾")]))
    with pytest.raises(BusinessException) as exc:
        await AlgorithmSelectService.recommend(None, keyword="去雾", top_n=11)
    assert exc.value.code == ResultCode.BUSINESS_ERROR

    with pytest.raises(BusinessException) as exc:
        await AlgorithmSelectService.recommend(None, keyword="去雾", top_n=0)
    assert exc.value.code == ResultCode.BUSINESS_ERROR


async def test_sample_algorithm_not_found_raises_a0401(monkeypatch):
    monkeypatch.setattr(algorithm_repository, "list_published", _stub_list_published([_algo(1, "夜景去雾")]))
    monkeypatch.setattr(
        algorithm_repository, "get_by_id_include_unpublished", _stub_list_published(None)
    )
    with pytest.raises(BusinessException) as exc:
        await AlgorithmSelectService.recommend(None, sample_algorithm_id=999999, top_n=3)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND
