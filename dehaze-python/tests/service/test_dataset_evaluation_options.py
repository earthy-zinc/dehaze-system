from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.dataset_repository import dataset_repository
from app.service.dataset.dataset_service import dataset_service

# 测试替身：仓储 find_datasets_with_clear_gt 已 monkeypatch，db 仅传参占位
_DB: AsyncSession = AsyncMock(spec=AsyncSession)


def _make_repo(datasets):
    class _Repo:
        def __init__(self):
            self.calls = []

        async def find_datasets_with_clear_gt(self, db, task_type=None):
            self.calls.append(task_type)
            return datasets

    return _Repo()


def _ds(did, name):
    return SimpleNamespace(id=did, name=name)


async def test_returns_label_value_flat_list(monkeypatch):
    repo = _make_repo([_ds(1, "去雾测试集"), _ds(2, "去雨测试集")])
    monkeypatch.setattr(
        dataset_repository, "find_datasets_with_clear_gt", repo.find_datasets_with_clear_gt
    )
    result = await dataset_service.get_evaluation_options(_DB)
    assert result == [
        {"value": 1, "label": "去雾测试集"},
        {"value": 2, "label": "去雨测试集"},
    ]
    assert repo.calls == [None]


async def test_task_type_passed_to_repository(monkeypatch):
    repo = _make_repo([_ds(1, "去雾测试集")])
    monkeypatch.setattr(
        dataset_repository, "find_datasets_with_clear_gt", repo.find_datasets_with_clear_gt
    )
    result = await dataset_service.get_evaluation_options(_DB, task_type="dehaze")
    assert len(result) == 1
    assert repo.calls == ["dehaze"]


async def test_empty_result_returns_empty_list(monkeypatch):
    repo = _make_repo([])
    monkeypatch.setattr(
        dataset_repository, "find_datasets_with_clear_gt", repo.find_datasets_with_clear_gt
    )
    result = await dataset_service.get_evaluation_options(_DB, task_type="denoise")
    assert result == []
