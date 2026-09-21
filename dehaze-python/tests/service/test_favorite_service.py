"""
收藏管理服务层单元测试（真实 MySQL 测试库 + SAVEPOINT 回滚）。

覆盖：targetType 白名单校验、收藏/取消幂等（复活/失效重置）、
容量边界（满额/超限 1 条/字典即时生效）、
删除失效联动（算法/数据集删除 → is_invalid=1）、越权隔离、
列表查询（双表关键词/对抗性脏语料/排序/分页）、
计数零值补齐、性能烟测。
"""

import time

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_dataset import SysDataset
from app.repository.dict_repository import dict_repository
from app.repository.favorite_repository import favorite_repository
from app.service.algorithm_service import algorithm_service
from app.service.dataset.dataset_service import dataset_service
from app.service.dict_service import _invalidate_dict_value_cache
from app.service.favorite_service import favorite_service

pytestmark = pytest.mark.requires_db

USER_A = 1006001
USER_B = 1006002


async def _create_algorithm(db, name: str = "测试算法", status: int = 1) -> int:
    algo = SysAlgorithm(name=name, type="dehaze", parent_id=0, status=status)
    db.add(algo)
    await db.flush()
    return int(algo.id)


async def _create_dataset(db, name: str = "测试数据集") -> int:
    ds = SysDataset(name=name, type="用户数据集", parent_id=0, status=1)
    db.add(ds)
    await db.flush()
    return int(ds.id)


async def _add_favorite(db, user_id: int, target_type: str, target_id: int) -> int:
    return await favorite_service.add(db, user_id, target_type, target_id)


async def _set_capacity(db, redis, key: str, value: int) -> None:
    item = await dict_repository.get_by_type_code_and_name(db, "favorite_capacity", key)
    assert item is not None
    item.value = str(value)
    await db.flush()
    await _invalidate_dict_value_cache(redis, "favorite_capacity")


class TestTargetTypeValidation:
    async def test_invalid_target_type_rejected(self, db):
        """非法 targetType（不在五类白名单中）应拒绝而非静默入库。"""
        for bad_type in ("hacker", "ALGORITHM", "", "algo"):
            with pytest.raises(BusinessException) as exc:
                await _add_favorite(db, USER_A, bad_type, 1)
            assert exc.value.code == ResultCode.PARAM_ERROR

    async def test_result_target_must_exist(self, db):
        with pytest.raises(BusinessException) as exc:
            await _add_favorite(db, USER_A, "result", 99999999)
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_reserved_type_skips_existence_check(self, db):
        """image/preset 为预留类型，跳过存在性校验（文档 §3.2）。"""
        fav_id = await _add_favorite(db, USER_A, "image", 99999999)
        assert fav_id > 0
        fav_id = await _add_favorite(db, USER_A, "preset", 99999999)
        assert fav_id > 0


class TestIdempotency:
    async def test_duplicate_add_returns_same_id_and_single_row(self, db):
        target_id = await _create_algorithm(db)
        first_id = await _add_favorite(db, USER_A, "algorithm", target_id)
        second_id = await _add_favorite(db, USER_A, "algorithm", target_id)
        assert second_id == first_id
        assert await favorite_repository.count_user_favorites(db, USER_A) == 1

    async def test_readd_after_cancel_revives_same_record(self, db):
        """取消后重新收藏成功；唯一键含 deleted，重收藏插入新行、软删历史行保留"""
        target_id = await _create_algorithm(db)
        first_id = await _add_favorite(db, USER_A, "algorithm", target_id)
        await favorite_service.delete_by_ids(db, USER_A, [first_id])
        readd_id = await _add_favorite(db, USER_A, "algorithm", target_id)
        assert readd_id != first_id
        assert readd_id > 0
        assert await favorite_repository.count_user_favorites(db, USER_A) == 1

    async def test_readd_resets_invalid_flag(self, db):
        """失效收藏取消后重新收藏，is_invalid 复位为正常。"""
        target_id = await _create_algorithm(db)
        fav_id = await _add_favorite(db, USER_A, "algorithm", target_id)
        await favorite_repository.mark_invalid(db, "algorithm", [target_id])

        page = await favorite_service.get_page(db, USER_A, {"pageNum": 1, "pageSize": 10})
        assert page["list"][0]["isInvalid"] is True

        await favorite_service.delete_by_ids(db, USER_A, [fav_id])
        readd_id = await _add_favorite(db, USER_A, "algorithm", target_id)
        assert readd_id != fav_id
        page = await favorite_service.get_page(db, USER_A, {"pageNum": 1, "pageSize": 10})
        assert page["list"][0]["isInvalid"] is False

    async def test_idempotent_short_circuit_before_capacity(self, db, mock_redis):
        """容量满时重复收藏已收藏对象仍幂等成功（先于容量校验短路）。"""
        await _set_capacity(db, mock_redis, "default", 1)
        target_id = await _create_algorithm(db)
        fav_id = await _add_favorite(db, USER_A, "algorithm", target_id)
        assert await _add_favorite(db, USER_A, "algorithm", target_id) == fav_id


class TestCapacityBoundary:
    async def test_capacity_full_rejects_new_favorite(self, db, mock_redis):
        """恰好满额后再新增 1 条应拒绝；重复收藏已收藏对象仍幂等成功。"""
        await _set_capacity(db, mock_redis, "default", 3)
        algo_ids = [await _create_algorithm(db) for _ in range(3)]
        fav_ids = [await _add_favorite(db, USER_A, "algorithm", aid) for aid in algo_ids]

        extra_algo = await _create_algorithm(db)
        with pytest.raises(BusinessException) as exc:
            await _add_favorite(db, USER_A, "algorithm", extra_algo)
        assert exc.value.code == ResultCode.BUSINESS_ERROR
        assert "收藏已达上限" in exc.value.message

        # 超限 1 条被拒，但已收藏对象重复收藏不受容量影响
        assert await _add_favorite(db, USER_A, "algorithm", algo_ids[0]) == fav_ids[0]
        assert await favorite_repository.count_user_favorites(db, USER_A) == 3

    async def test_capacity_dict_increase_takes_effect_immediately(self, db, mock_redis):
        """容量=2 满额拒绝后，运营调大到 3 立即可再收藏。"""
        await _set_capacity(db, mock_redis, "default", 2)
        algo_ids = [await _create_algorithm(db) for _ in range(2)]
        for aid in algo_ids:
            await _add_favorite(db, USER_A, "algorithm", aid)

        third = await _create_algorithm(db)
        with pytest.raises(BusinessException):
            await _add_favorite(db, USER_A, "algorithm", third)

        await _set_capacity(db, mock_redis, "default", 3)
        assert await _add_favorite(db, USER_A, "algorithm", third) > 0

    async def test_capacity_counts_favorites_not_records(self, db, mock_redis):
        """固定 seed 不变量：收藏数与有效记录数一致（取消/复活不虚增容量占用）。"""
        await _set_capacity(db, mock_redis, "default", 2)
        a1 = await _create_algorithm(db)
        a2 = await _create_algorithm(db)
        a3 = await _create_algorithm(db)
        f1 = await _add_favorite(db, USER_A, "algorithm", a1)
        await _add_favorite(db, USER_A, "algorithm", a2)

        # 取消后重新收藏：复活原记录，有效收藏数仍为 2
        await favorite_service.delete_by_ids(db, USER_A, [f1])
        await _add_favorite(db, USER_A, "algorithm", a1)
        assert await favorite_repository.count_user_favorites(db, USER_A) == 2

        # 第 3 个新目标仍被容量拒绝（复活未虚增占用）
        with pytest.raises(BusinessException):
            await _add_favorite(db, USER_A, "algorithm", a3)


class TestInvalidationLinkage:
    async def test_algorithm_delete_marks_favorite_invalid(self, db):
        target_id = await _create_algorithm(db)
        await _add_favorite(db, USER_A, "algorithm", target_id)

        await algorithm_service.delete_algorithms(db, [target_id])

        page = await favorite_service.get_page(db, USER_A, {"pageNum": 1, "pageSize": 10})
        assert page["list"][0]["isInvalid"] is True

    async def test_dataset_delete_marks_favorite_invalid(self, db, mock_redis, mongo_db):
        target_id = await _create_dataset(db)
        await _add_favorite(db, USER_A, "dataset", target_id)

        await dataset_service.delete_datasets(db, mock_redis, [target_id])

        page = await favorite_service.get_page(db, USER_A, {"pageNum": 1, "pageSize": 10})
        assert page["list"][0]["isInvalid"] is True
        # 已删除数据集的名称不再回显（JOIN 过滤 deleted=0）
        assert page["list"][0]["targetName"] is None

    async def test_invalidated_favorite_still_cancelable(self, db):
        """失效收藏可正常取消（文档 §6.2：用户可手动清理失效条目）。"""
        target_id = await _create_algorithm(db)
        fav_id = await _add_favorite(db, USER_A, "algorithm", target_id)
        await algorithm_service.delete_algorithms(db, [target_id])

        await favorite_service.delete_by_ids(db, USER_A, [fav_id])
        status = await favorite_service.get_status(db, USER_A, "algorithm", target_id)
        assert status["favorited"] is False

    async def test_refavorite_deleted_target_rejected(self, db):
        """对象删除后不可再次收藏同一对象（存在性校验过滤 deleted）。"""
        target_id = await _create_algorithm(db)
        await _add_favorite(db, USER_A, "algorithm", target_id)
        await algorithm_service.delete_algorithms(db, [target_id])

        with pytest.raises(BusinessException) as exc:
            await _add_favorite(db, USER_A, "algorithm", target_id)
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


class TestAuthorizationIsolation:
    async def test_cannot_cancel_others_favorite(self, db):
        target_id = await _create_algorithm(db)
        fav_id = await _add_favorite(db, USER_A, "algorithm", target_id)

        await favorite_service.delete_by_ids(db, USER_B, [fav_id])

        page = await favorite_service.get_page(db, USER_A, {"pageNum": 1, "pageSize": 10})
        assert any(item["id"] == fav_id for item in page["list"])

    async def test_page_only_returns_own_records(self, db):
        target_id = await _create_algorithm(db)
        fav_id = await _add_favorite(db, USER_A, "algorithm", target_id)

        page_b = await favorite_service.get_page(db, USER_B, {"pageNum": 1, "pageSize": 10})
        assert all(item["id"] != fav_id for item in page_b["list"])

    async def test_status_scoped_by_user(self, db):
        target_id = await _create_algorithm(db)
        await _add_favorite(db, USER_A, "algorithm", target_id)

        status_b = await favorite_service.get_status(db, USER_B, "algorithm", target_id)
        assert status_b["favorited"] is False


class TestListQuery:
    async def test_keywords_match_algorithm_name(self, db):
        target_id = await _create_algorithm(db, name="夜景去雾算法X")
        await _add_favorite(db, USER_A, "algorithm", target_id)
        await _create_algorithm(db, name="超分算法")  # 未收藏，不应出现

        page = await favorite_service.get_page(
            db,
            USER_A,
            {"pageNum": 1, "pageSize": 10, "keywords": "去雾"},
        )
        assert page["total"] == 1
        assert page["list"][0]["targetName"] == "夜景去雾算法X"

    async def test_keywords_match_dataset_name(self, db):
        """数据集收藏的名称关键词搜索（此前仅匹配算法名，已修复）。"""
        target_id = await _create_dataset(db, name="夜间数据集Alpha")
        await _add_favorite(db, USER_A, "dataset", target_id)

        page = await favorite_service.get_page(
            db,
            USER_A,
            {"pageNum": 1, "pageSize": 10, "keywords": "夜间"},
        )
        assert page["total"] == 1
        assert page["list"][0]["targetName"] == "夜间数据集Alpha"

    async def test_keywords_adversarial_dirty_corpus(self, db):
        """对抗性脏语料：LIKE 通配符转义 + emoji/零宽/CRLF/超长输入不报错。"""
        ds_under = await _create_dataset(db, name="数据a_b数据")
        await _add_favorite(db, USER_A, "dataset", ds_under)
        ds_plain = await _create_dataset(db, name="数据aXb数据")
        await _add_favorite(db, USER_A, "dataset", ds_plain)
        ds_pct = await _create_dataset(db, name="数据a%b数据")
        await _add_favorite(db, USER_A, "dataset", ds_pct)

        # 转义语义：下划线按字面匹配，不展开为任意单字符
        page = await favorite_service.get_page(
            db,
            USER_A,
            {"pageNum": 1, "pageSize": 10, "keywords": "a_b"},
        )
        assert {item["targetId"] for item in page["list"]} == {ds_under}

        # 百分号按字面匹配，不作为任意串通配符
        page = await favorite_service.get_page(
            db,
            USER_A,
            {"pageNum": 1, "pageSize": 10, "keywords": "a%b"},
        )
        assert {item["targetId"] for item in page["list"]} == {ds_pct}

        # 脏语料：不报错且无法误匹配
        # （单字符 "%" / "_" 已在上方断言按字面精确命中，此处覆盖无法命中的脏输入）
        dirty_no_match = [
            "\\",
            "%_",
            "🌫️",
            "零宽\u200b字符",
            "line\r\nbreak",
            "全角ＡＢＣ",
            "a" * 200,
        ]
        for kw in dirty_no_match:
            page = await favorite_service.get_page(
                db,
                USER_A,
                {"pageNum": 1, "pageSize": 10, "keywords": kw},
            )
            assert page["total"] == 0, f"脏关键词 {kw!r} 不应匹配到任何收藏"

    async def test_sort_asc_desc_and_unknown_sortby_ignored(self, db):
        """排序仅由 sortOrder 决定；sortBy=rating 等死值被忽略且不筛掉 dataset 行。"""
        algo_id = await _create_algorithm(db)
        ds_id = await _create_dataset(db)
        await _add_favorite(db, USER_A, "algorithm", algo_id)
        await _add_favorite(db, USER_A, "dataset", ds_id)

        desc_page = await favorite_service.get_page(
            db, USER_A, {"pageNum": 1, "pageSize": 10, "sortOrder": "desc"}
        )
        assert [i["targetType"] for i in desc_page["list"]] == ["dataset", "algorithm"]

        asc_page = await favorite_service.get_page(
            db, USER_A, {"pageNum": 1, "pageSize": 10, "sortOrder": "asc"}
        )
        assert [i["targetType"] for i in asc_page["list"]] == ["algorithm", "dataset"]

        # rating/usageCount 等未实现排序值：回退时间排序，且 dataset 行不被过滤
        legacy_page = await favorite_service.get_page(
            db,
            USER_A,
            {"pageNum": 1, "pageSize": 10, "sortBy": "rating", "sortOrder": "asc"},
        )
        assert [i["targetType"] for i in legacy_page["list"]] == ["algorithm", "dataset"]

    async def test_pagination_boundary(self, db):
        for i in range(3):
            await _add_favorite(db, USER_A, "image", 100000 + i)
        page = await favorite_service.get_page(db, USER_A, {"pageNum": 2, "pageSize": 2})
        assert page["total"] == 3
        assert len(page["list"]) == 1

        empty_page = await favorite_service.get_page(db, USER_A, {"pageNum": 99, "pageSize": 2})
        assert empty_page["total"] == 3
        assert empty_page["list"] == []

    async def test_count_zero_fill_and_type_filter(self, db):
        target_id = await _create_algorithm(db)
        await _add_favorite(db, USER_A, "algorithm", target_id)

        counts = await favorite_service.get_count(db, USER_A)
        assert [c["targetType"] for c in counts] == [
            "algorithm",
            "result",
            "dataset",
            "image",
            "preset",
        ]
        by_type = {c["targetType"]: c["count"] for c in counts}
        assert by_type["algorithm"] == 1
        assert by_type["image"] == 0

        only_dataset = await favorite_service.get_count(db, USER_A, "dataset")
        assert only_dataset == [{"targetType": "dataset", "count": 0}]


class TestPerformanceSmoke:
    async def test_page_with_30_favorites(self, db):
        """30 条收藏下列表分页查询性能烟测（无 N+1，单查询返回）。"""
        for i in range(30):
            ds_id = await _create_dataset(db, name=f"烟测数据集{i:03d}")
            await _add_favorite(db, USER_A, "dataset", ds_id)

        start = time.monotonic()
        page = await favorite_service.get_page(db, USER_A, {"pageNum": 1, "pageSize": 100})
        elapsed = time.monotonic() - start

        assert page["total"] == 30
        assert len(page["list"]) == 30
        assert elapsed < 5, f"列表查询耗时 {elapsed:.2f}s，疑似 N+1 查询"


class TestSoftDeleteUniqueKey:
    @pytest.mark.requires_db
    async def test_cancel_after_recollect_twice_no_unique_violation(self, db):
        """取消收藏两次（中途重新收藏）不得撞 uk_user_target。

        deleted 列语义是"删除时的行 id"：写常量 1 时，第二次取消会与本行第一次留下的
        deleted=1 记录冲突（1062 → 500）；复现路径即 收藏→取消→再收藏→再取消。
        """
        algo_id = await _create_algorithm(db)
        first_id = await _add_favorite(db, USER_A, "algorithm", algo_id)
        await favorite_service.delete_by_ids(db, USER_A, [first_id])

        # 软删历史行保留，重收藏插入新行（既有行为，见 TestIdempotency）
        revived_id = await _add_favorite(db, USER_A, "algorithm", algo_id)
        assert revived_id != first_id

        # 改前此处抛 IntegrityError（Duplicate entry '...-1' for key 'uk_user_target'）
        await favorite_service.delete_by_ids(db, USER_A, [revived_id])

        assert (
            await favorite_repository.get_by_user_and_target(db, USER_A, "algorithm", algo_id)
            is None
        )
