"""审计日志 Repository 跨端互认测试（mongomock 内存 Mongo）。

`audit_log` 集合为三端共享：java/go 写的 `target_id` 是 int64，pymongo 对 <2^31 的裸 int
编码为 int32（BSON 数值类型不相等）⇒ 只用裸 int 查询会静默读不到另两端的记录。
本用例同时守住两件事：①两端的编码都在候选内（类型断言）②python 能读出 int64 形态夹具。

注意 mongomock 不区分 BSON 数值类型（`Int64` 是 int 子类），故"编码差异"由候选类型断言守，
"读路径能处理 int64 形态 + 键名口径"由 mongomock 往返守。
"""

from datetime import UTC, datetime, timedelta

from bson.int64 import Int64

from app.repository.mongo_audit_log_repository import (
    MongoAuditLogRepository,
    _target_id_filter,
)

TARGET_ID = 790001


def _python_form_doc(action: str, create_time: datetime, target_id: int = TARGET_ID) -> dict:
    """python(pymongo) 写入形态：snake_case 键 + 小整数（编码为 int32）"""
    return {
        "operator_id": 2,
        "target_type": "member",
        "target_id": target_id,
        "action": action,
        "module": "member",
        "before_value": {"levelCode": "level_0"},
        "after_value": {"levelCode": "level_1"},
        "ip": "127.0.0.1",
        "user_agent": "pytest",
        "create_time": create_time,
    }


def test_target_id_filter_covers_int32_and_int64_encodings():
    """过滤条件必须同时含裸 int（pymongo 编码为 int32）与 Int64（java/go 编码）"""
    candidates = _target_id_filter("member", TARGET_ID)["target_id"]["$in"]

    assert len(candidates) == 2
    assert any(type(value) is Int64 for value in candidates), "缺 int64 候选：读不到 java/go 写入"
    assert any(type(value) is int for value in candidates), "缺 int32 候选：读不到 python 写入"
    assert {int(value) for value in candidates} == {TARGET_ID}


async def test_list_by_target_reads_int64_form_document(mongo_db):
    """java/go 形态夹具（target_id 为 int64）必须能被读出并回填字段"""
    base = datetime.now(UTC)
    await mongo_db["audit_log"].insert_one(
        _python_form_doc("level_change", base, target_id=Int64(TARGET_ID))
    )
    # 负向对照（用例自证）：camelCase 口径的文档不属本端读取契约，不应混入
    await mongo_db["audit_log"].insert_one(
        {
            "operatorId": 2,
            "targetType": "member",
            "targetId": Int64(TARGET_ID),
            "action": "camel_should_not_match",
            "module": "member",
            "create_time": base,
        }
    )

    repo = MongoAuditLogRepository()
    items, total = await repo.list_by_target("member", TARGET_ID, 1, 10)

    assert total == 1
    assert [item["action"] for item in items] == ["level_change"]
    assert items[0]["operator_id"] == 2
    assert items[0]["before_value"] == {"levelCode": "level_0"}


async def test_list_by_target_paginates_and_sorts_desc(mongo_db):
    """分页与 create_time 倒序（大整数 id 一并覆盖 2^31 以上区间）"""
    big_id = 2**31 + 17
    base = datetime.now(UTC)
    await mongo_db["audit_log"].insert_many(
        [
            _python_form_doc(f"action_{i}", base + timedelta(seconds=i), target_id=big_id)
            for i in range(3)
        ]
        + [_python_form_doc("other_target", base, target_id=big_id + 1)]
    )

    repo = MongoAuditLogRepository()
    items, total = await repo.list_by_target("member", big_id, 1, 2)

    assert total == 3
    assert [item["action"] for item in items] == ["action_2", "action_1"]

    page2, _ = await repo.list_by_target("member", big_id, 2, 2)
    assert [item["action"] for item in page2] == ["action_0"]

    assert await repo.list_by_target("member", 999999999, 1, 10) == ([], 0)
