"""共享 Mongo 日志集合：int64 写侧口径 + 跨端读互认（mongomock 内存 Mongo）。

背景（BSON 数值类型不相等 ⇒ 跨端**静默**读不到，不报错）：
- java 用 `Long`（int64）查 `audit_log.operator_id/target_id`（`MemberServiceImpl` 传
  `Long`）、`login_log.user_id`、`ai_api_call_log.user_id/key_id`；go 侧同为 int64；
- pymongo 对 < 2^31 的裸 int 编码为 **int32** ⇒ python 照原样写入，另两端的单类型
  int64 查询命不中 python 的记录，返回空列表。

本用例守两件事：
1. 写侧落的 BSON 数值类型是 **int64**——用真 BSON 编码字节断言，并带自证对照
   （`_bson_type_byte` 对裸 int 必须报 int32，否则该断言没有区分能力）；
2. 另两端形态的**单类型 int64 查询**能读回 python 写入的记录；python 自身读侧（双类型
   `$in`）对历史 int32 记录同样命中。

注意 mongomock 不区分 BSON 数值类型（`Int64` 是 int 子类，比较按值）⇒ "编码差异"由字节级
断言守，"查询命中"在 mongomock 里只是冒烟（真 Mongo 跨端互认由 go
`audit_log_interop_test.go` / 联调回归覆盖）。
"""

from datetime import UTC, datetime

from bson import BSON
from bson.int64 import Int64

import app.repository.mongo_ai_call_log_repository as call_log_module
from app.repository.login_log_repository import login_log_repository
from app.repository.mongo_ai_call_log_repository import (
    _build_doc,
    mongo_ai_call_log_repository,
)
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.repository.mongo_bson import as_int64

# BSON element type（https://bsonspec.org/spec.html）
BSON_STRING = 0x02
BSON_INT32 = 0x10
BSON_INT64 = 0x12

TARGET_ID = 790001
USER_ID = 7


def _bson_type_byte(value) -> int:
    """单字段文档中首元素的 BSON 类型字节（4 字节总长 + 类型字节 + 字段名 + 值）"""
    return BSON.encode({"v": value})[4]


def _legacy_login_log_doc(user_id: int) -> dict:
    """历史 python 写入形态：`user_id` 是裸 int（落 BSON int32）"""
    return {
        "user_id": user_id,
        "username": "legacy",
        "ip": "127.0.0.1",
        "location": "",
        "browser": "",
        "os": "",
        "device_type": "web",
        "status": 1,
        "message": "ok",
        "create_time": datetime.now(UTC),
    }


def _legacy_ai_call_doc(user_id: int, request_id: str) -> dict:
    """历史 python 写入形态：`user_id`/`key_id` 被还原为裸 int（落 BSON int32）"""
    doc = _build_doc({"user_id": user_id, "key_id": 3, "request_id": request_id})
    doc["user_id"] = user_id
    doc["key_id"] = 3
    return doc


def test_bson_type_detector_self_check():
    """自证：类型字节检测器必须能区分 int32 与 int64（否则后面的 int64 断言恒真）"""
    assert _bson_type_byte(USER_ID) == BSON_INT32
    assert _bson_type_byte(2**40) == BSON_INT64
    assert _bson_type_byte(Int64(USER_ID)) == BSON_INT64


def test_as_int64_keeps_non_integer_values():
    """非整数原样返回（审计 target_id 也可能是订单号字符串 / None），bool 不参与转换"""
    assert as_int64(None) is None
    assert as_int64("NO20260101") == "NO20260101"
    assert as_int64(True) is True
    assert type(as_int64(USER_ID)) is Int64
    assert [type(v) for v in as_int64([1, 2])] == [Int64, Int64]


async def test_audit_write_uses_int64_for_operator_and_target_id(mongo_db):
    """python 写审计日志：operator_id/target_id 落 BSON int64（java Long / go int64 口径）"""
    doc = await mongo_audit_log_repository.create_audit(
        operator_id=2,
        target_type="member",
        target_id=TARGET_ID,
        action="level_change",
        module="member",
    )

    assert _bson_type_byte(doc["operator_id"]) == BSON_INT64
    assert _bson_type_byte(doc["target_id"]) == BSON_INT64

    # 落库文档同样已是 int64（读回核对，而非只看返回值）
    stored = await mongo_db["audit_log"].find_one({"action": "level_change"})
    assert _bson_type_byte(stored["operator_id"]) == BSON_INT64
    assert _bson_type_byte(stored["target_id"]) == BSON_INT64


async def test_audit_write_covers_batch_ids_and_string_target(mongo_db):
    """批量 target_id（id 列表）逐元素 int64；非整型 target_id（订单号）保持原样"""
    batch = await mongo_audit_log_repository.create_audit(
        operator_id=2,
        target_type="dataset",
        target_id=[11, 22],
        action="delete",
        module="dataset",
    )
    assert [_bson_type_byte(v) for v in batch["target_id"]] == [BSON_INT64, BSON_INT64]

    order = await mongo_audit_log_repository.create_audit(
        operator_id=2,
        target_type="order",
        target_id="NO20260101",
        action="refund",
        module="order",
    )
    assert order["target_id"] == "NO20260101"
    assert _bson_type_byte(order["target_id"]) == BSON_STRING


async def test_audit_readable_by_single_type_int64_query(mongo_db):
    """模拟 java/go 的单类型 int64 查询读到 python 写入的审计记录"""
    await mongo_audit_log_repository.create_audit(
        operator_id=2,
        target_type="member",
        target_id=TARGET_ID,
        action="level_change",
        module="member",
    )

    total = await mongo_db["audit_log"].count_documents(
        {"target_type": "member", "target_id": Int64(TARGET_ID)}
    )
    assert total == 1

    by_operator = await mongo_db["audit_log"].count_documents({"operator_id": Int64(2)})
    assert by_operator == 1

    # 负向对照（自证查询真的在按条件筛）：camelCase 键名口径不属共享契约，必须 0 命中
    assert await mongo_db["audit_log"].count_documents({"targetType": "member"}) == 0


async def test_login_log_write_int64_and_read_covers_legacy_int32(mongo_db):
    """登录日志：写出 int64；本端读侧（双类型 $in）同时命中 int64 新记录与 int32 历史记录"""
    doc = await login_log_repository.create_log(
        user_id=USER_ID, username="admin", ip="127.0.0.1", status=1, message="ok"
    )
    assert _bson_type_byte(doc["user_id"]) == BSON_INT64

    # 另两端按单类型 int64 过滤用户日志（java `and("user_id").is(Long)` / go `$in []int64`）
    assert await mongo_db["login_log"].count_documents({"user_id": Int64(USER_ID)}) == 1

    # 历史 int32 记录：本端读侧必须仍能读到（双类型候选，见 mongo_bson.int64_candidates）
    await mongo_db["login_log"].insert_one(_legacy_login_log_doc(USER_ID))
    query = login_log_repository._build_query(
        username=None,
        ip=None,
        status=None,
        device_type=None,
        start_time=None,
        end_time=None,
        user_ids=[USER_ID],
    )
    candidates = query["user_id"]["$in"]
    assert len(candidates) == 2
    assert any(type(value) is Int64 for value in candidates)
    assert any(type(value) is int for value in candidates)

    docs, total = await login_log_repository.page_logs(user_ids=[USER_ID])
    assert total == 2
    assert {row["user_id"] for row in docs} == {USER_ID}


async def test_ai_call_log_write_int64_and_read_covers_legacy_int32(mongo_db):
    """兼容调用审计：写出 int64；另两端 int64 过滤读得到；本端读侧兼容 int32 历史记录"""
    before = set(call_log_module._BACKGROUND_CALL_LOG_TASKS)
    mongo_ai_call_log_repository.insert_async(
        user_id=USER_ID,
        key_id=3,
        key_prefix="dhak_ab",
        conversation_id=15,
        model="gpt-4o-mini",
        endpoint="chat/completions",
        protocol="openai",
        is_stream=False,
        input_tokens=11,
        output_tokens=22,
        credits=0.5,
        status_code=200,
        duration_ms=12,
        client_ip="127.0.0.1",
        request_id="req-new",
    )
    # fire-and-forget：task 注册集合差集 + await（禁用 sleep 等待）
    for task in call_log_module._BACKGROUND_CALL_LOG_TASKS - before:
        await task

    stored = await mongo_db["ai_api_call_log"].find_one({"request_id": "req-new"})
    assert _bson_type_byte(stored["user_id"]) == BSON_INT64
    assert _bson_type_byte(stored["key_id"]) == BSON_INT64
    # conversation_id 无任何端用于过滤，按原值写入（口径见 _build_doc）
    assert stored["conversation_id"] == 15

    counted = await mongo_db["ai_api_call_log"].count_documents(
        {"user_id": Int64(USER_ID), "key_id": Int64(3)}
    )
    assert counted == 1

    await mongo_db["ai_api_call_log"].insert_one(_legacy_ai_call_doc(USER_ID, "req-legacy"))
    records, total = await mongo_ai_call_log_repository.query(user_id=USER_ID)
    assert total == 2
    assert {row["request_id"] for row in records} == {"req-new", "req-legacy"}

    _, by_key_total = await mongo_ai_call_log_repository.query(user_id=USER_ID, key_id=3)
    assert by_key_total == 2


def test_build_doc_keeps_none_attribution():
    """401 被拒调用无凭证上下文：user_id/key_id 为 None 时不得强转（Int64(None) 会抛）"""
    doc = _build_doc({"user_id": None, "key_id": None, "endpoint": "models", "status_code": 401})
    assert doc["user_id"] is None
    assert doc["key_id"] is None
