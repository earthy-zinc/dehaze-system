"""会员管理端详情弹窗数据接口测试（mongomock 内存 Mongo）。

覆盖目标会员操作审计日志查询（MongoAuditLogRepository.list_by_target +
member_service.list_member_audit_logs）。成长值流水/消费记录/权益使用分别
复用 growth_service / order_service / get_benefit_summary 既有已测实现。
"""

from datetime import UTC, datetime, timedelta

from app.service.member.member_service import member_service

USER_ID = 1007001


def _make_doc(
    action: str, target_id=USER_ID, target_type: str = "member", create_time: datetime | None = None
) -> dict:
    return {
        "operator_id": 2,
        "target_type": target_type,
        "target_id": target_id,
        "action": action,
        "module": "member",
        "before_value": {"levelCode": "level_0"},
        "after_value": {"levelCode": "level_1"},
        "ip": "127.0.0.1",
        "user_agent": "pytest",
        "create_time": create_time or datetime.now(UTC),
    }


async def test_list_member_audit_logs_filter_and_order(mongo_db):
    """按 target_type+target_id 过滤，create_time 倒序，字段 camelCase 映射"""
    base = datetime.now(UTC)
    await mongo_db["audit_log"].insert_many(
        [
            _make_doc("level_change", create_time=base),
            _make_doc("growth_change", create_time=base + timedelta(seconds=1)),
            # 他人 / 其他模块的日志不应混入
            _make_doc("level_change", target_id=999999),
            _make_doc("refund", target_type="order"),
        ]
    )

    result = await member_service.list_member_audit_logs(USER_ID, 1, 10)
    assert result["total"] == 2
    assert result["list"][0]["action"] == "growth_change"
    assert result["list"][1]["action"] == "level_change"
    row = result["list"][0]
    assert row["operatorId"] == 2
    assert row["module"] == "member"
    assert row["afterValue"] == {"levelCode": "level_1"}
    assert row["beforeValue"] == {"levelCode": "level_0"}
    assert row["ip"] == "127.0.0.1"
    assert row["id"]
    assert row["createTime"]


async def test_list_member_audit_logs_pagination(mongo_db):
    base = datetime.now(UTC)
    await mongo_db["audit_log"].insert_many(
        [_make_doc(f"action_{i}", create_time=base + timedelta(seconds=i)) for i in range(3)]
    )

    result = await member_service.list_member_audit_logs(USER_ID, 1, 2)
    assert result["total"] == 3
    assert len(result["list"]) == 2
    assert result["list"][0]["action"] == "action_2"

    page2 = await member_service.list_member_audit_logs(USER_ID, 2, 2)
    assert len(page2["list"]) == 1
    assert page2["list"][0]["action"] == "action_0"


async def test_list_member_audit_logs_empty(mongo_db):
    result = await member_service.list_member_audit_logs(888888, 1, 10)
    assert result == {"list": [], "total": 0}
