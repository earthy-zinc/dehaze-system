from datetime import datetime

from app.dependencies.auth import UserContext
from app.service.auth_service import auth_service


def _user(admin: bool) -> UserContext:
    """构造真实 UserContext：管理员用 ADMIN 角色表达（is_admin 由角色派生）。"""
    return UserContext(
        id=1 if admin else 2,
        username="admin" if admin else "normal",
        roles=["ADMIN"] if admin else [],
    )


def _make_doc(oid="id-1", user_id=1, username="admin", status=1, ip="1.2.3.4", create_time=None):
    # mongomock/BSON 读回 naive UTC datetime，统一用 naive形式插入
    if create_time is None:
        create_time = datetime(2025, 1, 1, 12, 0, 0)
    return {
        "_id": oid,
        "user_id": user_id,
        "username": username,
        "ip": ip,
        "location": "",
        "browser": "Chrome",
        "os": "Windows",
        "status": status,
        "message": "登录成功",
        "create_time": create_time,
    }


async def test_admin_sees_all_and_no_user_filter(mongo_db):
    await mongo_db["login_log"].insert_many(
        [
            _make_doc(oid="a1", user_id=1, username="admin"),
            _make_doc(oid="a2", user_id=2, username="normal"),
            _make_doc(oid="a3", user_id=3, username="guest"),
        ]
    )
    result = await auth_service.list_login_logs(1, 10, user=_user(True))
    assert result["total"] == 3
    assert len(result["list"]) == 3
    assert {x["userId"] for x in result["list"]} == {1, 2, 3}


async def test_normal_user_forced_to_own_logs(mongo_db):
    await mongo_db["login_log"].insert_many(
        [
            _make_doc(oid="b1", user_id=1, username="admin"),
            _make_doc(oid="b2", user_id=2, username="normal"),
        ]
    )
    # 普通用户强制限定本人日志（T-AM-117）：即使传入他人 username，也绝不返回他人日志
    result = await auth_service.list_login_logs(1, 10, username="admin", user=_user(False))
    assert result["total"] == 0
    assert result["list"] == []

    # 不带冲突 username 时，普通用户仅能看到本人日志
    result = await auth_service.list_login_logs(1, 10, user=_user(False))
    assert result["total"] == 1
    assert result["list"][0]["userId"] == 2
    assert result["list"][0]["username"] == "normal"


async def test_empty_result(mongo_db):
    result = await auth_service.list_login_logs(1, 10, username="nobody", user=_user(True))
    assert result["total"] == 0
    assert result["list"] == []


async def test_time_parse_iso_and_space_formats(mongo_db):
    await mongo_db["login_log"].insert_many(
        [
            _make_doc("c1", user_id=1, create_time=datetime(2025, 1, 1, 6, 0, 0)),
            _make_doc("c2", user_id=1, create_time=datetime(2025, 1, 1, 18, 0, 0)),
            _make_doc("c3", user_id=1, create_time=datetime(2025, 1, 2, 12, 0, 0)),
        ]
    )
    # start_time 用空格格式、end_time 用 ISO 格式，验证两种解析均生效
    result = await auth_service.list_login_logs(
        1,
        10,
        start_time="2025-01-01 00:00:00",
        end_time="2025-01-01T23:59:59",
        user=_user(True),
    )
    assert result["total"] == 2
    assert {x["id"] for x in result["list"]} == {"c1", "c2"}


async def test_pagination_and_sort_desc(mongo_db):
    await mongo_db["login_log"].insert_many(
        [
            _make_doc(oid=f"d{i}", user_id=1, create_time=datetime(2025, 1, i, 12, 0, 0))
            for i in range(1, 6)
        ]
    )
    result = await auth_service.list_login_logs(1, 2, user=_user(True))
    assert result["total"] == 5
    assert len(result["list"]) == 2
    # 按 create_time 倒序，首页应为最新的两条
    assert result["list"][0]["id"] == "d5"
    assert result["list"][1]["id"] == "d4"


async def test_device_type_filter_and_response_field(mongo_db):
    docs = []
    for i, device in enumerate(["web", "android", "miniprogram"], start=1):
        doc = _make_doc(oid=f"e{i}", user_id=1)
        doc["device_type"] = device
        docs.append(doc)
    await mongo_db["login_log"].insert_many(docs)

    # deviceType 精确筛选
    result = await auth_service.list_login_logs(1, 10, device_type="android", user=_user(True))
    assert result["total"] == 1
    assert result["list"][0]["id"] == "e2"
    assert result["list"][0]["deviceType"] == "android"

    # 响应默认回填 web（历史日志无 device_type 字段）
    result = await auth_service.list_login_logs(1, 10, device_type="web", user=_user(True))
    web_rows = [x for x in result["list"] if x["id"] == "e1"]
    assert web_rows
    assert web_rows[0]["deviceType"] == "web"

    # 非法设备类型不入查询条件（返回全量）
    result = await auth_service.list_login_logs(1, 10, device_type="hacker", user=_user(True))
    assert result["total"] == 3
