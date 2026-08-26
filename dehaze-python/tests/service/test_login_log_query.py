from datetime import UTC, datetime

from app.service.auth_service import auth_service


class _User:
    def __init__(self, admin: bool):
        self._admin = admin
        self.id = 1 if admin else 2

    @property
    def is_admin(self) -> bool:
        return self._admin


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
    result = await auth_service.list_login_logs(1, 10, user=_User(admin=True))
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
    result = await auth_service.list_login_logs(
        1, 10, username="admin", user=_User(admin=False)
    )
    assert result["total"] == 0
    assert result["list"] == []

    # 不带冲突 username 时，普通用户仅能看到本人日志
    result = await auth_service.list_login_logs(1, 10, user=_User(admin=False))
    assert result["total"] == 1
    assert result["list"][0]["userId"] == 2
    assert result["list"][0]["username"] == "normal"


async def test_empty_result(mongo_db):
    result = await auth_service.list_login_logs(
        1, 10, username="nobody", user=_User(admin=True)
    )
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
        user=_User(admin=True),
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
    result = await auth_service.list_login_logs(1, 2, user=_User(admin=True))
    assert result["total"] == 5
    assert len(result["list"]) == 2
    # 按 create_time 倒序，首页应为最新的两条
    assert result["list"][0]["id"] == "d5"
    assert result["list"][1]["id"] == "d4"
