"""消息未读数集成测试（本次 Android 联调场景）。"""
from __future__ import annotations

from utils import auth, api, mysql


class TestMessages:
    def test_unread_count_api(self, session):
        """API 返回的未读数应为非负整数。"""
        resp = api.get("/api/v1/messages/unread-count")
        count = resp["data"]["count"]
        assert isinstance(count, int) and count >= 0

    def test_unread_count_db(self):
        """DB 直接查未读数应为非负整数。"""
        user_id = auth.get_user_id("admin")
        count = mysql.get_unread_message_count(user_id)
        assert count >= 0

    def test_api_db_consistent(self, session):
        """API 与 DB 的未读数应一致。"""
        resp = api.get("/api/v1/messages/unread-count")
        api_count = resp["data"]["count"]

        user_id = auth.get_user_id("admin")
        db_count = mysql.get_unread_message_count(user_id)

        # 允许有 1-2 条误差（并发标记已读），但应大致一致
        assert abs(api_count - db_count) <= 2, (
            f"API={api_count}, DB={db_count}, 误差过大"
        )

    def test_messages_page(self, session):
        """消息列表应能正常分页查询。"""
        resp = api.get("/api/v1/messages", params={"pageNum": 1, "pageSize": 5})
        data = resp["data"]
        assert "list" in data
        assert isinstance(data["list"], list)
        assert len(data["list"]) <= 5
