"""API Key 服务层测试（对照测试用例.md §4.8，T-AM-060 ~ T-AM-068）。

db fixture 为真实 MySQL 测试库 + 测试级事务回滚；用户使用种子账号
（user_id=5 普通 / user_id=2 admin）验证归属隔离与越权删除。
"""

import hashlib

import pytest

from app.service.api_key_service import api_key_service

USER_ID = 5
OTHER_USER_ID = 2


@pytest.fixture
async def created_key(db):
    """创建一个无过期限制的密钥，返回服务层结果（含明文）。"""
    return await api_key_service.create_api_key(db, USER_ID, "测试Key")


class TestCreateApiKey:
    async def test_returns_plaintext_with_dhak_prefix(self, db, created_key):
        """T-AM-060：创建返回 dhak_ 前缀明文、keyPrefix 与名称。"""
        assert created_key["apiKey"].startswith("dhak_")
        assert created_key["keyPrefix"] == created_key["apiKey"][:12]
        assert created_key["name"] == "测试Key"
        assert created_key["id"] > 0

    async def test_stored_hash_is_sha256_hex(self, db, created_key):
        """T-AM-062：数据库仅存 SHA-256 十六进制哈希，非明文。"""
        entity = await api_key_service.api_key_repository.get_active_by_id_and_user(
            db, created_key["id"], USER_ID
        )
        assert entity is not None
        assert entity.key_hash == hashlib.sha256(created_key["apiKey"].encode()).hexdigest()
        assert entity.key_hash is not None
        assert len(entity.key_hash) == 64
        assert created_key["apiKey"] not in entity.key_hash


class TestListApiKey:
    async def test_list_hides_plaintext(self, db, created_key):
        """T-AM-061：列表不含明文 apiKey，仅含 keyPrefix。"""
        items = await api_key_service.list_api_keys(db, USER_ID)
        target = next(i for i in items if i["id"] == created_key["id"])
        assert target["apiKey"] is None
        assert target["keyPrefix"] == created_key["keyPrefix"]

    async def test_list_scoped_to_owner(self, db, created_key):
        """T-AM-061a：归属隔离——其他用户列表不包含该 Key。"""
        other_items = await api_key_service.list_api_keys(db, OTHER_USER_ID)
        assert all(i["id"] != created_key["id"] for i in other_items)


class TestDeleteApiKey:
    async def test_cross_user_delete_is_forbidden(self, db, created_key):
        """T-AM-067：跨用户越权删除返回 False（等价于不存在，不泄露归属）。"""
        deleted = await api_key_service.delete_api_key(db, OTHER_USER_ID, created_key["id"])
        assert deleted is False
        # 越权删除不得真正吊销原属主的 Key
        entity = await api_key_service.api_key_repository.get_active_by_id_and_user(
            db, created_key["id"], USER_ID
        )
        assert entity is not None

    async def test_delete_nonexistent_returns_false(self, db):
        """T-AM-068：删除不存在的 Key 返回 False。"""
        assert await api_key_service.delete_api_key(db, USER_ID, 99999) is False

    async def test_revoked_key_no_longer_active(self, db, created_key):
        """T-AM-065：吊销后按归属查询也取不到（认证层将以"无此 Key"拒绝）。"""
        assert await api_key_service.delete_api_key(db, USER_ID, created_key["id"]) is True
        entity = await api_key_service.api_key_repository.get_active_by_id_and_user(
            db, created_key["id"], USER_ID
        )
        assert entity is None


class TestExpiredKey:
    async def test_expired_expiry_is_persisted(self, db):
        """T-AM-064 前置：创建时写入的 expiresAt 落库（认证层按它拒绝过期 Key）。"""
        from datetime import datetime, timedelta

        past = datetime.now() - timedelta(days=1)
        result = await api_key_service.create_api_key(db, USER_ID, "过期Key", expires_at=past)
        entity = await api_key_service.api_key_repository.get_active_by_id_and_user(
            db, result["id"], USER_ID
        )
        assert entity is not None
        assert entity.expires_at is not None
        assert entity.expires_at < datetime.now()
