from app.infrastructure.provider.provider_key_selector import (
    KEY_DAILY_PREFIX,
    KEY_FAIL_STREAK_PREFIX,
    KEY_LAST_USED_PREFIX,
    KEY_UNAVAILABLE_PREFIX,
    _cooldown_seconds,
    provider_key_selector,
)


def _make_key(
    key_id: int, priority: int = 0, weight: int = 1, daily_quota=None, status: int = 1
) -> object:
    key = type("Key", (), {})()
    key.id = key_id
    key.priority = priority
    key.weight = weight
    key.daily_quota = daily_quota
    key.status = status
    key.key_cipher = f"cipher-{key_id}"
    return key


class TestCooldownEscalation:
    def test_cooldown_seconds_gradient(self):
        assert _cooldown_seconds(1) == 300
        assert _cooldown_seconds(2) == 300
        assert _cooldown_seconds(3) == 900
        assert _cooldown_seconds(4) == 900
        assert _cooldown_seconds(5) == 1800
        assert _cooldown_seconds(10) == 1800

    async def test_cooldown_escalation_via_redis(self, mock_redis):
        key_id = 7
        await provider_key_selector.mark_call_failed(mock_redis, key_id, "429")
        assert await mock_redis.ttl(KEY_UNAVAILABLE_PREFIX.format(key_id)) == 300
        assert int(await mock_redis.get(KEY_FAIL_STREAK_PREFIX.format(key_id))) == 1

        await provider_key_selector.mark_call_failed(mock_redis, key_id, "429")
        assert await mock_redis.ttl(KEY_UNAVAILABLE_PREFIX.format(key_id)) == 300

        await provider_key_selector.mark_call_failed(mock_redis, key_id, "429")
        assert await mock_redis.ttl(KEY_UNAVAILABLE_PREFIX.format(key_id)) == 900

        await provider_key_selector.mark_call_failed(mock_redis, key_id, "500")
        assert await mock_redis.ttl(KEY_UNAVAILABLE_PREFIX.format(key_id)) == 900

        await provider_key_selector.mark_call_failed(mock_redis, key_id, "401")
        assert await mock_redis.ttl(KEY_UNAVAILABLE_PREFIX.format(key_id)) == 1800

    async def test_success_resets_fail_streak(self, mock_redis):
        key_id = 8
        for _ in range(4):
            await provider_key_selector.mark_call_failed(mock_redis, key_id, "429")
        assert int(await mock_redis.get(KEY_FAIL_STREAK_PREFIX.format(key_id))) == 4

        await provider_key_selector.mark_call_success(mock_redis, key_id, used_by=100)
        assert await mock_redis.get(KEY_FAIL_STREAK_PREFIX.format(key_id)) is None
        assert await mock_redis.get(KEY_LAST_USED_PREFIX.format(key_id)) is not None

    async def test_success_does_not_clear_cooldown_marker(self, mock_redis):
        key_id = 9
        await provider_key_selector.mark_call_failed(mock_redis, key_id, "429")
        assert await mock_redis.exists(KEY_UNAVAILABLE_PREFIX.format(key_id))
        await provider_key_selector.mark_call_success(mock_redis, key_id, used_by=1)
        assert await mock_redis.exists(KEY_UNAVAILABLE_PREFIX.format(key_id))


class TestListUsableKeys:
    async def test_filter_and_sort(self, mock_redis, monkeypatch):
        from app.repository.ai_provider_key_repository import ai_provider_key_repository
        from datetime import datetime

        keys = [
            _make_key(1, priority=2, weight=5),
            _make_key(2, priority=0, weight=1),
            _make_key(3, priority=0, weight=10),
            _make_key(4, priority=1, weight=1, daily_quota=2),
            _make_key(5, priority=0, weight=1),
            _make_key(6, priority=3, weight=1, daily_quota=2),
        ]

        async def _list_enabled(db, pid):
            return keys

        monkeypatch.setattr(
            ai_provider_key_repository,
            "list_enabled_by_provider",
            _list_enabled,
        )

        await mock_redis.set(KEY_UNAVAILABLE_PREFIX.format(5), 1)
        today = datetime.now().strftime("%Y%m%d")
        await mock_redis.set(KEY_DAILY_PREFIX.format(6, today), 5)

        usable = await provider_key_selector.list_usable_keys(None, mock_redis, 1)
        ids = [k.id for k in usable]
        assert 5 not in ids and 6 not in ids
        assert ids == [3, 2, 4, 1]

    async def test_select_key_uses_same_qualification(self, mock_redis, monkeypatch):
        from app.repository.ai_provider_key_repository import ai_provider_key_repository

        keys = [
            _make_key(1, priority=0, weight=1),
            _make_key(2, priority=0, weight=1),
            _make_key(3, priority=0, weight=1),
        ]

        async def _list_enabled(db, pid):
            return keys

        monkeypatch.setattr(
            ai_provider_key_repository,
            "list_enabled_by_provider",
            _list_enabled,
        )
        monkeypatch.setattr(
            "app.infrastructure.provider.provider_key_selector.decrypt",
            lambda cipher: cipher,
        )
        await mock_redis.set(KEY_UNAVAILABLE_PREFIX.format(1), 1)

        picked = []
        for _ in range(30):
            key = await provider_key_selector.select_key(None, mock_redis, 1)
            assert key is not None
            picked.append(key)
        assert "cipher-1" not in picked
        assert set(picked) <= {"cipher-2", "cipher-3"}
