"""语音热词服务单元测试（T-VS-031~T-VS-045）。

用户级 / 全局热词 CRUD 走真实仓储（db fixture 落库，断言业务结果）；
生效范围合并、超限拒绝按规则构造数据后断言业务异常码。
"""

import pytest
from sqlalchemy import text

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.voice import HotwordForm
from app.service.voice import hotword_service as hs_module
from app.service.voice.hotword_service import hotword_service


def _count_rows(db, scope, user_id=None):
    stmt = "SELECT COUNT(*) AS c FROM sys_voice_hotword WHERE deleted=0 AND scope=:scope"
    params = {"scope": scope}
    if user_id is not None:
        stmt += " AND user_id=:uid"
        params["uid"] = user_id
    else:
        stmt += " AND user_id IS NULL"
    return db.execute(text(stmt), params).scalar()


# ── T-VS-031/032：用户级热词 CRUD ──


@pytest.mark.asyncio
async def test_user_hotword_add_and_list(db):
    added = await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="量子纠缠"))
    assert added.id is not None
    assert added.word == "量子纠缠"

    results = await hotword_service.list_user_hotwords(db, 1001)
    assert [r.word for r in results] == ["量子纠缠"]


@pytest.mark.asyncio
async def test_user_hotword_delete(db):
    added = await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="待删除词"))
    await hotword_service.delete_user_hotword(db, added.id, 1001)

    results = await hotword_service.list_user_hotwords(db, 1001)
    assert results == []


@pytest.mark.asyncio
async def test_user_hotword_rejects_other_user_delete(db):
    added = await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="私有词"))
    # 用户 2002 删除用户 1001 的词 → 归属不匹配，按不存在处理
    with pytest.raises(BusinessException) as exc:
        await hotword_service.delete_user_hotword(db, added.id, 2002)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


# ── T-VS-033/034：全局热词 CRUD ──


@pytest.mark.asyncio
async def test_global_hotword_add_and_list(db):
    added = await hotword_service.add_global_hotword(db, HotwordForm(word="标准术语"))
    assert added.word == "标准术语"

    results = await hotword_service.list_global_hotwords(db)
    assert "标准术语" in [r.word for r in results]


@pytest.mark.asyncio
async def test_global_hotword_delete(db):
    added = await hotword_service.add_global_hotword(db, HotwordForm(word="待删全局词"))
    await hotword_service.delete_global_hotword(db, added.id)

    results = await hotword_service.list_global_hotwords(db)
    assert all(r.word != "待删全局词" for r in results)


@pytest.mark.asyncio
async def test_global_hotword_rejects_when_not_global_scope(db):
    # 删除一个用户级热词当作全局热词删 → 作用域不匹配，按不存在处理
    user_word = await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="误删词"))
    with pytest.raises(BusinessException) as exc:
        await hotword_service.delete_global_hotword(db, user_word.id)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


# ── T-VS-035/036/037：生效范围隔离（用户合并全局+本人；他人仅全局）──


@pytest.mark.asyncio
async def test_effective_words_merge_global_and_self(db):
    await hotword_service.add_global_hotword(db, HotwordForm(word="全局词"))
    await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="A私有词"))

    words = await hotword_service.get_effective_words(db, 1001)
    assert "全局词" in words
    assert "A私有词" in words


@pytest.mark.asyncio
async def test_effective_words_other_user_only_global(db):
    await hotword_service.add_global_hotword(db, HotwordForm(word="全局词"))
    await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="A私有词"))

    # 用户 2002 不继承用户 1001 的私有词，仅看到全局词
    words = await hotword_service.get_effective_words(db, 2002)
    assert "全局词" in words
    assert "A私有词" not in words


# ── T-VS-038/039：超限拒绝（仅用户级有上限，全局无上限）──


@pytest.mark.asyncio
async def test_user_hotword_rejects_when_limit_exceeded(db, monkeypatch):
    monkeypatch.setattr(hs_module, "_MAX_USER_HOTWORDS", 2)
    await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="词1"))
    await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="词2"))

    with pytest.raises(BusinessException) as exc:
        await hotword_service.add_user_hotword(db, 1001, HotwordForm(word="词3"))
    assert exc.value.code == ResultCode.BUSINESS_ERROR
    assert "上限" in exc.value.message
