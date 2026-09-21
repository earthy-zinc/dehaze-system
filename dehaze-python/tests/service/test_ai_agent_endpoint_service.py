"""外部 A2A 端点服务测试：base_url 唯一键与软删行复活。

唯一键 uk_base_url 含软删行：删除端点后以相同 base_url 重新注册复活原行，
活跃端点重复注册拒绝。SSRF 校验与 Agent Card 拉取为外部依赖，测试中打桩。
"""

from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.ai_agent_endpoint_repository import ai_agent_endpoint_repository
from app.service.ai_agent_endpoint_service import (
    AiAgentEndpointService,
    ai_agent_endpoint_service,
)

pytestmark = pytest.mark.requires_db


@pytest.fixture(autouse=True)
def _stub_external(monkeypatch):
    async def _safe_url(url):
        return True

    async def _no_refresh(db, endpoint_id):
        return None

    monkeypatch.setattr("app.service.ai_agent_endpoint_service.is_safe_url", _safe_url)
    monkeypatch.setattr(AiAgentEndpointService, "_refresh_agent_card", _no_refresh)


def _form(**kwargs):
    defaults = {
        "name": "端点A",
        "agent_card_url": None,
        "base_url": "https://a2a-revive.example.com",
        "auth_type": "http",
        "credential": None,
        "status": 1,
    }
    defaults.update(kwargs)

    class _F:
        pass

    form = _F()
    form.__dict__.update(defaults)
    return form


async def test_recreate_after_delete_revives_row(db):
    created = await ai_agent_endpoint_service.create_endpoint(db, _form())
    await ai_agent_endpoint_service.delete_endpoint(db, created.id)

    recreated = await ai_agent_endpoint_service.create_endpoint(db, _form(name="端点A重建"))

    assert recreated.id == created.id
    assert recreated.deleted == 0
    assert recreated.name == "端点A重建"
    assert await ai_agent_endpoint_repository.get_by_base_url(db, recreated.base_url) is recreated


async def test_register_same_base_url_when_alive_raises(db):
    await ai_agent_endpoint_service.create_endpoint(db, _form())
    with pytest.raises(BusinessException) as ei:
        await ai_agent_endpoint_service.create_endpoint(db, _form(name="端点B"))
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_delete_endpoint_writes_audit(db, monkeypatch):
    from app.database import run_after_commit_callbacks
    from app.models.base import set_current_user_id
    from app.service import ai_agent_endpoint_service as m

    audits = []
    monkeypatch.setattr(
        m,
        "mongo_audit_log_repository",
        SimpleNamespace(create_audit_async=lambda **kw: audits.append(kw)),
    )
    set_current_user_id(5)
    try:
        endpoint = await ai_agent_endpoint_service.create_endpoint(db, _form())
        await ai_agent_endpoint_service.delete_endpoint(db, endpoint.id)
        await run_after_commit_callbacks(db)
    finally:
        set_current_user_id(None)

    assert len(audits) == 1
    assert audits[0]["operator_id"] == 5
    assert audits[0]["target_type"] == "ai_agent_endpoint"
    assert audits[0]["target_id"] == endpoint.id
    assert audits[0]["action"] == "delete"
    assert audits[0]["before_value"]["base_url"] == "https://a2a-revive.example.com"
