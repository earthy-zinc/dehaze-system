"""MCP Server 契约字段增强单测：credential_configured（凭据已配置标记）"""

import pytest

pytestmark = pytest.mark.requires_db

from app.models.schema.ai_mcp import McpCredentialForm, McpServerCreate
from app.service.ai_mcp.ai_mcp_server_service import AiMcpServerService


class TestCredentialConfigured:
    async def test_new_server_not_configured(self, db):
        result = await AiMcpServerService().create_server(
            db, McpServerCreate(name="cred-new")
        )
        assert result.credential_configured is False

    async def test_credentials_update_marks_configured(self, db):
        svc = AiMcpServerService()
        created = await svc.create_server(db, McpServerCreate(name="cred-set"))
        await svc.update_credentials(
            db, created.id, McpCredentialForm(api_key="sk-test")
        )
        result = await svc.get_server(db, created.id)
        assert result.credential_configured is True

    async def test_extra_only_credentials_mark_configured(self, db):
        svc = AiMcpServerService()
        created = await svc.create_server(db, McpServerCreate(name="cred-extra"))
        await svc.update_credentials(
            db, created.id, McpCredentialForm(extra={"token": "abc"})
        )
        result = await svc.get_server(db, created.id)
        assert result.credential_configured is True
