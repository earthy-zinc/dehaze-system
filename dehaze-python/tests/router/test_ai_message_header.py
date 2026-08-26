import pytest

pytestmark = pytest.mark.api


def test_idempotency_key_header_required_in_schema(app):
    schema = app.openapi()
    path = "/api/v1/ai/conversations/{conv_id}/messages"
    op = schema["paths"][path]["post"]
    params = {p["name"]: p for p in op["parameters"]}
    header = params.get("Idempotency-Key")
    assert header is not None, "Idempotency-Key 头应在 schema 中声明"
    assert header["in"] == "header"
    assert header["required"] is True
