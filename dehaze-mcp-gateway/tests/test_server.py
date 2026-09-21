"""dehaze-mcp-gateway 单元测试：不依赖真实后端/网关进程。

运行方式（网关目录下）：
    uv run --with pytest pytest tests -q
"""
import asyncio
import json
import time

import pytest

time.sleep = lambda s: None  # 跳过模块导入时的启动重试等待
import server  # noqa: E402


# ---------- 测试用 OpenAPI 规格 ----------

def _make_spec():
    """构造覆盖过滤管道/参数映射/递归 schema 的最小 OpenAPI 规格。"""
    return {
        "components": {
            "schemas": {
                # 自引用树形结构（菜单树类），验证递归深度上限
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "children": {"type": "array", "items": {"$ref": "#/components/schemas/TreeNode"}},
                    },
                },
                # 循环引用：A -> B -> A
                "LoopA": {"type": "object", "properties": {"b": {"$ref": "#/components/schemas/LoopB"}}},
                "LoopB": {"type": "object", "properties": {"a": {"$ref": "#/components/schemas/LoopA"}}},
            }
        },
        "paths": {
            "/api/v1/prediction": {
                "post": {
                    "tags": ["去雾处理"],
                    "summary": "执行模型预测（去雾处理，异步）",
                    "requestBody": {
                        "content": {"application/json": {"schema": {
                            "type": "object",
                            "properties": {"algorithmId": {"type": "integer"}, "fileId": {"type": "string"}},
                            "required": ["algorithmId"],
                        }}}
                    },
                }
            },
            "/api/v1/presets/{id}": {
                "put": {
                    "tags": ["预设管理"],
                    "summary": "更新预设",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "keyword", "in": "query", "schema": {"type": "string"}},
                    ],
                    "requestBody": {
                        "content": {"application/json": {"schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        }}}
                    },
                }
            },
            "/api/v1/prediction/{taskId}": {
                "get": {
                    "tags": ["去雾处理"],
                    "summary": "获取预测任务",
                    "parameters": [
                        {"name": "taskId", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                },
                "delete": {"tags": ["去雾处理"], "summary": "删除任务"},
            },
            "/api/v1/files/upload": {
                "post": {
                    "tags": ["文件"],
                    "summary": "上传文件",
                    "requestBody": {"content": {
                        "multipart/form-data": {"schema": {
                            "type": "object",
                            "properties": {"file": {"type": "string", "format": "binary"}},
                        }}
                    }},
                }
            },
            # springdoc 风格：MultipartFile 参数在 query 参数位置携带 format binary
            "/api/v1/import": {
                "post": {
                    "tags": ["导入"],
                    "summary": "导入数据",
                    "parameters": [
                        {"name": "file", "in": "query", "required": True,
                         "schema": {"type": "string", "format": "binary"}}
                    ],
                }
            },
            "/api/v1/login": {"post": {"tags": ["认证"], "summary": "登录"}},
            "/api/v1/menus/tree": {
                "get": {
                    "tags": ["菜单"],
                    "summary": "菜单树",
                    "responses": {"200": {"content": {"application/json": {"schema": {
                        "$ref": "#/components/schemas/TreeNode"}}}}},
                }
            },
            "/api/v1/loop": {
                "get": {
                    "tags": ["循环"],
                    "summary": "循环引用 schema",
                    "responses": {"200": {"content": {"application/json": {"schema": {
                        "$ref": "#/components/schemas/LoopA"}}}}},
                }
            },
        },
    }


@pytest.fixture()
def apis(monkeypatch):
    # 默认写白名单仅放行去雾预测；参数映射类用例需要 PUT 工具，临时扩充
    monkeypatch.setattr(
        server, "_WRITE_TOOLS", server._DEFAULT_WRITE_TOOLS | {"put_api_v1_presets_by_id"}
    )
    server._SPEC = _make_spec()
    server._apis = server._parse_spec(_make_spec())
    yield server._apis
    server._apis, server._SPEC = {}, {}


# ---------- 写操作白名单 ----------

def test_default_whitelist_keeps_documented_prediction_write(monkeypatch):
    """发起去雾预测是文档定义的唯一常规 LLM 写场景，默认白名单必须放行。"""
    monkeypatch.setattr(server, "_WRITE_TOOLS", server._DEFAULT_WRITE_TOOLS)
    apis = server._parse_spec(_make_spec())
    assert "post_api_v1_prediction" in apis
    assert apis["post_api_v1_prediction"]["method"] == "POST"


def test_default_whitelist_filters_other_writes(monkeypatch):
    """非白名单 POST/PUT 一律不注册，LLM 从源头不可见（防提示注入改管理数据）。"""
    monkeypatch.setattr(server, "_WRITE_TOOLS", server._DEFAULT_WRITE_TOOLS)
    apis = server._parse_spec(_make_spec())
    assert "put_api_v1_presets_by_id" not in apis
    # GET 不受白名单约束
    assert "get_api_v1_prediction_by_taskid" in apis


def test_write_whitelist_extensible(monkeypatch):
    monkeypatch.setattr(
        server, "_WRITE_TOOLS", server._DEFAULT_WRITE_TOOLS | {"put_api_v1_presets_by_id"}
    )
    apis = server._parse_spec(_make_spec())
    assert "put_api_v1_presets_by_id" in apis


# ---------- 网关共享密钥鉴权（X-MCP-Key） ----------

def _http_scope(headers: dict):
    return {
        "type": "http", "method": "POST", "path": "/mcp",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }


def _call_asgi(app, scope):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    asyncio.run(app(scope, receive, send))
    return sent


@pytest.fixture()
def gateway_key(monkeypatch):
    monkeypatch.setattr(server, "MCP_GATEWAY_KEY", "test-key-123")


def test_auth_missing_header_rejected(gateway_key):
    sent = _call_asgi(server._auth_asgi_app, _http_scope({}))
    assert sent[0]["status"] == 401


def test_auth_wrong_key_rejected(gateway_key):
    sent = _call_asgi(server._auth_asgi_app, _http_scope({"X-MCP-Key": "wrong"}))
    assert sent[0]["status"] == 401


def test_auth_no_gateway_key_configured_rejects_all(monkeypatch):
    """密钥未配置时全部拒绝（fail closed），不因空密钥退化为无鉴权。"""
    monkeypatch.setattr(server, "MCP_GATEWAY_KEY", "")
    sent = _call_asgi(server._auth_asgi_app, _http_scope({"X-MCP-Key": "anything"}))
    assert sent[0]["status"] == 401


def test_auth_correct_key_passthrough(gateway_key, monkeypatch):
    passed = []

    async def stub_asgi(scope, receive, send):
        passed.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    monkeypatch.setattr(server, "_mcp_asgi", stub_asgi)
    sent = _call_asgi(server._auth_asgi_app, _http_scope({"X-MCP-Key": "test-key-123"}))
    assert sent[0]["status"] == 200
    assert passed == ["/mcp"]


# ---------- 过滤管道 ----------

def test_skip_delete_method(apis):
    assert "delete_api_v1_prediction_by_taskid" not in apis


def test_skip_login_path(apis):
    assert not any("/login" in a["path"] for a in apis.values())


def test_skip_multipart_request_body(apis):
    assert not any(a["path"] == "/api/v1/files/upload" for a in apis.values())


def test_skip_binary_query_param(apis):
    assert not any(a["path"] == "/api/v1/import" for a in apis.values())


def test_keep_normal_apis_with_param_mapping(apis):
    api = apis["put_api_v1_presets_by_id"]
    assert api["method"] == "PUT"
    locs = {p["name"]: p["in"] for p in api["params"]}
    assert locs == {"id": "path", "keyword": "query", "name": "body"}
    req = {p["name"] for p in api["params"] if p["required"]}
    assert req == {"id", "name"}


# ---------- 递归 / 循环 schema 健壮性 ----------

def test_recursive_schema_describe_terminates(apis):
    tree = apis["get_api_v1_menus_tree"]
    spec = _make_spec()
    ref = spec["paths"]["/api/v1/menus/tree"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    text = server._describe_schema(ref, spec)
    assert "…" in text  # 深度截断标记，且未抛 RecursionError


def test_recursive_schema_example_terminates(apis):
    node = {"$ref": "#/components/schemas/TreeNode"}
    example = server._schema_example(node)
    assert isinstance(example, dict)
    assert example["id"] == 1
    children = example["children"]
    assert isinstance(children, list)
    assert isinstance(children[0], dict)
    grandchildren = children[0]["children"]
    assert isinstance(grandchildren, list)
    assert isinstance(grandchildren[0], dict)
    # 递归结构在深度上限收敛（object/array 交替，depth>=5 返回 "…"），未爆栈
    assert grandchildren[0]["children"] == "…"


def test_circular_ref_resolve_terminates():
    spec = _make_spec()
    node = server._resolve({"$ref": "#/components/schemas/LoopA"}, spec)
    assert isinstance(node, dict)  # 循环解引用在 10 次上限处停止，不死循环


# ---------- 工具名与搜索 ----------

def test_tool_name_rules():
    assert server._tool_name("put", "/api/v1/presets/{id}") == "put_api_v1_presets_by_id"
    assert server._tool_name("get", "/api/v1/prediction/{taskId}") == "get_api_v1_prediction_by_taskid"
    assert server._tool_name("get", "/api/v1/a-b/c.d") == "get_api_v1_a_b_c_d"


def test_search_chinese_bigram(apis):
    names = [n for _, n, _ in server._search_apis("预测任务")]
    assert "get_api_v1_prediction_by_taskid" in names


def test_search_path_style_query(apis):
    names = [n for _, n, _ in server._search_apis("/api/v1/presets/{id}")]
    assert names[0] == "put_api_v1_presets_by_id"


def test_search_empty_and_deterministic(apis):
    assert server._search_apis("") == []
    first = server._search_apis("presets")
    assert first == server._search_apis("presets")  # 同分排序稳定可复现


# ---------- execute_tool ----------

class _FakeResp:
    def __init__(self, status_code=200, text="{}"):
        self.status_code = status_code
        self.text = text


@pytest.fixture()
def http_capture(monkeypatch):
    calls = {}
    _json = json  # fake_request 的 json 形参会遮蔽模块，提前捕获

    async def fake_request(method, url, params=None, json=None, headers=None):
        calls.update(method=method, url=url, params=params, json_body=json, headers=headers)
        return _FakeResp(200, _json.dumps({"code": "00000", "data": "ok"}))

    monkeypatch.setattr(server._http, "request", fake_request)
    return calls


def test_execute_unknown_tool(apis):
    out = asyncio.run(server.execute_tool("no_such_tool", "{}"))
    assert "不存在" in out


@pytest.mark.parametrize("bad", ["not-json", "[1,2]", '"str"', "42"])
def test_execute_rejects_non_object_arguments(apis, bad):
    out = asyncio.run(server.execute_tool("get_api_v1_prediction_by_taskid", bad))
    assert ("不是合法 JSON" in out) or ("必须是 JSON 对象" in out)


def test_execute_missing_required(apis):
    out = asyncio.run(server.execute_tool("put_api_v1_presets_by_id", json.dumps({"name": "x"})))
    assert "缺少必填参数" in out and "id" in out


def test_execute_param_mapping_and_auth_header(apis, http_capture):
    out = asyncio.run(server.execute_tool(
        "put_api_v1_presets_by_id", json.dumps({"id": 3, "name": "n", "keyword": "k"})))
    assert http_capture["method"] == "PUT"
    assert http_capture["url"].endswith("/api/v1/presets/3")
    assert http_capture["params"] == {"keyword": "k"}
    assert http_capture["json_body"] == {"name": "n"}
    assert http_capture["headers"]["Authorization"].startswith("Bearer dhak_")
    assert '"code": "00000"' in out or '"code":"00000"' in out


def test_execute_path_traversal_encoded(apis, http_capture):
    """路径参数含 / 与 .. 必须整体编码，不能注入额外路径段。"""
    evil = "../../api/v1/admin/users"
    asyncio.run(server.execute_tool(
        "get_api_v1_prediction_by_taskid", json.dumps({"taskId": evil})))
    url = http_capture["url"]
    assert "../" not in url
    assert "%2F" in url.upper()  # 整体编码，未注入额外路径段
    assert http_capture["url"].startswith(server.BACKEND_URL + "/api/v1/prediction/")


@pytest.mark.parametrize("status,frag", [
    (500, "后端服务不可用(500)"),
    (502, "后端服务不可用(502)"),
    (400, "错误(400)"),
    (401, "错误(401)"),
])
def test_execute_backend_error_passthrough(apis, monkeypatch, status, frag):
    async def fake_request(method, url, **kw):
        return _FakeResp(status, '{"msg":"boom"}')

    monkeypatch.setattr(server._http, "request", fake_request)
    out = asyncio.run(server.execute_tool("get_api_v1_prediction_by_taskid", '{"taskId": "t1"}'))
    assert frag in out and "boom" in out


def test_execute_long_response_truncated(apis, monkeypatch):
    async def fake_request(method, url, **kw):
        return _FakeResp(200, "x" * 9000)

    monkeypatch.setattr(server._http, "request", fake_request)
    out = asyncio.run(server.execute_tool("get_api_v1_prediction_by_taskid", '{"taskId": "t1"}'))
    assert "响应过长已截断" in out and len(out) < 9000


# ---------- 启动加载 ----------

def test_load_apis_failure_returns_false(monkeypatch):
    def fake_get(*a, **kw):
        raise ConnectionError("refused")

    monkeypatch.setattr(server.httpx, "get", fake_get)
    assert server._load_apis() is False


def test_load_apis_success_parses(monkeypatch):
    monkeypatch.setattr(server.httpx, "get", lambda *a, **kw: type("R", (), {
        "raise_for_status": lambda self: None, "json": lambda self: _make_spec()})())
    old = (server._apis, server._SPEC)
    try:
        assert server._load_apis() is True
        assert "post_api_v1_prediction" in server._apis  # 默认写白名单内
    finally:
        server._apis, server._SPEC = old
