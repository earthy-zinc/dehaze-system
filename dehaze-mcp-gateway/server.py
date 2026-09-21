"""MCP 工具：读取后端 OpenAPI，暴露 3 个元 tool 供大模型按需发现和调用后端 API。

配置传递（server / CLI 模式一致，均通过环境变量，不接受请求头覆盖——
请求头可被任意 MCP 客户端伪造，会把 M2M API Key 发往任意 URL）：
- MCP_BACKEND_URL / M2M_API_KEY / MCP_GATEWAY_KEY / MCP_WRITE_TOOLS / MCP_BIND_HOST

安全：
- 网关持有后端 M2M 凭证，HTTP 访问必须携带 X-MCP-Key 共享密钥（缺失/错误一律 401），
  MCP_GATEWAY_KEY 未配置时拒绝启动
- LLM 可用写操作收敛为白名单（默认仅 GET + post_api_v1_prediction），可用
  MCP_WRITE_TOOLS 扩充

三个 tool：
- lookup_tool: 搜索后端 API，返回工具名和参数概要
- lookup_tool_param_schema: 查看指定工具的完整参数 schema（含描述/必填/枚举/嵌套结构）
- execute_tool: 调用指定工具，传入参数
"""

import asyncio
import hmac
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

# 与其余三端一致从仓库根 .env 读取配置（run.py 启动网关时不注入环境变量，
# 网关需自行加载；已设置的 shell 环境变量优先）
_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ROOT_ENV.is_file():
    for _line in _ROOT_ENV.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

OPENAPI_URL = os.getenv("MCP_OPENAPI_URL", "http://127.0.0.1:8989/v3/api-docs")
BACKEND_URL = os.getenv("MCP_BACKEND_URL", "http://127.0.0.1:8989")
API_KEY = os.getenv("M2M_API_KEY")
PORT = int(os.getenv("MCP_PORT", "8082"))
# 网关持有后端 M2M 凭证，默认仅监听回环地址；跨机部署时显式设置 MCP_BIND_HOST
BIND_HOST = os.getenv("MCP_BIND_HOST", "127.0.0.1")
# 客户端访问网关的共享密钥（X-MCP-Key 请求头），未配置时 HTTP 服务拒绝启动
MCP_GATEWAY_KEY = os.getenv("MCP_GATEWAY_KEY", "")

mcp = FastMCP("dehaze", host=BIND_HOST)
_http = httpx.AsyncClient(timeout=30)
_apis: dict[str, dict] = {}
_SPEC: dict = {}
_load_lock = asyncio.Lock()

_SKIP_PATHS = ("/login", "/register", "/logout", "/upload", "/api-docs")
_SKIP_METHODS = {"delete", "head", "options"}

# LLM 可用的写操作白名单（工具名粒度）：发起去雾预测是文档定义的唯一常规写场景
# （需求规格 §2.3.1 示例流程），其余 POST/PUT 一律收敛为只读，防止提示注入场景下
# LLM 借 admin 身份的 M2M 凭证篡改管理数据。可用 MCP_WRITE_TOOLS（逗号分隔）扩充。
_DEFAULT_WRITE_TOOLS = frozenset({"post_api_v1_prediction"})
_WRITE_TOOLS = _DEFAULT_WRITE_TOOLS | {
    t.strip() for t in os.getenv("MCP_WRITE_TOOLS", "").split(",") if t.strip()
}


def _resolve(node, spec, _depth=0):
    # 循环 $ref（自引用 schema）会导致死循环，解引用次数设上限
    while isinstance(node, dict) and "$ref" in node and _depth < 10:
        ref = node["$ref"]
        node = spec
        for p in ref[2:].split("/"):
            node = node.get(p, {})
        _depth += 1
    return node


def _describe_schema(node, spec, depth=0):
    """将 JSON Schema 递归转成人类可读文本，展开嵌套 object/array 结构与约束。"""
    node = _resolve(node, spec)
    if not isinstance(node, dict):
        return str(node)
    t = node.get("type", "any")
    if depth >= 5 and t in ("object", "array"):
        return f"{t}(…)"
    parts = []
    if t == "object" and node.get("properties"):
        props = []
        for k, v in node["properties"].items():
            req = "必填" if k in node.get("required", []) else "可选"
            props.append(f"{k}({_describe_schema(v, spec, depth + 1)},{req})")
        parts.append("object{" + ", ".join(props) + "}")
    elif t == "array":
        parts.append(f"array[{_describe_schema(node.get('items', {}), spec, depth + 1)}]")
    else:
        parts.append(t)
    if node.get("enum"):
        parts.append(f"枚举:{node['enum']}")
    if node.get("default") is not None:
        parts.append(f"默认:{node['default']}")
    desc = node.get("description")
    if desc:
        parts.append(desc)
    return " ".join(parts)


def _is_binary_schema(schema) -> bool:
    return isinstance(schema, dict) and (schema.get("format") == "binary" or schema.get("type") == "file")


def _has_binary_param(op, spec) -> bool:
    """API 是否含文件/二进制参数，或请求体不是 JSON（multipart/octet-stream 等 LLM 无法构造的形态）。"""
    for p in op.get("parameters", []):
        p = _resolve(p, spec)
        if _is_binary_schema(_resolve(p.get("schema", {}), spec)):
            return True
    body = _resolve(op.get("requestBody"), spec)
    if body:
        content = body.get("content", {})
        if "application/json" not in content:
            return True
        root = _resolve(content["application/json"].get("schema", {}), spec)
        return any(_is_binary_schema(_resolve(prop, spec)) for prop in root.get("properties", {}).values())
    return False


def _parse_params(op, spec):
    """提取参数完整信息：名称、位置、schema（含描述/必填/枚举/嵌套结构/示例）。"""
    params = []
    for p in op.get("parameters", []):
        p = _resolve(p, spec)
        schema = _resolve(p.get("schema", {}), spec)
        params.append({
            "name": p.get("name", ""),
            "in": p.get("in", "query"),
            "required": p.get("required", False),
            "desc": _describe_schema(schema, spec),
            "schema": schema,
        })
    body = _resolve(op.get("requestBody"), spec)
    if body:
        root = _resolve(body.get("content", {}).get("application/json", {}).get("schema", {}), spec)
        for name, prop in root.get("properties", {}).items():
            params.append({
                "name": name,
                "in": "body",
                "required": name in root.get("required", []),
                "desc": _describe_schema(prop, spec),
                "schema": _resolve(prop, spec),
            })
    return params


def _schema_example(node, depth=0):
    """从 JSON Schema 生成示例值，递归展开 $ref，嵌套结构直接产出可用的 example JSON。"""
    node = _resolve(node, _SPEC) if _SPEC else node
    if not isinstance(node, dict):
        return "any"
    t = node.get("type")
    if depth >= 5 and t in ("object", "array"):
        return "…"
    if t == "object":
        return {k: _schema_example(v, depth + 1) for k, v in node.get("properties", {}).items()}
    if t == "array":
        return [_schema_example(node.get("items", {}), depth + 1)]
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    if node.get("enum"):
        return node["enum"][0]
    return "string"


def _tool_name(method, path):
    segs = []
    for s in path.split("/"):
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            segs.append("by_" + s[1:-1].lower())
        else:
            segs.append(re.sub(r"[^a-zA-Z0-9_]", "_", s).lower())
    return f"{method}_{'_'.join(segs)}"


def _parse_spec(spec: dict) -> dict[str, dict]:
    apis = {}
    for path, methods in spec.get("paths", {}).items():
        if any(path.endswith(s) for s in _SKIP_PATHS):
            continue
        for method, op in methods.items():
            if method in _SKIP_METHODS or method not in ("get", "post", "put", "patch"):
                continue
            if _has_binary_param(op, spec):
                continue
            name = _tool_name(method, path)
            # 写操作白名单：非 GET 工具仅在白名单内才注册，LLM 从源头就不可见
            if method != "get" and name not in _WRITE_TOOLS:
                continue
            params = _parse_params(op, spec)
            tags = op.get("tags", []) or []
            desc = op.get("summary") or op.get("description") or f"{method.upper()} {path}"
            apis[name] = {"path": path, "method": method.upper(), "params": params,
                          "description": desc, "namespace": tags[0] if tags else "other"}
    return apis


_COMMON_PATH_TOKENS = {"api", "v1", "v2", "v3", "admin", "backend"}


def _tokenize(text):
    """分词：英文按单词，中文按双字 bigram（避免单字噪声，如"日"命中"每日签到"）。"""
    text = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    for cjk in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(cjk) == 1:
            tokens.add(cjk)
        else:
            tokens.update(cjk[i:i + 2] for i in range(len(cjk) - 1))
    return tokens


def _match_score(q_tokens, doc_tokens, weight=1.0):
    """doc 与 query 的匹配分：精确命中 weight，前缀命中（词形变化如 model→models）weight*0.5。"""
    score = 0.0
    for q in q_tokens:
        if q in doc_tokens:
            score += weight
        elif len(q) >= 3 and any(d.startswith(q) for d in doc_tokens):
            score += weight * 0.5
    return score


def _search_apis(query, limit=10):
    """按相关度搜索 API，返回 [(score, name, api)] 降序。

    路径风格 query（含 /）优先按 path 子串匹配；token 匹配时过滤公共路径前缀（api/v1），
    搜索源加权：工具名+path（权重3），描述+命名空间（权重1）。
    得分按 query 词数归一化：长 query 必须命中多数词才有高分，
    避免命中单个常见词（如"去雾"）即宽泛召回整个模块；同分按工具名字典序，排序稳定可复现。
    """
    q = query.strip().lower()
    if not q:
        return []

    # 路径风格（如 /api/v1/model、presets/{id}）：直接按 path 子串匹配，最精确
    if "/" in q:
        scored = [(3.0, n, a) for n, a in _apis.items() if q in a["path"].lower() or a["path"].lower() in q]
        if scored:
            scored.sort(reverse=True)
            return scored[:limit]

    q_tokens = _tokenize(q) - _COMMON_PATH_TOKENS
    if not q_tokens:
        return []
    scored = []
    for n, a in _apis.items():
        name_doc = _tokenize(f"{n} {a['path']}") - _COMMON_PATH_TOKENS
        param_doc = _tokenize(" ".join(p["name"] for p in a["params"])) - _COMMON_PATH_TOKENS
        desc_doc = _tokenize(a["description"])
        # namespace（分类标签）是弱信号；参数描述只是"蹭词"辅助信号（如"月度去雾配额"），
        # 权重层层递减：name(3) > 参数名(2) > 描述(1) > namespace(0.5) > 参数描述(0.3)
        ns_doc = _tokenize(a.get("namespace", ""))
        param_desc_doc = _tokenize(" ".join(p["desc"] for p in a["params"]))
        score = (_match_score(q_tokens, name_doc, 3.0)
                 + _match_score(q_tokens, param_doc, 2.0)
                 + _match_score(q_tokens, desc_doc, 1.0)
                 + _match_score(q_tokens, ns_doc, 0.5)
                 + _match_score(q_tokens, param_desc_doc, 0.3))
        if score > 0:
            # 归一化：按 query 词数摊分，长 query 需命中多数词才有高分，避免单词命中即宽泛召回
            scored.append((score / len(q_tokens), n, a))
    scored.sort(reverse=True)
    return scored[:limit]


async def _ensure_apis() -> bool:
    """注册表为空时按需补载一次（run.py 启动 java→mcp 无健康等待，java 冷启动期间
    首次加载必然失败，不能让网关以空注册表静默服务）。"""
    if _apis:
        return True
    async with _load_lock:
        if _apis:
            return True
        return await asyncio.to_thread(_load_apis)


def _load_apis() -> bool:
    global _SPEC, _apis
    try:
        resp = httpx.get(OPENAPI_URL, timeout=10)
        resp.raise_for_status()
        _SPEC = resp.json()
        _apis = _parse_spec(_SPEC)
        print(f"[MCP] 已加载 {len(_apis)} 个后端 API")
        return True
    except Exception as e:
        print(f"[MCP] 读取 OpenAPI 失败: {e}")
        return False


def _param_names(p):
    return f"{p['name']}*" if p["required"] else p["name"]


@mcp.tool()
async def lookup_tool(query: str) -> str:
    """搜索可用的后端 API 工具，返回工具名、描述和参数名列表（* 为必填）。"""
    if not query or not query.strip():
        return "查询不能为空"
    if not await _ensure_apis():
        return "后端 OpenAPI 未加载，请确认后端服务已启动后重试"
    scored = _search_apis(query)
    if not scored:
        # 无匹配时给出相近候选（按工具名/描述子串包含排序），避免 LLM 反复换词
        q = query.strip().lower()
        near = sorted(
            (n, a) for n, a in _apis.items()
            if q in n.lower() or q in a["description"].lower() or q in a.get("namespace", "").lower()
        )[:3]
        if near:
            hint = "；相近候选: " + ", ".join(n for n, _ in near)
        else:
            hint = "；可尝试更简短的关键词"
        return f"无匹配工具{hint}"
    lines = []
    for _, n, a in scored:
        params = ", ".join(_param_names(p) for p in a["params"])
        lines.append(f"{n}: {a['description']} | 参数: {params}" if params else f"{n}: {a['description']} | 无参数")
    return "\n".join(lines)


@mcp.tool()
async def lookup_tool_param_schema(tool_name: str) -> str:
    """查看指定工具的完整参数 schema（参数含义、必填、枚举、嵌套结构）。"""
    if not await _ensure_apis():
        return "后端 OpenAPI 未加载，请确认后端服务已启动后重试"
    api = _apis.get(tool_name)
    if not api:
        return f"工具 {tool_name} 不存在，请先调用 lookup_tool 搜索"
    return json.dumps({
        "tool_name": tool_name,
        "description": api["description"],
        "method": api["method"],
        "path": api["path"],
        "namespace": api["namespace"],
        "params": [
            {"name": p["name"], "location": p["in"], "required": p["required"],
             "schema": p["desc"], "example": _schema_example(p.get("schema"))}
            for p in api["params"]
        ],
        "arguments_example": {
            p["name"]: _schema_example(p.get("schema")) for p in api["params"] if p["required"]
        },
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def execute_tool(tool_name: str, arguments: str = "{}") -> str:
    """调用指定工具，执行后端 API 调用。arguments 为 JSON 字符串。"""
    if not await _ensure_apis():
        return "后端 OpenAPI 未加载，请确认后端服务已启动后重试"
    api = _apis.get(tool_name)
    if not api:
        return f"工具 {tool_name} 不存在，请先调用 lookup_tool 搜索"

    try:
        kwargs = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return "arguments 不是合法 JSON"
    if not isinstance(kwargs, dict):
        return "arguments 必须是 JSON 对象"

    missing = [p["name"] for p in api["params"] if p["required"] and kwargs.get(p["name"]) is None]
    if missing:
        return f"缺少必填参数: {', '.join(missing)}（可用 lookup_tool_param_schema 查看完整定义）"

    path, query, body = api["path"], {}, {}
    for p in api["params"]:
        val = kwargs.get(p["name"])
        if val is None:
            continue
        if p["in"] == "path":
            # 参数值来自 LLM（最终源自用户输入），必须整体编码，防止 / 等字符注入额外路径段
            path = path.replace("{" + p["name"] + "}", quote(str(val), safe=""))
        elif p["in"] == "query":
            query[p["name"]] = val
        else:
            body[p["name"]] = val

    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    resp = await _http.request(api["method"], BACKEND_URL + path, params=query, json=body or None, headers=headers)
    text = resp.text
    if resp.status_code >= 500:
        # 透传响应体中的可诊断信息（如熔断/超时原因），而非只给状态码
        return f"后端服务不可用({resp.status_code}): {text[:300] or '无详细信息'}"
    if resp.status_code >= 400:
        return f"错误({resp.status_code}): {text[:500]}"
    if len(text) > 8000:
        return f"响应过长已截断（共{len(text)}字符，仅展示前8000）：\n{text[:8000]}"
    return text


# 启动时加载 OpenAPI：有限重试（run.py 启动顺序 java→mcp 但无健康等待），
# 仍失败则依赖首次工具调用时的按需补载，注册表为空不提供任何工具
for _i in range(3):
    if _load_apis():
        break
    if _i < 2:
        time.sleep(2)


# MCP 协议层 ASGI 应用（含 lifespan 会话管理，由 uvicorn 拉起）
# host 在 FastMCP 构造时给出（1.x 的 streamable_http_app 不接受 host 参数，
# 它据此决定 localhost 的 DNS rebinding 白名单）
_mcp_asgi = mcp.streamable_http_app()


async def _auth_asgi_app(scope, receive, send):
    """网关共享密钥鉴权：校验 X-MCP-Key 请求头，缺失/错误一律 401。

    网关持有后端 M2M 凭证，不能裸奔；常量时间比较防时序攻击逐位猜测。
    """
    if scope["type"] == "http":
        headers = dict(scope.get("headers", []))
        provided = headers.get(b"x-mcp-key", b"").decode("latin-1")
        if not MCP_GATEWAY_KEY or not hmac.compare_digest(
            provided.encode("utf-8"), MCP_GATEWAY_KEY.encode("utf-8")
        ):
            resp = JSONResponse(
                {"code": "A0231", "msg": "未授权：网关共享密钥缺失或错误", "data": None},
                status_code=401,
            )
            await resp(scope, receive, send)
            return
    await _mcp_asgi(scope, receive, send)


def _run_cli():
    if not _apis:
        print(f"[MCP] 启动失败：无法从 {OPENAPI_URL} 加载 OpenAPI，请检查后端是否可用")
        sys.exit(1)
    if len(sys.argv) < 3:
        print("用法: python server.py cli lookup <keyword>")
        print("      python server.py cli schema <tool_name>")
        print("      python server.py cli execute <tool_name> '{\"key\":\"value\"}'")
        print("      python server.py cli list")
        sys.exit(1)

    cmd = sys.argv[2]

    if cmd == "list":
        for n, a in sorted(_apis.items()):
            print(f"  {n}: {a['description']}")
        print(f"\n共 {len(_apis)} 个工具")
        return

    if cmd == "lookup":
        keyword = sys.argv[3] if len(sys.argv) > 3 else ""
        for _, n, a in _search_apis(keyword):
            params = ", ".join(_param_names(p) for p in a["params"])
            print(f"  {n}: {a['description']} | 参数: {params}" if params else f"  {n}: {a['description']} | 无参数")
        if not _search_apis(keyword):
            print("无匹配工具")
        return

    if cmd == "schema":
        name = sys.argv[3]
        api = _apis.get(name)
        if not api:
            print(f"工具 {name} 不存在")
            return
        print(f"描述: {api['description']}")
        print(f"方法: {api['method']} {api['path']}")
        for p in api["params"]:
            req = "必填" if p["required"] else "可选"
            print(f"  - {p['name']} (位置:{p['in']}, {req}) {p['desc']}")
        return

    if cmd == "execute":
        name = sys.argv[3]
        args = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        print(asyncio.run(execute_tool(name, json.dumps(args))))
        return

    print(f"未知命令: {cmd}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        _run_cli()
    else:
        if not MCP_GATEWAY_KEY:
            print("[MCP] 启动失败：MCP_GATEWAY_KEY 未配置（网关持有 M2M 凭证，必须启用共享密钥鉴权）")
            sys.exit(1)
        import uvicorn

        uvicorn.run(_auth_asgi_app, host=BIND_HOST, port=PORT)
