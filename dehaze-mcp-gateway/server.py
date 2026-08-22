"""MCP 工具：读取后端 OpenAPI，暴露 3 个元 tool 供大模型按需发现和调用后端 API。

配置传递：
- server 模式：请求头 x-backend-url / x-dehaze-api-key（覆盖默认值）
- CLI 模式：环境变量 MCP_BACKEND_URL / DEHAZE_API_KEY

三个 tool：
- lookup_tool: 搜索后端 API，返回工具名和参数概要
- lookup_tool_param_schema: 查看指定工具的完整参数 schema（含描述/必填/枚举/嵌套结构）
- execute_tool: 调用指定工具，传入参数
"""

import json
import os
import re
import sys

import httpx
from mcp.server.mcpserver import Context, MCPServer

OPENAPI_URL = os.getenv("MCP_OPENAPI_URL", "http://127.0.0.1:8989/v3/api-docs")
BACKEND_URL = os.getenv("MCP_BACKEND_URL", "http://127.0.0.1:8989")
API_KEY = os.getenv("DEHAZE_API_KEY", "dhak_m2m_internal_python_service_key_2024")
PORT = int(os.getenv("MCP_PORT", "8082"))

mcp = MCPServer("dehaze")
_http = httpx.AsyncClient(timeout=30)
_apis: dict[str, dict] = {}
_SPEC: dict = {}

_SKIP_PATHS = ("/login", "/register", "/logout", "/upload", "/api-docs")
_SKIP_METHODS = {"delete", "head", "options"}


def _resolve(node, spec):
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        node = spec
        for p in ref[2:].split("/"):
            node = node.get(p, {})
    return node


def _describe_schema(node, spec):
    """将 JSON Schema 递归转成人类可读文本，展开嵌套 object/array 结构与约束。"""
    node = _resolve(node, spec)
    if not isinstance(node, dict):
        return str(node)
    t = node.get("type", "any")
    parts = []
    if t == "object" and node.get("properties"):
        props = []
        for k, v in node["properties"].items():
            req = "必填" if k in node.get("required", []) else "可选"
            props.append(f"{k}({_describe_schema(v, spec)},{req})")
        parts.append("object{" + ", ".join(props) + "}")
    elif t == "array":
        parts.append(f"array[{_describe_schema(node.get('items', {}), spec)}]")
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


def _schema_example(node):
    """从 JSON Schema 生成示例值，递归展开 $ref，嵌套结构直接产出可用的 example JSON。"""
    node = _resolve(node, _SPEC) if _SPEC else node
    if not isinstance(node, dict):
        return "any"
    t = node.get("type")
    if t == "object":
        return {k: _schema_example(v) for k, v in node.get("properties", {}).items()}
    if t == "array":
        return [_schema_example(node.get("items", {}))]
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
            params = _parse_params(op, spec)
            if any(p["desc"].split(" ")[0] in ("file", "binary") for p in params):
                continue
            name = _tool_name(method, path)
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


def _get_config(ctx: Context = None):
    if ctx and ctx.headers:
        return ctx.headers.get("x-backend-url", BACKEND_URL), ctx.headers.get("x-dehaze-api-key", API_KEY)
    return BACKEND_URL, API_KEY


def _param_names(p):
    return f"{p['name']}*" if p["required"] else p["name"]


@mcp.tool()
async def lookup_tool(query: str, ctx: Context = None) -> str:
    """搜索可用的后端 API 工具，返回工具名、描述和参数名列表（* 为必填）。"""
    if not query or not query.strip():
        return "查询不能为空"
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
async def lookup_tool_param_schema(tool_name: str, ctx: Context = None) -> str:
    """查看指定工具的完整参数 schema（参数含义、必填、枚举、嵌套结构）。"""
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
async def execute_tool(tool_name: str, arguments: str = "{}", ctx: Context = None) -> str:
    """调用指定工具，执行后端 API 调用。arguments 为 JSON 字符串。"""
    api = _apis.get(tool_name)
    if not api:
        return f"工具 {tool_name} 不存在，请先调用 lookup_tool 搜索"

    try:
        kwargs = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return "arguments 不是合法 JSON"

    missing = [p["name"] for p in api["params"] if p["required"] and kwargs.get(p["name"]) is None]
    if missing:
        return f"缺少必填参数: {', '.join(missing)}（可用 lookup_tool_param_schema 查看完整定义）"

    backend_url, api_key = _get_config(ctx)

    path, query, body = api["path"], {}, {}
    for p in api["params"]:
        val = kwargs.get(p["name"])
        if val is None:
            continue
        if p["in"] == "path":
            path = path.replace("{" + p["name"] + "}", str(val))
        elif p["in"] == "query":
            query[p["name"]] = val
        else:
            body[p["name"]] = val

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = await _http.request(api["method"], backend_url + path, params=query, json=body or None, headers=headers)
    text = resp.text
    if resp.status_code >= 500:
        # 透传响应体中的可诊断信息（如熔断/超时原因），而非只给状态码
        return f"后端服务不可用({resp.status_code}): {text[:300] or '无详细信息'}"
    if resp.status_code >= 400:
        return f"错误({resp.status_code}): {text[:500]}"
    if len(text) > 8000:
        return f"响应过长已截断（共{len(text)}字符，仅展示前8000）：\n{text[:8000]}"
    return text


# 启动时加载 OpenAPI
try:
    resp = httpx.get(OPENAPI_URL)
    _SPEC = resp.json()
    _apis = _parse_spec(_SPEC)
    print(f"[MCP] 已加载 {len(_apis)} 个后端 API")
except Exception as e:
    print(f"[MCP] 读取 OpenAPI 失败: {e}")


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
        import asyncio
        name = sys.argv[3]
        args = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        print(asyncio.run(execute_tool(name, json.dumps(args))))
        return

    print(f"未知命令: {cmd}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        _run_cli()
    else:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
