"""MCP 网关 live 集成验证：dehaze-python venv（mcp 1.24 客户端）连真实网关（同版本 server）。

前置：java(8989) + mcp(8082) 已启动（网关需 MCP_GATEWAY_KEY 已配置）。
运行（仓库根目录）：
    /data/workspace/dehaze-system/dehaze-python/.venv/bin/python dehaze-mcp-gateway/tests/live_test.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

GW_URL = "http://127.0.0.1:8082/mcp"


def _load_gateway_key() -> str:
    """取网关共享密钥：环境变量优先，回退仓库根 .env，都没有则终止。"""
    key = os.getenv("MCP_GATEWAY_KEY")
    if not key:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MCP_GATEWAY_KEY="):
                key = line.partition("=")[2].strip()
                break
    if not key:
        print("[FAIL] MCP_GATEWAY_KEY 未配置（环境变量或根 .env）")
        sys.exit(1)
    return key


_KEY = _load_gateway_key()

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{mark}] {name}" + (f" | {detail}" if detail else ""))


async def call(session, tool, args):
    result = await session.call_tool(tool, args)
    for block in result.content:
        if hasattr(block, "text"):
            return block.text
    return str(result.content)


async def main():
    # 0. 网关共享密钥鉴权（原始 HTTP 层，无需 MCP 会话）
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(GW_URL, headers={}, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        check("无 X-MCP-Key 被拒 401", r.status_code == 401, str(r.status_code))
        r = await client.post(GW_URL, headers={"X-MCP-Key": "wrong-key"}, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        check("错误 X-MCP-Key 被拒 401", r.status_code == 401, str(r.status_code))
        r = await client.post(GW_URL, headers={"X-MCP-Key": _KEY}, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        check("正确 X-MCP-Key 通过鉴权", r.status_code != 401, str(r.status_code))

    t0 = time.perf_counter()
    async with streamablehttp_client(GW_URL, headers={"X-MCP-Key": _KEY}) as (read, write, _):
        connect_ms = (time.perf_counter() - t0) * 1000
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"# 客户端与网关握手（两端 mcp 同版本）: {connect_ms:.0f}ms")

            # 1. tools/list：3 个元 tool
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            check("tools/list 返回 3 个元 tool",
                  names == {"lookup_tool", "lookup_tool_param_schema", "execute_tool"},
                  str(sorted(names)))

            # 1.1 写操作白名单：非白名单 POST/PUT/PATCH 不注册，POST 仅限去雾预测
            write_tools = [n for n in names if n.startswith(("post_", "put_", "patch_"))]
            check("写工具仅白名单内可见",
                  all(n in {"post_api_v1_prediction"} for n in write_tools),
                  str(write_tools))

            # 2. lookup_tool：中文检索
            out = await call(session, "lookup_tool", {"query": "去雾预测"})
            check("lookup_tool 中文检索命中", "prediction" in out.lower(), out.splitlines()[0] if out else "空")

            # 3. lookup_tool 空查询 / 无匹配
            out = await call(session, "lookup_tool", {"query": "   "})
            check("lookup_tool 空查询拒绝", "不能为空" in out)
            out = await call(session, "lookup_tool", {"query": "zzz不存在的工具xyz"})
            check("lookup_tool 无匹配给出候选提示", "无匹配工具" in out)

            # 4. lookup_tool_param_schema
            tool_name = "get_api_v1_prediction_by_taskid"
            out = await call(session, "lookup_tool_param_schema", {"tool_name": tool_name})
            ok = tool_name in out and "taskId" in out
            check("param_schema 返回参数定义", ok, "含 method/path/params" if ok else out[:120])
            out = await call(session, "lookup_tool_param_schema", {"tool_name": "no_such"})
            check("param_schema 未知工具提示", "不存在" in out)

            # 5. execute_tool：本地校验
            out = await call(session, "execute_tool", {"tool_name": "no_such", "arguments": "{}"})
            check("execute 未知工具提示", "不存在" in out)
            out = await call(session, "execute_tool", {"tool_name": tool_name, "arguments": "not-json"})
            check("execute 非法 JSON 拒绝", "不是合法 JSON" in out)
            out = await call(session, "execute_tool", {"tool_name": tool_name, "arguments": "[1,2]"})
            check("execute 非 JSON 对象拒绝", "必须是 JSON 对象" in out)
            out = await call(session, "execute_tool", {"tool_name": "post_api_v1_prediction", "arguments": "{}"})
            check("execute 缺必填参数提示", "缺少必填参数" in out)
            # 写白名单：非白名单 PUT 工具不注册，LLM 不可见
            out = await call(session, "execute_tool", {"tool_name": "put_api_v1_presets_by_id", "arguments": "{}"})
            check("execute 非白名单写工具不可见", "不存在" in out)

            # 6. 对抗：路径注入（../ 与额外路径段）——应被整体编码，后端 4xx/404，而非执行到其他端点
            evil = "../api/v1/admin/users"
            out = await call(session, "execute_tool",
                             {"tool_name": tool_name, "arguments": json.dumps({"taskId": evil})})
            check("execute 路径注入被编码（未跨端点执行）",
                  ("错误(" in out or "不可用" in out), out[:100])

            # 7. 对抗：超长 query / 特殊字符透传
            out = await call(session, "lookup_tool", {"query": "去" * 500})
            check("lookup 超长 query 不崩溃", isinstance(out, str))
            out = await call(session, "execute_tool",
                             {"tool_name": tool_name,
                              "arguments": json.dumps({"taskId": "<script>alert(1)</script>"})})
            check("execute 特殊字符透传不崩溃（错误透传）", ("错误(" in out or "不可用" in out), out[:80])

            # 8. 真实只读调用：预测日志列表（M2M key 认证 + 业务信封透传）
            out = await call(session, "execute_tool",
                             {"tool_name": "get_api_v1_prediction_logs",
                              "arguments": json.dumps({"pageNum": 1, "pageSize": 1})})
            ok = '"00000"' in out or "错误(" in out
            check("execute 真实后端调用", ok, out[:120].replace("\n", " "))

            # 9. 性能烟测：20 次串行 lookup
            t0 = time.perf_counter()
            for _ in range(20):
                await call(session, "lookup_tool", {"query": "算法列表"})
            avg_ms = (time.perf_counter() - t0) / 20 * 1000
            check("性能：lookup 平均延迟 < 200ms", avg_ms < 200, f"{avg_ms:.1f}ms/次")

    # 10. dehaze-python 生产客户端路径（McpGatewayClient）联通：会话装配含 initialize 前置
    os.environ.setdefault("MCP_GATEWAY_URL", GW_URL)
    os.environ.setdefault("MCP_GATEWAY_KEY", _KEY)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "dehaze-python"))
    from app.infrastructure.clients.mcp_gateway_client import mcp_gateway_client

    tools = await mcp_gateway_client.list_tools()
    names = sorted(t["name"] for t in tools)
    check("McpGatewayClient.list_tools 返回 3 个元 tool",
          names == ["execute_tool", "lookup_tool", "lookup_tool_param_schema"], str(names))

    out = await mcp_gateway_client.lookup_tool("去雾预测")
    check("McpGatewayClient.lookup_tool 元工具 call 命中",
          "prediction" in out.lower(), out.splitlines()[0] if out else "空")

    out = await mcp_gateway_client.lookup_tool_param_schema("get_api_v1_prediction_by_taskid")
    check("McpGatewayClient 参数 schema 元工具 call", "taskId" in out, out[:80] if out else "空")

    print(f"\n结果: {PASS} pass / {FAIL} fail")
    sys.exit(1 if FAIL else 0)


asyncio.run(main())
