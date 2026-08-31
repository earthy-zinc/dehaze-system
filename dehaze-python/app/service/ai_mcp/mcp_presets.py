"""MCP 市场预设目录（内置静态配置）

installed 由同名 Server 推导，不落表。endpoint 为各外部服务的官方接入端点，
安装后可再调整。本地自建 Server（MySQL/网络搜索）为内网 http 端点，
在 SSRF 校验（mcp_connection.is_mcp_endpoint_allowed）中按白名单放行。
"""

MARKET_PRESETS: list[dict[str, object]] = [
    {
        "preset_id": "github",
        "name": "GitHub",
        "description": "GitHub 仓库/Issue/PR/代码管理",
        "capability_tags": ["github", "repo", "issue", "code"],
        "protocol_type": "streamable-http",
        "endpoint": "https://api.githubcopilot.com/mcp/",
        "auth_type": "oauth2",
    },
    {
        "preset_id": "mysql",
        "name": "MySQL",
        "description": "MySQL 数据库查询与运维",
        "capability_tags": ["database", "mysql", "sql"],
        "protocol_type": "streamable-http",
        "endpoint": "http://127.0.0.1:8083/mcp",
        "auth_type": "api_key",
    },
    {
        "preset_id": "search",
        "name": "网络搜索",
        "description": "联网搜索与网页摘要获取",
        "capability_tags": ["search", "web", "browser"],
        "protocol_type": "streamable-http",
        "endpoint": "http://127.0.0.1:8084/mcp",
        "auth_type": "api_key",
    },
]

# 市场预设内置端点的 SSRF 放行白名单（本平台自建 Server，位于受信内网）。
# GitHub 预设为公网 https，经 is_safe_url 自然放行，无需入白名单。
PRESET_ENDPOINTS: frozenset[str] = frozenset(
    p["endpoint"] for p in MARKET_PRESETS if isinstance(p["endpoint"], str)
)
