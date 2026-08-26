"""网络搜索客户端（WebSearchClient）与搜索配额

- provider 通过环境变量 SEARCH_PROVIDER_URL / SEARCH_PROVIDER_API_KEY 配置，
  未配置（默认空串）视为不可用，供工具层降级判断（对齐后端实现 §5.1）。
- 请求/响应按通用搜索 API 设计：query 参数 + JSON 返回 {title, url, snippet} 列表，
  具体厂商适配留 provider 层扩展点（覆写 _parse_response 即可）。
- 超时 10s，未配置/不可用/超时 search 返回 None，供工具层降级为知识库检索。
- 配额：用户维度固定窗口（Redis ``ai:websearch:{user_id}:{yyyyMMddHH}``，TTL 65min），
  INCR 后判定（计数累计但超限拒绝），超限返回不可用（供工具层降级）。

注：配置项暂以模块常量 + 环境变量读取，未写入 app/config.py（该文件由
capability-mcp-dev 独占追加），Lead 验收时统一收编进 Settings。
"""

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger(__name__)

# ── 配置（待 Lead 收编进 Settings）──────────────────────
SEARCH_PROVIDER_URL = os.getenv("SEARCH_PROVIDER_URL", "")
SEARCH_PROVIDER_API_KEY = os.getenv("SEARCH_PROVIDER_API_KEY", "")
SEARCH_TIMEOUT_SECONDS = float(os.getenv("SEARCH_TIMEOUT_SECONDS", "10"))
WEBSEARCH_QUOTA_PER_HOUR = int(os.getenv("WEBSEARCH_QUOTA_PER_HOUR", "30"))
WEBSEARCH_QUOTA_TTL = int(os.getenv("WEBSEARCH_QUOTA_TTL", str(65 * 60)))

_TZ = ZoneInfo("Asia/Shanghai")
_QUOTA_KEY_PREFIX = "ai:websearch"


class WebSearchClient:
    """外部搜索服务抽象（httpx 异步）。

    未配置/不可用/超时时 ``search`` 返回 None，供工具层降级为知识库检索。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = SEARCH_TIMEOUT_SECONDS,
    ):
        self._base_url = base_url or None
        self._api_key = api_key or None
        self._timeout = timeout

    @property
    def available(self) -> bool:
        """provider 是否已配置（base_url 非空）"""
        return self._base_url is not None

    async def search(self, query: str, max_results: int = 8) -> list[dict] | None:
        """执行网络搜索，返回 [{title, url, snippet}] 列表；不可用/超时返回 None。"""
        if not self.available:
            logger.info("网络搜索未配置（SEARCH_PROVIDER_URL 为空），不可用")
            return None
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        params = {"query": query, "max_results": max_results}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self._base_url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except (TimeoutError, httpx.HTTPError, ValueError) as e:
            logger.warning("网络搜索调用失败: %s", e)
            return None
        results = self._parse_response(data)
        return results[:max_results]

    @staticmethod
    def _parse_response(data) -> list[dict]:
        """解析默认 provider 响应（通用搜索 API，JSON 返回 [{title, url, snippet}]）。

        _parse_response 是厂商适配扩展点：接入新厂商时覆写以解析其特有响应结构，
        统一返回 [{title, url, snippet}] 列表。
        """
        out: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or item.get("link") or ""),
                    "snippet": str(item.get("snippet") or item.get("abstract") or ""),
                }
            )
        return out


def _quota_key(user_id: int) -> str:
    window = datetime.now(_TZ).strftime("%Y%m%d%H")
    return f"{_QUOTA_KEY_PREFIX}:{user_id}:{window}"


async def check_search_quota(redis, user_id: int) -> bool:
    """检查搜索配额并消费一次，返回是否允许本次搜索。

    Redis 固定窗口 INCR 后判定：计数累计，但超限返回 False（供工具层降级）。
    """
    key = _quota_key(user_id)
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, WEBSEARCH_QUOTA_TTL)
    return count <= WEBSEARCH_QUOTA_PER_HOUR


def format_websearch_results(results: list[dict]) -> str:
    """将搜索结果格式化为"标题+摘要+来源链接"列表文本。"""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.get('title', '')}\n   {r.get('snippet', '')}\n   来源: {r.get('url', '')}"
        )
    return "\n\n".join(lines)


# 模块级单例（工具层引用）
web_search_client = WebSearchClient(SEARCH_PROVIDER_URL, SEARCH_PROVIDER_API_KEY)
