"""BillingStatService 统计维度白名单（GET /ai-billing/stats 的 groupBy 参数）。

回归重点：非法 groupBy 必须落 A0400 业务码，不得让 repository 的
ValueError 冒成 500 把内部异常泄漏给调用方。
"""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.ai_billing import BillingStatQuery
from app.service.billing.billing_stat_service import BillingStatService


def _service():
    repo = AsyncMock()
    repo.stats_by_dimension = AsyncMock(return_value=[])
    return BillingStatService(ai_billing_repository=repo), repo


class TestStatsDimensionWhitelist:
    @pytest.mark.parametrize("group_by", ["user", "model", "billType", "day"])
    async def test_supported_dimension_passed_to_repository(self, group_by):
        svc, repo = _service()

        result = await svc.stats(AsyncSession(), BillingStatQuery(group_by=group_by))

        assert result == []
        assert repo.stats_by_dimension.await_args.args[1] == group_by

    @pytest.mark.parametrize(
        "group_by",
        [
            "bill_type",  # 命名混淆（契约是 camelCase billType）
            "week",
            "",  # 空串
            "USER",  # 大小写
            "模型",  # 中文
            "user ",  # 尾空格（不得 strip 后放行，避免与白名单字面量不等却被误用）
            "model;drop",  # 注入串
            "a" * 256,  # 超长
        ],
    )
    async def test_invalid_dimension_rejected_without_query(self, group_by):
        svc, repo = _service()

        with pytest.raises(BusinessException) as exc:
            await svc.stats(AsyncSession(), BillingStatQuery(group_by=group_by))

        assert exc.value.code == ResultCode.PARAM_ERROR
        repo.stats_by_dimension.assert_not_awaited()
