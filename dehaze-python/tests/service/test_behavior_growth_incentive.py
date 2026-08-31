"""
图像处理 / 效果评估完成后的成长值激励（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：图像处理完成写 process 流水并带任务 ID、每日上限、激励失败不阻断主流程、
评估完成写 evaluate 流水、未登录用户跳过激励。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from unittest.mock import AsyncMock

import pytest

from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.service.evaluation_service import evaluation_service
from app.service.member.growth_service import BEHAVIOR_GROWTH_RULES, member_growth_service
from app.service.prediction.prediction_service import prediction_service

pytestmark = pytest.mark.requires_db

USER_ID = 1005001
LOG_ID = 8801

PROCESS_DAILY_LIMIT = BEHAVIOR_GROWTH_RULES["process"][1]
EVALUATE_DAILY_LIMIT = BEHAVIOR_GROWTH_RULES["evaluate"][1]


async def _growth_logs(db):
    logs, _ = await member_growth_log_repository.get_page(db, USER_ID, 1, 50)
    return logs


async def test_award_process_growth_writes_log(db, mock_redis):
    """图像处理完成：+1 成长值，流水 change_type=process 且 related_id 为任务 ID"""
    await member_repository.get_or_init_member(db, USER_ID)
    await prediction_service._award_process_growth(USER_ID, LOG_ID)

    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 1
    logs = await _growth_logs(db)
    assert any(
        log.change_type == "process" and log.change_value == 1 and log.related_id == str(LOG_ID)
        for log in logs
    )


async def test_award_process_growth_daily_limit(db, mock_redis):
    """图像处理激励每日上限 10 次，第 11 次不再累计"""
    await member_repository.get_or_init_member(db, USER_ID)
    for i in range(PROCESS_DAILY_LIMIT):
        await prediction_service._award_process_growth(USER_ID, LOG_ID + i)
    await prediction_service._award_process_growth(USER_ID, LOG_ID + 100)

    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == PROCESS_DAILY_LIMIT


async def test_award_process_growth_failure_not_raised(db, mock_redis, monkeypatch):
    """激励写入异常被吞掉并告警，不向图像处理主流程抛出"""
    monkeypatch.setattr(
        member_growth_service,
        "add_behavior_growth",
        AsyncMock(side_effect=RuntimeError("成长值服务不可用")),
    )
    await prediction_service._award_process_growth(USER_ID, LOG_ID)


async def test_award_evaluate_growth_writes_log(db, mock_redis):
    """效果评估完成：+1 成长值，流水 change_type=evaluate"""
    await member_repository.get_or_init_member(db, USER_ID)
    await evaluation_service._award_evaluate_growth(USER_ID, LOG_ID)

    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 1
    logs = await _growth_logs(db)
    assert any(
        log.change_type == "evaluate" and log.related_id == str(LOG_ID) for log in logs
    )


async def test_award_evaluate_growth_daily_limit(db, mock_redis):
    """评估激励每日上限 5 次，两类行为分别计数"""
    await member_repository.get_or_init_member(db, USER_ID)
    for i in range(EVALUATE_DAILY_LIMIT):
        await evaluation_service._award_evaluate_growth(USER_ID, LOG_ID + i)
    await evaluation_service._award_evaluate_growth(USER_ID, LOG_ID + 100)
    await prediction_service._award_process_growth(USER_ID, LOG_ID + 200)

    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == EVALUATE_DAILY_LIMIT + 1


async def test_award_growth_skipped_without_user(db, mock_redis):
    """未登录场景（user_id 为空）不写成长值流水"""
    await prediction_service._award_process_growth(None, LOG_ID)
    await evaluation_service._award_evaluate_growth(None, LOG_ID)

    assert await _growth_logs(db) == []
