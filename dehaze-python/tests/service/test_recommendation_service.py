"""推荐管理服务层测试（真实 MySQL 测试库，覆盖推荐匹配/反馈归属/规则校验/对抗语料）。

对应测试用例.md：T-REC 推荐匹配与排序、反馈归属校验、规则参数校验、
图像特征分析确定性（固定 seed 不变量）、报表日期校验。
"""

import pytest
from pydantic import ValidationError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_recommendation import SysRecommendation
from app.models.schema.recommendation import RecommendationRuleForm
from app.service.recommendation_service import recommendation_service
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.requires_db

# 种子算法（config/sql/data/sys_algorithm.sql，status=4 已发布）
ALG_A = 1
ALG_B = 2
ALG_C = 3


def _rule_form(**overrides) -> RecommendationRuleForm:
    data = {
        "ruleName": "测试规则_svc",
        "sceneType": "urban",
        "algorithmIds": [ALG_A],
        "weight": 60,
        "enabled": True,
    }
    data.update(overrides)
    return RecommendationRuleForm(**data)


def _other_user():
    return make_user_context(990001, username="rec_user_a")


async def _disable_seed_rules(db):
    """禁用 sys_rule_* 种子规则（6 场景全覆盖且关联已发布算法 13），隔离自建规则的匹配口径"""
    for r in await recommendation_service.get_rules(db):
        if r.ruleName.startswith("sys_rule_"):
            await recommendation_service.update_rule(
                db,
                r.id,
                _rule_form(
                    ruleName=r.ruleName,
                    sceneType=r.sceneType,
                    algorithmIds=r.algorithmIds,
                    weight=r.weight,
                    enabled=False,
                ),
            )


# ===== 图像特征分析（固定 seed 不变量） =====


async def test_analyze_same_url_returns_stable_features():
    """同一 URL 两次分析结果完全一致（seed 取自图像 MD5，跨进程/重启稳定）"""
    url = "http://nginx/datasets/NH-HAZE-2023/hazy/001.JPG"
    first = await recommendation_service.analyze(None, url)
    second = await recommendation_service.analyze(None, url)
    assert first.model_dump() == second.model_dump()


async def test_analyze_different_urls_return_different_md5():
    a = await recommendation_service.analyze(None, "http://nginx/a.jpg")
    b = await recommendation_service.analyze(None, "http://nginx/b.jpg")
    assert a.imageMd5 != b.imageMd5
    assert a.imageMd5 is not None
    assert len(a.imageMd5) == 32


async def test_analyze_rejects_image_id():
    with pytest.raises(BusinessException) as ei:
        await recommendation_service.analyze(1, None)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_analyze_rejects_missing_input():
    with pytest.raises(BusinessException) as ei:
        await recommendation_service.analyze(None, None)
    assert ei.value.code == ResultCode.PARAM_ERROR


async def test_analyze_rejects_non_image_url():
    with pytest.raises(BusinessException) as ei:
        await recommendation_service.analyze(None, "http://nginx/datasets/test.txt")
    assert ei.value.code == ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH


async def test_analyze_feature_domains_valid():
    vo = await recommendation_service.analyze(None, "http://nginx/x.png")
    assert vo.hazeLevel in ("light", "moderate", "heavy")
    assert 0 <= vo.hazeConfidence <= 1
    assert vo.sceneType in ("urban", "landscape", "building", "night", "backlight", "indoor")
    assert 0 <= vo.sceneConfidence <= 1
    assert vo.lighting in ("bright", "normal", "dark", "veryDark", "backlight")
    assert 0 <= vo.complexity <= 1
    assert vo.resolution in ("sd", "hd", "uhd")
    assert vo.noiseLevel in ("low", "medium", "high")


# ===== 算法推荐（规则匹配 / 排序不变量 / 落库） =====


async def test_get_algorithms_matches_enabled_rule_and_persists_record(db):
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A], weight=60)
    )
    user = _other_user()

    result = await recommendation_service.get_algorithms(db, user.id, None, "svc-md5-0001")

    assert [vo.algorithmId for vo in result] == [ALG_A]
    assert result[0].matchScore == 60
    assert result[0].recommendationId is not None

    rec = await db.get(SysRecommendation, result[0].recommendationId)
    assert rec.user_id == user.id
    assert rec.image_md5 == "svc-md5-0001"
    assert rec.feedback == 0
    assert rec.top_algorithms[0]["algorithmId"] == ALG_A


async def test_get_algorithms_tie_break_by_algorithm_id(db):
    """同分推荐按 algorithmId 升序，保证跨端排序一致"""
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_C, ALG_A, ALG_B], weight=80)
    )

    result = await recommendation_service.get_algorithms(db, 990001, None, "svc-md5-tie")

    assert [vo.algorithmId for vo in result] == [ALG_A, ALG_B, ALG_C]
    assert all(vo.matchScore == 80 for vo in result)


async def test_get_algorithms_top_n_limit(db):
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db,
        0,
        _rule_form(sceneType="urban", algorithmIds=[ALG_A, ALG_B, ALG_C, 5, 6], weight=90),
    )

    result = await recommendation_service.get_algorithms(db, 990001, None, "svc-md5-topn")

    assert len(result) == 3


async def test_get_algorithms_filters_unpublished_candidates(db):
    """候选 = 规则关联算法 ∩ 已发布算法，不存在的算法被过滤"""
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[999999], weight=99)
    )

    result = await recommendation_service.get_algorithms(db, 990001, None, "svc-md5-norule")
    assert result == []


async def test_get_algorithms_ignores_disabled_rule(db):
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A], weight=95, enabled=False)
    )
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_B], weight=50, enabled=True)
    )

    result = await recommendation_service.get_algorithms(db, 990001, None, "svc-md5-disabled")

    assert [vo.algorithmId for vo in result] == [ALG_B]


# ===== 推荐反馈（归属校验） =====


async def _create_recommendation_for(db, user_id: int, md5: str) -> int:
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A], weight=60)
    )
    result = await recommendation_service.get_algorithms(db, user_id, None, md5)
    assert result, "前置推荐结果为空，无法构造反馈场景"
    assert result[0].recommendationId is not None
    return result[0].recommendationId


async def test_submit_feedback_owner_succeeds(db):
    rec_id = await _create_recommendation_for(db, 990002, "svc-md5-fb1")

    vo = await recommendation_service.submit_feedback(db, 990002, rec_id, True)
    assert vo.id == rec_id

    rec = await db.get(SysRecommendation, rec_id)
    assert rec.feedback == 1


async def test_submit_feedback_rejects_other_user(db):
    """越权：他人推荐记录不可反馈，统一 404 不泄露存在性"""
    rec_id = await _create_recommendation_for(db, 990002, "svc-md5-fb2")

    with pytest.raises(BusinessException) as ei:
        await recommendation_service.submit_feedback(db, 990003, rec_id, False)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND

    rec = await db.get(SysRecommendation, rec_id)
    assert rec.feedback == 0


async def test_submit_feedback_rejects_nonexistent_record(db):
    with pytest.raises(BusinessException) as ei:
        await recommendation_service.submit_feedback(db, 990002, 99999999, True)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_submit_feedback_useless_marks_2(db):
    rec_id = await _create_recommendation_for(db, 990002, "svc-md5-fb3")
    await recommendation_service.submit_feedback(db, 990002, rec_id, False)

    rec = await db.get(SysRecommendation, rec_id)
    assert rec.feedback == 2


# ===== 规则管理（参数校验 + 对抗语料） =====


async def test_update_rule_create_and_update(db):
    created = await recommendation_service.update_rule(db, 0, _rule_form(weight=70))
    assert created.id > 0

    updated = await recommendation_service.update_rule(
        db, created.id, _rule_form(weight=20, enabled=False)
    )
    assert updated.id == created.id

    rules = await recommendation_service.get_rules(db)
    target = next(r for r in rules if r.id == created.id)
    assert target.weight == 20
    assert target.enabled is False


async def test_update_rule_rejects_invalid_scene_type(db):
    with pytest.raises(BusinessException) as ei:
        await recommendation_service.update_rule(db, 0, _rule_form(sceneType="desert"))
    assert ei.value.code == ResultCode.PARAM_ERROR


async def test_rule_form_rejects_weight_out_of_range():
    with pytest.raises(ValidationError):
        _rule_form(weight=101)
    with pytest.raises(ValidationError):
        _rule_form(weight=-1)


async def test_rule_form_rejects_empty_algorithm_ids():
    with pytest.raises(ValidationError):
        _rule_form(algorithmIds=[])


async def test_update_rule_rejects_nonexistent_rule(db):
    with pytest.raises(BusinessException) as ei:
        await recommendation_service.update_rule(db, 99999999, _rule_form())
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


@pytest.mark.parametrize(
    "dirty_name",
    [
        "规则🌫️零宽​测试",  # emoji + 零宽空格
        "规则\r\nCRLF测试",  # CRLF
        "规则ＡＢＣ全角１２３",  # 全角
        "规则\U0001f9f8重音é中文",  # emoji 补充平面 + 重音字符
        "规" * 64,  # 64 字符边界
    ],
)
async def test_update_rule_accepts_dirty_rule_name_within_length(db, dirty_name):
    created = await recommendation_service.update_rule(
        db, 0, _rule_form(ruleName=dirty_name, sceneType="landscape")
    )
    rules = await recommendation_service.get_rules(db)
    target = next(r for r in rules if r.id == created.id)
    assert target.ruleName == dirty_name


async def test_update_rule_rejects_oversized_rule_name(db):
    """超长规则名称由数据库层拒绝，不允许静默截断"""

    with pytest.raises(Exception, match="Data too long") as ei:
        await recommendation_service.update_rule(db, 0, _rule_form(ruleName="规" * 65))
    assert not isinstance(ei.value, BusinessException)


# ===== 推荐效果报表（日期校验） =====


async def test_get_report_rejects_invalid_date_format(db):
    for bad in ("2026/09/01", "not-a-date", "2026-13-40"):
        with pytest.raises(BusinessException) as ei:
            await recommendation_service.get_report(db, bad, None)
        assert ei.value.code == ResultCode.PARAM_ERROR

        with pytest.raises(BusinessException):
            await recommendation_service.get_report(db, None, bad)


async def test_get_report_counts_recommendations_and_feedback(db):
    """仅有用反馈、无推荐来源预测：满意度为 1，采纳率（推荐来源预测口径）为 0"""
    rec_id = await _create_recommendation_for(db, 990004, "svc-md5-report")
    await recommendation_service.submit_feedback(db, 990004, rec_id, True)

    report = await recommendation_service.get_report(db, None, None)

    assert report.totalRecommendations >= 1
    assert report.satisfactionRate == 1.0
    assert report.adoptionRate == 0.0
    assert 0 <= report.coverageRate <= 1
    assert report.coldStartSuccessRate == 0.0


async def test_get_report_no_feedback_zero_rates(db):
    report = await recommendation_service.get_report(db, None, None)
    assert 0 <= report.adoptionRate <= 1
    assert isinstance(report.trend, list)


# ===== 规则防重（A0501，同场景+同算法集合） =====


async def test_update_rule_rejects_duplicate_scene_and_algorithms(db):
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A, ALG_B], weight=60)
    )

    with pytest.raises(BusinessException) as ei:
        await recommendation_service.update_rule(
            db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A, ALG_B], weight=90)
        )
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_update_rule_duplicate_is_order_insensitive(db):
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A, ALG_B], weight=60)
    )

    with pytest.raises(BusinessException) as ei:
        await recommendation_service.update_rule(
            db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_B, ALG_A], weight=70)
        )
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_update_rule_allows_same_algorithms_in_different_scene(db):
    await _disable_seed_rules(db)
    await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A], weight=60)
    )

    vo = await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="landscape", algorithmIds=[ALG_A], weight=50)
    )
    assert vo.id > 0


async def test_update_rule_self_update_not_treated_as_duplicate(db):
    await _disable_seed_rules(db)
    created = await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A], weight=60)
    )

    vo = await recommendation_service.update_rule(
        db, created.id, _rule_form(sceneType="urban", algorithmIds=[ALG_A], weight=80)
    )
    assert vo.id == created.id


async def test_update_rule_rejects_update_colliding_with_other_rule(db):
    await _disable_seed_rules(db)
    first = await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_A], weight=60)
    )
    second = await recommendation_service.update_rule(
        db, 0, _rule_form(sceneType="urban", algorithmIds=[ALG_B], weight=50)
    )

    with pytest.raises(BusinessException) as ei:
        await recommendation_service.update_rule(
            db, second.id, _rule_form(sceneType="urban", algorithmIds=[ALG_A], weight=70)
        )
    assert ei.value.code == ResultCode.DATA_EXISTS
    assert first.id > 0
