"""图像模型推荐（统一模型口径）：规则匹配 / 排序不变量 / Top-N / 候选过滤。

被测入口 `recommendation_service.get_models`：候选池 = 已启用图像模型 ∩ 规则关联
model_ids，按 matchScore 降序、modelId 升序稳定排序，取 Top-3，并落 sys_recommendation。
"""

from types import SimpleNamespace

import pytest

from app.models.entity.sys_recommendation import SysRecommendation
from app.service import recommendation_service as rec_module
from app.service.recommendation_service import recommendation_service

pytestmark = pytest.mark.requires_db

MODEL_A = "aecrnet"
MODEL_B = "aodnet"
MODEL_C = "c2pnet-its"
USER_ID = 990001


def _model(model_id: str):
    return SimpleNamespace(model_id=model_id, display_name=f"模型-{model_id}")


def _rule(scene_type: str, model_ids: list[str], weight: int):
    return SimpleNamespace(scene_type=scene_type, model_ids=model_ids, weight=weight)


def _stub(monkeypatch, *, models, rules, latest=None):
    async def _list_enabled(db, model_type):
        return models

    async def _get_enabled_rules(db):
        return rules

    async def _get_latest(db, image_md5):
        return latest

    monkeypatch.setattr(rec_module.ai_model_repository, "list_enabled", _list_enabled)
    monkeypatch.setattr(
        rec_module.recommendation_rule_repository, "get_enabled_rules", _get_enabled_rules
    )
    monkeypatch.setattr(
        rec_module.recommendation_repository, "get_latest_by_image_md5", _get_latest
    )


async def test_matches_enabled_rule_and_persists_record(db, monkeypatch):
    _stub(
        monkeypatch,
        models=[_model(MODEL_A), _model(MODEL_B)],
        rules=[_rule("urban", [MODEL_A], 60)],
    )

    result = await recommendation_service.get_models(db, USER_ID, None, "svc-md5-0001")

    assert [vo.modelId for vo in result] == [MODEL_A]
    assert result[0].modelName == f"模型-{MODEL_A}"
    assert result[0].matchScore == 60
    assert result[0].recommendationId is not None

    rec = await db.get(SysRecommendation, result[0].recommendationId)
    assert rec.user_id == USER_ID
    assert rec.image_md5 == "svc-md5-0001"
    assert rec.target_type == "model"
    assert rec.top_models[0]["modelId"] == MODEL_A


async def test_tie_break_by_model_id(db, monkeypatch):
    """同分推荐按 modelId 升序，保证跨端排序一致"""
    _stub(
        monkeypatch,
        models=[_model(MODEL_A), _model(MODEL_B), _model(MODEL_C)],
        rules=[_rule("urban", [MODEL_C, MODEL_A, MODEL_B], 80)],
    )

    result = await recommendation_service.get_models(db, USER_ID, None, "svc-md5-tie")

    assert [vo.modelId for vo in result] == [MODEL_A, MODEL_B, MODEL_C]
    assert all(vo.matchScore == 80 for vo in result)


async def test_top_n_limit(db, monkeypatch):
    extra_ids = [MODEL_A, MODEL_B, MODEL_C, "d4", "fcd"]
    _stub(
        monkeypatch,
        models=[_model(m) for m in extra_ids],
        rules=[_rule("urban", extra_ids, 90)],
    )

    result = await recommendation_service.get_models(db, USER_ID, None, "svc-md5-topn")

    assert len(result) == 3


async def test_filters_candidates_not_in_enabled_models(db, monkeypatch):
    """候选 = 规则关联 model_ids ∩ 已启用图像模型，未启用/不存在的被过滤"""
    _stub(
        monkeypatch,
        models=[_model(MODEL_A)],
        rules=[_rule("urban", ["ghost-model", MODEL_A], 99)],
    )

    result = await recommendation_service.get_models(db, USER_ID, None, "svc-md5-filter")

    assert [vo.modelId for vo in result] == [MODEL_A]


async def test_scene_from_recent_analysis_record(db, monkeypatch):
    """image_md5 命中历史分析记录时按记录场景匹配规则（而非默认 urban）"""
    latest = SimpleNamespace(analysis_result={"sceneType": "landscape"})
    _stub(
        monkeypatch,
        models=[_model(MODEL_A), _model(MODEL_B)],
        rules=[_rule("urban", [MODEL_A], 90), _rule("landscape", [MODEL_B], 40)],
        latest=latest,
    )

    result = await recommendation_service.get_models(db, USER_ID, None, "svc-md5-scene")

    assert [vo.modelId for vo in result] == [MODEL_B]
