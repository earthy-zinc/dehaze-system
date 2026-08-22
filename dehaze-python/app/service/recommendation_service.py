"""
推荐管理服务
"""

import hashlib
from datetime import date, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_recommendation import SysRecommendation
from app.models.entity.sys_recommendation_rule import SysRecommendationRule
from app.models.schema.recommendation import (
    ColorDistribution,
    IdVO,
    ImageFeatureAnalysisVO,
    RecommendationReportVO,
    RecommendationRuleVO,
    RecommendedAlgorithmVO,
    TrendItem,
)
from app.repository.algorithm_repository import AlgorithmStatus, algorithm_repository
from app.repository.recommendation_repository import recommendation_repository
from app.repository.recommendation_rule_repository import recommendation_rule_repository

VALID_HAZE_LEVELS = ["light", "moderate", "heavy"]
VALID_SCENE_TYPES = ["urban", "landscape", "building", "night", "backlight", "indoor"]
VALID_LIGHTINGS = ["bright", "normal", "dark", "veryDark", "backlight"]
VALID_RESOLUTIONS = ["sd", "hd", "uhd"]
VALID_NOISE_LEVELS = ["low", "medium", "high"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")
TOP_N = 3

SCENE_REASON_TEMPLATES = {
    "urban": "处理速度快，对城市雾霾效果出色",
    "landscape": "在自然场景下表现稳定，色彩还原度高",
    "building": "深度模型，对建筑场景处理能力强",
    "night": "低光照增强组合，避免过度暗化",
    "backlight": "HDR预处理提升暗部细节",
    "indoor": "室内场景适配，细节保留好",
}


def _resolve_and_validate_image_url(image_url: str | None, image_id: int | None) -> str:
    if image_id is not None and image_id > 0:
        raise BusinessException(
            ResultCode.RESOURCE_NOT_FOUND, "imageId方式暂不支持，请使用imageUrl"
        )
    if not image_url:
        raise BusinessException(ResultCode.PARAM_ERROR, "imageId和imageUrl至少提供一个")

    lower = image_url.lower()
    q_idx = lower.find("?")
    if q_idx > 0:
        lower = lower[:q_idx]
    if not lower.endswith(IMAGE_EXTENSIONS):
        raise BusinessException(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH)
    return image_url


class RecommendationService:
    async def analyze(self, image_id: int | None, image_url: str | None) -> ImageFeatureAnalysisVO:
        url = _resolve_and_validate_image_url(image_url, image_id)
        md5_val = hashlib.md5(url.encode("utf-8")).hexdigest()
        seed = abs(hash(md5_val))

        return ImageFeatureAnalysisVO(
            imageMd5=md5_val,
            hazeLevel=VALID_HAZE_LEVELS[seed % len(VALID_HAZE_LEVELS)],
            hazeConfidence=round(0.5 + (seed % 50) / 100.0, 2),
            sceneType=VALID_SCENE_TYPES[seed % len(VALID_SCENE_TYPES)],
            sceneConfidence=round(0.5 + ((seed // 7) % 50) / 100.0, 2),
            lighting=VALID_LIGHTINGS[seed % len(VALID_LIGHTINGS)],
            complexity=round(0.3 + ((seed // 11) % 70) / 100.0, 2),
            colorDistribution=ColorDistribution(
                temperature=round(4000.0 + (seed % 6000), 2),
                saturation=round(0.3 + ((seed // 13) % 70) / 100.0, 2),
            ),
            resolution=VALID_RESOLUTIONS[seed % len(VALID_RESOLUTIONS)],
            noiseLevel=VALID_NOISE_LEVELS[seed % len(VALID_NOISE_LEVELS)],
        )

    async def get_algorithms(
        self,
        db: AsyncSession,
        user_id: int,
        analysis_id: int | None,
        image_md5: str | None,
    ) -> list[RecommendedAlgorithmVO]:
        # 全量取出（含软删）后过滤已发布
        all_algorithms = await algorithm_repository.get_all(db, with_deleted=True)
        published = [a for a in all_algorithms if a.status == AlgorithmStatus.PUBLISHED]

        # 获取启用规则
        rules = await recommendation_rule_repository.get_enabled_rules(db)

        # 确定场景类型
        scene_type = "urban"
        if analysis_id and analysis_id > 0:
            rec = await recommendation_repository.get_by_id(db, analysis_id)
            if rec and rec.analysis_result:
                st = rec.analysis_result.get("sceneType")
                if isinstance(st, str) and st in VALID_SCENE_TYPES:
                    scene_type = st

        if image_md5 and scene_type == "urban":
            rec = await recommendation_repository.get_latest_by_image_md5(db, image_md5)
            if rec and rec.analysis_result:
                st = rec.analysis_result.get("sceneType")
                if isinstance(st, str) and st in VALID_SCENE_TYPES:
                    scene_type = st

        # 规则匹配
        matched_rules = [r for r in rules if r.scene_type == scene_type] if rules else []
        candidate_ids: set[int] = set()
        rule_weight_map: dict[int, int] = {}
        for r in matched_rules:
            for aid in r.algorithm_ids:
                candidate_ids.add(aid)
                current = rule_weight_map.get(aid, 0)
                if r.weight > current:
                    rule_weight_map[aid] = r.weight

        # 筛选已发布算法中的候选
        candidates = [a for a in published if a.id in candidate_ids] if published else []

        # 构建推荐结果
        reason = SCENE_REASON_TEMPLATES.get(scene_type, "综合表现优秀")
        result: list[RecommendedAlgorithmVO] = []
        for alg in candidates:
            match_score = min(100, rule_weight_map.get(alg.id, 0))
            result.append(
                RecommendedAlgorithmVO(
                    algorithmId=alg.id,
                    algorithmName=alg.name or "",
                    matchScore=match_score,
                    reason=f"{alg.name}：{reason}",
                    effectDescription=f"该算法在{scene_type}场景下表现稳定",
                )
            )

        result.sort(key=lambda x: x.matchScore, reverse=True)
        result = result[:TOP_N]

        # 无论有无结果，都写入 sys_recommendation 记录，确保 feedback 能找到记录
        top_algorithms = [
            {
                "algorithmId": vo.algorithmId,
                "algorithmName": vo.algorithmName,
                "matchScore": vo.matchScore,
            }
            for vo in result
        ]
        rec = SysRecommendation(
            user_id=user_id,
            image_md5=image_md5 or "",
            target_type="algorithm",
            top_algorithms=top_algorithms,
            feedback=0,
        )
        db.add(rec)
        await db.flush()
        await db.refresh(rec)

        # 回填 recommendationId 到 VO
        for vo in result:
            vo.recommendationId = rec.id

        return result

    async def submit_feedback(self, db: AsyncSession, recommendation_id: int, useful: bool) -> IdVO:
        rec = await recommendation_repository.get_by_id(db, recommendation_id)
        if not rec:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND)
        rec.feedback = 1 if useful else 2
        await db.flush()
        return IdVO(id=rec.id)

    async def get_rules(self, db: AsyncSession) -> list[RecommendationRuleVO]:
        rules = await recommendation_rule_repository.get_all_rules(db)
        return [
            RecommendationRuleVO(
                id=r.id,
                ruleName=r.rule_name,
                sceneType=r.scene_type,
                algorithmIds=r.algorithm_ids or [],
                weight=r.weight,
                enabled=r.enabled == 1,
            )
            for r in rules
        ]

    async def update_rule(self, db: AsyncSession, rule_id: int, form: dict) -> IdVO:
        if rule_id == 0:
            # 新增
            rule = SysRecommendationRule(
                rule_name=form["ruleName"],
                scene_type=form["sceneType"],
                algorithm_ids=form["algorithmIds"],
                weight=form["weight"],
                enabled=1 if form.get("enabled", True) else 0,
            )
            db.add(rule)
            await db.flush()
            await db.refresh(rule)
            return IdVO(id=rule.id)

        # 更新
        rule = await recommendation_rule_repository.get_by_id(db, rule_id)
        if not rule:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND)

        rule.rule_name = form["ruleName"]
        rule.scene_type = form["sceneType"]
        rule.algorithm_ids = form["algorithmIds"]
        rule.weight = form["weight"]
        rule.enabled = 1 if form.get("enabled", True) else 0
        await db.flush()
        return IdVO(id=rule.id)

    async def get_report(
        self,
        db: AsyncSession,
        start_date: str | None,
        end_date: str | None,
    ) -> RecommendationReportVO:
        start = None
        end = None
        if start_date:
            try:
                d = date.fromisoformat(start_date)
                start = datetime(d.year, d.month, d.day, 0, 0, 0)
            except ValueError:
                pass
        if end_date:
            try:
                d = date.fromisoformat(end_date)
                end = datetime(d.year, d.month, d.day, 23, 59, 59)
            except ValueError:
                pass

        total = await recommendation_repository.count_total(db, start, end)
        useful_count = await recommendation_repository.count_useful(db, start, end)
        feedback_total = await recommendation_repository.count_feedback_total(db, start, end)
        adopted_distinct = await recommendation_repository.count_adopted_algorithm_distinct(
            db, start, end
        )

        all_algorithms = await algorithm_repository.get_all(db, with_deleted=True)
        published_count = len([a for a in all_algorithms if a.status == AlgorithmStatus.PUBLISHED])

        adoption_rate = useful_count / feedback_total if feedback_total > 0 else 0.0

        daily_data = await recommendation_repository.select_daily_adoption_rate(db, start, end)
        trend = [TrendItem(date=d["date"], adoptionRate=d["adoptionRate"]) for d in daily_data]

        return RecommendationReportVO(
            totalRecommendations=total,
            adoptionRate=adoption_rate,
            satisfactionRate=adoption_rate,
            coverageRate=adopted_distinct / published_count if published_count > 0 else 0.0,
            coldStartSuccessRate=0.0,
            trend=trend,
        )


recommendation_service = RecommendationService()
