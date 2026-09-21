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
    RecommendationRuleForm,
    RecommendationRuleVO,
    RecommendedAlgorithmVO,
    TrendItem,
)
from app.repository.algorithm_repository import algorithm_repository
from app.repository.pred_eval_log_repository import pred_log_repository
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


def _validate_rule_form(form: RecommendationRuleForm) -> None:
    if form.sceneType not in VALID_SCENE_TYPES:
        raise BusinessException(
            ResultCode.PARAM_ERROR,
            f"场景类型不合法，仅支持：{'/'.join(VALID_SCENE_TYPES)}",
        )


class RecommendationService:
    async def analyze(self, image_id: int | None, image_url: str | None) -> ImageFeatureAnalysisVO:
        url = _resolve_and_validate_image_url(image_url, image_id)
        md5_val = hashlib.md5(url.encode("utf-8")).hexdigest()
        # 以图像 MD5 为固定种子：同一 URL 的特征跨进程、跨重启保持稳定
        seed = int(md5_val, 16)

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
        # 仅取已发布且未删除的算法作为推荐候选池
        published = await algorithm_repository.list_published(db)

        rules = await recommendation_rule_repository.get_enabled_rules(db)

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

        matched_rules = [r for r in rules if r.scene_type == scene_type] if rules else []
        candidate_ids: set[int] = set()
        rule_weight_map: dict[int, int] = {}
        for r in matched_rules:
            for aid in r.algorithm_ids:
                candidate_ids.add(aid)
                current = rule_weight_map.get(aid, 0)
                if r.weight > current:
                    rule_weight_map[aid] = r.weight

        candidates = [a for a in published if a.id in candidate_ids] if published else []

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

        # 主排序 matchScore 降序，次排序 algorithmId 升序，保证跨端排序一致
        result.sort(key=lambda x: (-x.matchScore, x.algorithmId))
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
            image_md5=image_md5,
            target_type="algorithm",
            top_algorithms=top_algorithms,
            feedback=0,
        )
        db.add(rec)
        await db.flush()
        await db.refresh(rec)

        for vo in result:
            vo.recommendationId = rec.id

        return result

    async def submit_feedback(
        self, db: AsyncSession, user_id: int, recommendation_id: int, useful: bool
    ) -> IdVO:
        # 仅允许反馈本人产生的推荐记录；他人/不存在记录统一 404，不泄露存在性
        rec = await recommendation_repository.get_by_id(db, recommendation_id)
        if not rec or rec.user_id != user_id:
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

    async def update_rule(
        self, db: AsyncSession, rule_id: int, form: RecommendationRuleForm
    ) -> IdVO:
        _validate_rule_form(form)

        if rule_id != 0:
            existing = await recommendation_rule_repository.get_by_id(db, rule_id)
            if not existing:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND)

        rules = await recommendation_rule_repository.get_all_rules(db)
        target = set(form.algorithmIds)
        for r in rules:
            if r.id == rule_id or not r.enabled:
                continue
            if r.scene_type == form.sceneType and set(r.algorithm_ids) == target:
                raise BusinessException(ResultCode.DATA_EXISTS, "同场景下已存在相同算法组合的规则")

        if rule_id == 0:
            rule = SysRecommendationRule(
                rule_name=form.ruleName,
                scene_type=form.sceneType,
                algorithm_ids=form.algorithmIds,
                weight=form.weight,
                enabled=1 if form.enabled else 0,
            )
            db.add(rule)
            await db.flush()
            await db.refresh(rule)
            return IdVO(id=rule.id)

        rule = await recommendation_rule_repository.get_by_id(db, rule_id)
        if not rule:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND)

        rule.rule_name = form.ruleName
        rule.scene_type = form.sceneType
        rule.algorithm_ids = form.algorithmIds
        rule.weight = form.weight
        rule.enabled = 1 if form.enabled else 0
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
            except ValueError as e:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, "日期格式不正确，应为 yyyy-MM-dd"
                ) from e
            start = datetime(d.year, d.month, d.day, 0, 0, 0)
        if end_date:
            try:
                d = date.fromisoformat(end_date)
            except ValueError as e:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, "日期格式不正确，应为 yyyy-MM-dd"
                ) from e
            end = datetime(d.year, d.month, d.day, 23, 59, 59)

        total = await recommendation_repository.count_total(db, start, end)
        useful_count = await recommendation_repository.count_useful(db, start, end)
        feedback_total = await recommendation_repository.count_feedback_total(db, start, end)
        adopted_distinct = await recommendation_repository.count_adopted_algorithm_distinct(
            db, start, end
        )
        adopted_pred_count = await pred_log_repository.count_recommended(db, start, end)

        all_published = await algorithm_repository.list_published(db)
        published_count = len(all_published)

        # 采纳率口径：带推荐来源（recommended_by）的预测记录数 / 推荐总数，
        # 即用户从推荐入口真正发起去雾处理的比例
        adoption_rate = adopted_pred_count / total if total > 0 else 0.0
        # 满意度口径：有用反馈占比（有用即满意）
        satisfaction_rate = useful_count / feedback_total if feedback_total > 0 else 0.0

        daily_totals = await recommendation_repository.select_daily_totals(db, start, end)
        daily_adopted = await pred_log_repository.select_daily_recommended(db, start, end)
        adopted_map = {d["date"]: d["count"] for d in daily_adopted}
        trend = [
            TrendItem(
                date=d["date"],
                adoptionRate=adopted_map.get(d["date"], 0) / d["total"] if d["total"] > 0 else 0.0,
            )
            for d in daily_totals
        ]

        return RecommendationReportVO(
            totalRecommendations=total,
            adoptionRate=adoption_rate,
            satisfactionRate=satisfaction_rate,
            coverageRate=adopted_distinct / published_count if published_count > 0 else 0.0,
            coldStartSuccessRate=0.0,
            trend=trend,
        )


recommendation_service = RecommendationService()
