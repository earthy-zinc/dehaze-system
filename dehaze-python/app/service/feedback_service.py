import logging
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.constants import SYSTEM_USER_ID
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CacheService
from app.infrastructure.mq.connection import get_publisher
from app.models.entity.sys_feedback import SysFeedback
from app.models.entity.sys_feedback_reply import SysFeedbackReply
from app.models.entity.sys_rating import SysRating
from app.models.enum.log_status import LogStatus
from app.repository.feedback_repository import (
    DAILY_FEEDBACK_LIMIT,
    FEEDBACK_STATUS_MAP,
    FEEDBACK_STATUS_REVERSE_MAP,
    feedback_reply_repository,
    feedback_repository,
    rating_repository,
)
from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.service.member_service import _check_and_adjust_level, _invalidate_member_cache

logger = logging.getLogger(__name__)

RATING_TIME_LIMIT_DAYS = 30
RATING_GROWTH_VALUE = 5
RATING_DAILY_GROWTH_LIMIT = 5
RATING_IMAGE_LIMIT = 3
FEEDBACK_IMAGE_LIMIT = 5
LOW_RATING_THRESHOLD = 2

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

STATS_CACHE_TTL = 600
DAILY_COUNT_TTL = 25 * 3600

RATING_STATS_CACHE_KEY = "rating:stats:global"
RATING_STATS_ALGORITHM_CACHE_KEY = "rating:stats:algorithm:{algorithm_id}"
FEEDBACK_STATS_CACHE_KEY = "feedback:stats"
RATING_DAILY_COUNT_KEY = "rating:daily:{user_id}:{date}"
FEEDBACK_DAILY_COUNT_KEY = "feedback:daily:{user_id}:{date}"

STATUS_PENDING = FEEDBACK_STATUS_MAP["pending"]
STATUS_PROCESSING = FEEDBACK_STATUS_MAP["processing"]
STATUS_REPLIED = FEEDBACK_STATUS_MAP["replied"]
STATUS_CLOSED = FEEDBACK_STATUS_MAP["closed"]


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _get_allowed_image_url_prefixes() -> list[str]:
    prefixes = []
    if settings.FILE_BASE_URL:
        prefixes.append(settings.FILE_BASE_URL.rstrip("/") + "/")
    protocol = "https" if settings.MINIO_SECURE else "http"
    prefixes.append(f"{protocol}://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET_NAME}/")
    return prefixes


def _validate_image_urls(urls: Optional[list[str]], max_count: int) -> None:
    if not urls:
        return
    if len(urls) > max_count:
        raise BusinessException(
            ResultCode.PARAM_ERROR,
            f"图片数量不能超过{max_count}张",
        )
    allowed_prefixes = _get_allowed_image_url_prefixes()
    for url in urls:
        if not any(url.startswith(prefix) for prefix in allowed_prefixes):
            raise BusinessException(
                ResultCode.PARAM_ERROR,
                "图片URL域名不合法",
            )
        path = urlparse(url).path.lower()
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise BusinessException(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH)


def _rating_to_my_vo(rating: SysRating, algorithm_name: str = "") -> dict:
    return {
        "id": rating.id,
        "predLogId": rating.pred_log_id,
        "algorithmName": algorithm_name or "",
        "rating": rating.rating,
        "comment": rating.comment,
        "tags": rating.tags,
        "imageUrls": rating.image_urls,
        "isAnonymous": rating.is_anonymous,
        "adminReply": rating.admin_reply,
        "replyTime": _format_dt(rating.reply_time),
        "createTime": _format_dt(rating.create_time),
    }


def _rating_to_page_vo(
    rating: SysRating,
    username: str = "",
    nickname: Optional[str] = None,
    avatar: Optional[str] = None,
    algorithm_name: str = "",
) -> dict:
    vo = _rating_to_my_vo(rating, algorithm_name)
    vo["userId"] = rating.user_id
    vo["username"] = username or ""
    vo["userAvatar"] = avatar
    vo["isHidden"] = rating.is_hidden
    return vo


def _rating_to_detail_vo(
    rating: SysRating,
    username: str = "",
    nickname: Optional[str] = None,
    avatar: Optional[str] = None,
    algorithm_name: str = "",
) -> dict:
    vo = _rating_to_page_vo(rating, username, nickname, avatar, algorithm_name)
    vo["algorithmId"] = rating.algorithm_id
    return vo


def _feedback_to_page_vo(
    feedback: SysFeedback,
    username: str = "",
    assignee_name: Optional[str] = None,
) -> dict:
    return {
        "id": feedback.id,
        "userId": feedback.user_id,
        "username": username or "",
        "feedbackType": feedback.feedback_type,
        "title": feedback.title,
        "content": feedback.content,
        "status": FEEDBACK_STATUS_REVERSE_MAP.get(feedback.status, "pending"),
        "priority": feedback.priority,
        "assigneeId": feedback.assignee_id,
        "assigneeName": assignee_name,
        "relatedModule": feedback.related_module,
        "tags": feedback.tags,
        "createTime": _format_dt(feedback.create_time),
        "updateTime": _format_dt(feedback.update_time),
    }


def _feedback_to_detail_vo(
    feedback: SysFeedback,
    username: str = "",
    assignee_name: Optional[str] = None,
    replies: Optional[list] = None,
) -> dict:
    vo = _feedback_to_page_vo(feedback, username, assignee_name)
    vo["contact"] = feedback.contact
    vo["images"] = feedback.images
    vo["assignedTime"] = _format_dt(feedback.assigned_time)
    vo["closeReason"] = feedback.close_reason
    vo["replies"] = replies or []
    return vo


def _reply_to_vo(reply: SysFeedbackReply, username: str = "") -> dict:
    return {
        "id": reply.id,
        "feedbackId": reply.feedback_id,
        "replierId": reply.replier_id,
        "replierName": username or "",
        "replierType": reply.replier_type,
        "content": reply.content,
        "replyType": reply.reply_type,
        "attachments": reply.attachments,
        "createTime": _format_dt(reply.create_time),
    }


class FeedbackService:

    @staticmethod
    async def create_rating(db: AsyncSession, redis: Redis, user_id: int, form: dict) -> dict:
        pred_log_id = form["predLogId"]
        pred_log = await rating_repository.get_prediction_log(db, pred_log_id)
        if not pred_log:
            raise BusinessException(ResultCode.PREDICTION_LOG_NOT_FOUND)

        if pred_log.status != LogStatus.COMPLETED.value:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "处理记录未完成")

        if pred_log.create_by != user_id:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "无权评价他人的处理记录")

        existing = await rating_repository.get_by_pred_log_id(db, pred_log_id)
        if existing:
            raise BusinessException(ResultCode.RATING_ALREADY_EXISTS)

        if pred_log.update_time:
            time_diff = datetime.now(pred_log.update_time.tzinfo) - pred_log.update_time
            if time_diff.days > RATING_TIME_LIMIT_DAYS:
                raise BusinessException(ResultCode.RATING_EXPIRED)

        _validate_image_urls(form.get("imageUrls"), RATING_IMAGE_LIMIT)

        rating = SysRating(
            user_id=user_id,
            pred_log_id=pred_log_id,
            algorithm_id=pred_log.algorithm_id or 0,
            rating=form["rating"],
            comment=form.get("comment"),
            tags=form.get("tags"),
            image_urls=form.get("imageUrls"),
            is_anonymous=form.get("isAnonymous", 0),
        )

        async with db.begin():
            await rating_repository.create(db, rating)
            await FeedbackService._award_rating_growth(db, redis, user_id, rating.id)

        cache = CacheService(redis)
        await cache.delete(RATING_STATS_CACHE_KEY)
        await cache.delete(
            RATING_STATS_ALGORITHM_CACHE_KEY.format(algorithm_id=rating.algorithm_id)
        )

        if rating.rating <= LOW_RATING_THRESHOLD:
            await FeedbackService._publish_low_rating_alert(rating)

        return {"id": rating.id}

    @staticmethod
    async def _award_rating_growth(db: AsyncSession, redis: Redis, user_id: int, rating_id: int) -> None:
        today = date.today().isoformat()
        count_key = RATING_DAILY_COUNT_KEY.format(user_id=user_id, date=today)
        current_count = await redis.get(count_key)
        if current_count is not None and int(current_count) >= RATING_DAILY_GROWTH_LIMIT:
            return

        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            return

        old_growth = member.growth_value
        member.growth_value = old_growth + RATING_GROWTH_VALUE
        await db.flush()

        await member_growth_log_repository.create_log(
            db,
            user_id=user_id,
            change_type="rating",
            change_value=RATING_GROWTH_VALUE,
            balance=old_growth + RATING_GROWTH_VALUE,
            related_id=str(rating_id),
            reason="评价处理结果奖励",
            operator_id=SYSTEM_USER_ID,
        )

        old_level = member.level_code
        await _check_and_adjust_level(db, member)
        if member.level_code != old_level:
            await _invalidate_member_cache(user_id=user_id, level_code=old_level)
            await _invalidate_member_cache(level_code=member.level_code)

        await redis.incr(count_key)
        await redis.expire(count_key, DAILY_COUNT_TTL)

    @staticmethod
    async def _publish_low_rating_alert(rating: SysRating) -> None:
        publisher = get_publisher()
        if publisher is None:
            logger.warning("RabbitMQ 未启用，跳过低分告警消息发布")
            return

        await publisher.publish(
            "feedback.low_rating",
            {
                "ratingId": rating.id,
                "userId": rating.user_id,
                "algorithmId": rating.algorithm_id,
                "rating": rating.rating,
                "comment": rating.comment,
                "createTime": rating.create_time.isoformat() if rating.create_time else None,
            },
        )

    @staticmethod
    async def update_rating(db: AsyncSession, redis: Redis, user_id: int, rating_id: int, form: dict) -> None:
        data = await rating_repository.get_detail_with_user(db, rating_id)
        if not data:
            raise BusinessException(ResultCode.RATING_NOT_FOUND)

        rating = data["rating"]
        if rating.user_id != user_id:
            raise BusinessException(ResultCode.RATING_NOT_FOUND)

        _validate_image_urls(form.get("imageUrls"), RATING_IMAGE_LIMIT)

        rating.rating = form["rating"]
        rating.comment = form.get("comment")
        rating.tags = form.get("tags")
        rating.image_urls = form.get("imageUrls")
        rating.is_anonymous = form.get("isAnonymous", 0)
        await db.flush()

        cache = CacheService(redis)
        await cache.delete(RATING_STATS_CACHE_KEY)
        await cache.delete(
            RATING_STATS_ALGORITHM_CACHE_KEY.format(algorithm_id=rating.algorithm_id)
        )

    @staticmethod
    async def list_my_ratings(db: AsyncSession, user_id: int, query: dict) -> dict:
        items, total = await rating_repository.get_my_page(
            db,
            user_id,
            query["pageNum"],
            query["pageSize"],
        )
        list_data = [
            _rating_to_my_vo(item["rating"], item.get("algorithm_name") or "")
            for item in items
        ]
        return {"list": list_data, "total": total}

    @staticmethod
    async def get_rating_by_prediction(
        db: AsyncSession,
        user_id: int,
        pred_log_id: int,
    ) -> Optional[dict]:
        rating = await rating_repository.get_by_pred_log_id(db, pred_log_id)
        if not rating:
            return None

        data = await rating_repository.get_detail_with_user(db, rating.id)
        if not data:
            return None

        return _rating_to_detail_vo(
            data["rating"],
            data.get("username") or "",
            data.get("nickname"),
            data.get("avatar"),
            data.get("algorithm_name") or "",
        )

    @staticmethod
    async def list_paged_ratings(db: AsyncSession, query: dict) -> dict:
        items, total = await rating_repository.get_admin_page(
            db,
            query["pageNum"],
            query["pageSize"],
            keywords=query.get("keywords"),
            algorithm_id=query.get("algorithmId"),
            rating_min=query.get("ratingMin"),
            rating_max=query.get("ratingMax"),
            has_comment=query.get("hasComment"),
            start_time=query.get("startTime"),
            end_time=query.get("endTime"),
        )
        list_data = [
            _rating_to_page_vo(
                item["rating"],
                item.get("username") or "",
                item.get("nickname"),
                item.get("avatar"),
                item.get("algorithm_name") or "",
            )
            for item in items
        ]
        return {"list": list_data, "total": total}

    @staticmethod
    async def hide_rating(db: AsyncSession, rating_id: int) -> None:
        rating = await rating_repository.get_by_id(db, rating_id)
        if not rating:
            raise BusinessException(ResultCode.RATING_NOT_FOUND)
        rating.is_hidden = 1
        await db.flush()

    @staticmethod
    async def reply_rating(db: AsyncSession, rating_id: int, content: str, admin_id: int) -> None:
        rating = await rating_repository.get_by_id(db, rating_id)
        if not rating:
            raise BusinessException(ResultCode.RATING_NOT_FOUND)
        rating.admin_reply = content
        rating.reply_time = datetime.now()
        await db.flush()

    @staticmethod
    async def get_rating_stats(
        db: AsyncSession,
        redis: Redis,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> dict:
        cache = CacheService(redis)
        cache_key = RATING_STATS_CACHE_KEY
        if not start_time and not end_time:
            cached = await cache.get_json(cache_key)
            if cached is not None:
                return cached

        stats = await rating_repository.get_stats(db, start_time, end_time)

        if not start_time and not end_time:
            await cache.set_json(cache_key, stats, STATS_CACHE_TTL)
        return stats

    @staticmethod
    async def create_feedback(db: AsyncSession, redis: Redis, user_id: int, form: dict) -> dict:
        _validate_image_urls(form.get("images"), FEEDBACK_IMAGE_LIMIT)

        today = date.today().isoformat()
        count_key = FEEDBACK_DAILY_COUNT_KEY.format(user_id=user_id, date=today)
        current_count = await redis.get(count_key)
        if current_count is not None and int(current_count) >= DAILY_FEEDBACK_LIMIT:
            raise BusinessException(ResultCode.FEEDBACK_LIMIT_EXCEEDED)

        feedback = SysFeedback(
            user_id=user_id,
            feedback_type=form["feedbackType"],
            title=form["title"],
            content=form["content"],
            contact=form.get("contact"),
            images=form.get("images"),
            related_module=form.get("relatedModule"),
            status=STATUS_PENDING,
            priority=1,
        )
        await feedback_repository.create(db, feedback)

        await redis.incr(count_key)
        await redis.expire(count_key, DAILY_COUNT_TTL)

        cache = CacheService(redis)
        await cache.delete(FEEDBACK_STATS_CACHE_KEY)

        return {"id": feedback.id}

    @staticmethod
    async def list_my_feedback(db: AsyncSession, user_id: int, query: dict) -> dict:
        items, total = await feedback_repository.get_my_page(
            db,
            user_id,
            query["pageNum"],
            query["pageSize"],
        )
        list_data = [
            _feedback_to_page_vo(
                item["feedback"],
                item.get("username") or "",
                item.get("assignee_name"),
            )
            for item in items
        ]
        return {"list": list_data, "total": total}

    @staticmethod
    async def get_feedback_detail(
        db: AsyncSession,
        feedback_id: int,
        user_id: int,
        is_admin: bool,
    ) -> dict:
        data = await feedback_repository.get_detail_with_users(db, feedback_id)
        if not data:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND)

        feedback = data["feedback"]
        if not is_admin and feedback.user_id != user_id:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND)

        reply_rows, _ = await feedback_reply_repository.list_by_feedback_id(db, feedback_id)
        replies = [
            _reply_to_vo(row["reply"], row.get("username") or "")
            for row in reply_rows
        ]

        return _feedback_to_detail_vo(
            feedback,
            data.get("username") or "",
            data.get("assignee_name"),
            replies,
        )

    @staticmethod
    async def supplement_feedback(
        db: AsyncSession,
        user_id: int,
        feedback_id: int,
        form: dict,
    ) -> None:
        feedback = await feedback_repository.get_by_id(db, feedback_id)
        if not feedback:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND)
        if feedback.user_id != user_id:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND)
        if feedback.status == STATUS_CLOSED:
            raise BusinessException(ResultCode.FEEDBACK_CLOSED)

        reply = SysFeedbackReply(
            feedback_id=feedback_id,
            replier_id=user_id,
            replier_type=1,
            content=form["content"],
            reply_type="supplement",
            attachments=form.get("attachments"),
        )
        await feedback_reply_repository.create(db, reply)

    @staticmethod
    async def list_paged_feedback(db: AsyncSession, query: dict) -> dict:
        items, total = await feedback_repository.get_admin_page(
            db,
            query["pageNum"],
            query["pageSize"],
            keywords=query.get("keywords"),
            feedback_type=query.get("feedbackType"),
            status=query.get("status"),
            related_module=query.get("relatedModule"),
            priority=query.get("priority"),
            assignee_id=query.get("assigneeId"),
            start_time=query.get("startTime"),
            end_time=query.get("endTime"),
        )
        list_data = [
            _feedback_to_page_vo(
                item["feedback"],
                item.get("username") or "",
                item.get("assignee_name"),
            )
            for item in items
        ]
        return {"list": list_data, "total": total}

    @staticmethod
    async def assign_feedback(
        db: AsyncSession,
        feedback_id: int,
        assignee_id: int,
        admin_id: int,
    ) -> None:
        feedback = await feedback_repository.get_by_id(db, feedback_id)
        if not feedback:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND)
        if feedback.status == STATUS_CLOSED:
            raise BusinessException(ResultCode.FEEDBACK_CLOSED)

        feedback.assignee_id = assignee_id
        feedback.assigned_time = datetime.now()
        if feedback.status == STATUS_PENDING:
            feedback.status = STATUS_PROCESSING
        await db.flush()

    @staticmethod
    async def reply_feedback(
        db: AsyncSession,
        feedback_id: int,
        form: dict,
        admin_id: int,
    ) -> None:
        feedback = await feedback_repository.get_by_id(db, feedback_id)
        if not feedback:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND)
        if feedback.status == STATUS_CLOSED:
            raise BusinessException(ResultCode.FEEDBACK_CLOSED)

        reply = SysFeedbackReply(
            feedback_id=feedback_id,
            replier_id=admin_id,
            replier_type=2,
            content=form["content"],
            reply_type=form.get("replyType"),
            attachments=form.get("attachments"),
        )
        await feedback_reply_repository.create(db, reply)

        feedback.status = STATUS_REPLIED
        await db.flush()

    @staticmethod
    async def close_feedback(
        db: AsyncSession,
        feedback_id: int,
        close_reason: str,
        admin_id: int,
    ) -> None:
        feedback = await feedback_repository.get_by_id(db, feedback_id)
        if not feedback:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND)
        if feedback.status == STATUS_CLOSED:
            raise BusinessException(ResultCode.FEEDBACK_CLOSED)

        feedback.status = STATUS_CLOSED
        feedback.close_reason = close_reason
        await db.flush()

    @staticmethod
    async def update_feedback_tags(db: AsyncSession, feedback_id: int, tags: list) -> None:
        feedback = await feedback_repository.get_by_id(db, feedback_id)
        if not feedback:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND)
        feedback.tags = tags if tags else []
        await db.flush()

    @staticmethod
    async def get_feedback_stats(
        db: AsyncSession,
        redis: Redis,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> dict:
        cache = CacheService(redis)
        cache_key = FEEDBACK_STATS_CACHE_KEY
        if not start_time and not end_time:
            cached = await cache.get_json(cache_key)
            if cached is not None:
                return cached

        stats = await feedback_repository.get_stats(db, start_time, end_time)

        if not start_time and not end_time:
            await cache.set_json(cache_key, stats, STATS_CACHE_TTL)
        return stats
