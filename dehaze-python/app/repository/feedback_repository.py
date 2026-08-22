from datetime import datetime, timedelta

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_feedback import SysFeedback
from app.models.entity.sys_feedback_reply import SysFeedbackReply
from app.models.entity.sys_log import SysPredLog
from app.models.entity.sys_rating import SysRating
from app.models.entity.sys_user import SysUser
from app.repository.base import BaseRepository, escape_like

FEEDBACK_STATUS_MAP = {"pending": 1, "processing": 2, "replied": 3, "closed": 4}
FEEDBACK_STATUS_REVERSE_MAP = {v: k for k, v in FEEDBACK_STATUS_MAP.items()}


class RatingRepository(BaseRepository[SysRating]):
    model = SysRating

    async def get_prediction_log(self, db: AsyncSession, pred_log_id: int) -> SysPredLog | None:
        stmt = select(SysPredLog).where(SysPredLog.id == pred_log_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_pred_log_id(self, db: AsyncSession, pred_log_id: int) -> SysRating | None:
        stmt = select(SysRating).where(
            SysRating.pred_log_id == pred_log_id,
            SysRating.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_visible_by_pred_log_id(
        self, db: AsyncSession, pred_log_id: int
    ) -> SysRating | None:
        """按处理记录查用户端可见评价（过滤已隐藏），供用户端查询展示用。"""
        stmt = select(SysRating).where(
            SysRating.pred_log_id == pred_log_id,
            SysRating.deleted == 0,
            SysRating.is_hidden == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_detail_with_user(
        self,
        db: AsyncSession,
        rating_id: int,
    ) -> dict | None:
        stmt = (
            select(
                SysRating,
                SysUser.username.label("username"),
                SysUser.nickname.label("nickname"),
                SysUser.avatar.label("avatar"),
                SysAlgorithm.name.label("algorithm_name"),
            )
            .outerjoin(SysUser, SysRating.user_id == SysUser.id)
            .outerjoin(SysAlgorithm, SysRating.algorithm_id == SysAlgorithm.id)
            .where(SysRating.id == rating_id, SysRating.deleted == 0)
        )
        result = await db.execute(stmt)
        row = result.first()
        if not row:
            return None
        return {
            "rating": row[0],
            "username": row[1],
            "nickname": row[2],
            "avatar": row[3],
            "algorithm_name": row[4],
        }

    async def get_my_page(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        page_size: int,
    ) -> tuple[list[dict], int]:
        stmt = (
            select(
                SysRating,
                SysAlgorithm.name.label("algorithm_name"),
            )
            .outerjoin(SysAlgorithm, SysRating.algorithm_id == SysAlgorithm.id)
            .where(SysRating.user_id == user_id, SysRating.deleted == 0, SysRating.is_hidden == 0)
        )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysRating.create_time.desc(), SysRating.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        rows = result.all()

        items = [{"rating": row[0], "algorithm_name": row[1]} for row in rows]
        return items, total

    async def get_admin_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        *,
        keywords: str | None = None,
        algorithm_id: int | None = None,
        rating_min: int | None = None,
        rating_max: int | None = None,
        has_comment: bool | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[list[dict], int]:
        stmt = (
            select(
                SysRating,
                SysUser.username.label("username"),
                SysUser.nickname.label("nickname"),
                SysUser.avatar.label("avatar"),
                SysAlgorithm.name.label("algorithm_name"),
            )
            .outerjoin(SysUser, SysRating.user_id == SysUser.id)
            .outerjoin(SysAlgorithm, SysRating.algorithm_id == SysAlgorithm.id)
            .where(SysRating.deleted == 0)
        )

        if keywords:
            escaped = escape_like(keywords)
            like_pattern = f"%{escaped}%"
            stmt = stmt.where(
                or_(
                    SysUser.username.like(like_pattern, escape="\\"),
                    SysUser.nickname.like(like_pattern, escape="\\"),
                )
            )
        if algorithm_id is not None:
            stmt = stmt.where(SysRating.algorithm_id == algorithm_id)
        if rating_min is not None:
            stmt = stmt.where(SysRating.rating >= rating_min)
        if rating_max is not None:
            stmt = stmt.where(SysRating.rating <= rating_max)
        if has_comment is True:
            stmt = stmt.where(SysRating.comment.isnot(None), SysRating.comment != "")
        elif has_comment is False:
            stmt = stmt.where(or_(SysRating.comment.is_(None), SysRating.comment == ""))
        if start_time:
            stmt = stmt.where(
                SysRating.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            stmt = stmt.where(
                SysRating.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysRating.create_time.desc(), SysRating.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        rows = result.all()

        items = [
            {
                "rating": row[0],
                "username": row[1],
                "nickname": row[2],
                "avatar": row[3],
                "algorithm_name": row[4],
            }
            for row in rows
        ]
        return items, total

    async def get_stats(
        self,
        db: AsyncSession,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict:
        base = select(SysRating).where(SysRating.deleted == 0)
        if start_time:
            base = base.where(
                SysRating.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            base = base.where(
                SysRating.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )

        total_stmt = select(func.count()).select_from(base.subquery())
        total_ratings = int((await db.execute(total_stmt)).scalar() or 0)

        avg_stmt = select(func.coalesce(func.avg(SysRating.rating), 0)).where(
            SysRating.deleted == 0
        )
        if start_time:
            avg_stmt = avg_stmt.where(
                SysRating.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            avg_stmt = avg_stmt.where(
                SysRating.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        avg_value = (await db.execute(avg_stmt)).scalar()
        average_rating = round(float(avg_value or 0), 2)

        dist_stmt = (
            select(SysRating.rating, func.count())
            .where(SysRating.deleted == 0)
            .group_by(SysRating.rating)
        )
        if start_time:
            dist_stmt = dist_stmt.where(
                SysRating.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            dist_stmt = dist_stmt.where(
                SysRating.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        dist_rows = (await db.execute(dist_stmt)).all()
        rating_distribution = {i: 0 for i in range(1, 6)}
        for r, c in dist_rows:
            rating_distribution[r] = int(c)

        tags_stmt = select(SysRating.tags).where(
            SysRating.deleted == 0,
            SysRating.tags.isnot(None),
        )
        if start_time:
            tags_stmt = tags_stmt.where(
                SysRating.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            tags_stmt = tags_stmt.where(
                SysRating.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        tag_rows = (await db.execute(tags_stmt)).all()

        positive_counts = {
            t: 0 for t in ["去雾彻底", "色彩自然", "细节清晰", "处理速度快", "整体提升明显"]
        }
        negative_counts = {
            t: 0 for t in ["残留雾气", "色彩失真", "细节丢失", "处理速度慢", "无明显改善"]
        }
        for (tags_list,) in tag_rows:
            if tags_list:
                for tag in tags_list:
                    if tag in positive_counts:
                        positive_counts[tag] += 1
                    elif tag in negative_counts:
                        negative_counts[tag] += 1

        positive_ranking = sorted(
            [{"tag": t, "count": c} for t, c in positive_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )
        negative_ranking = sorted(
            [{"tag": t, "count": c} for t, c in negative_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )

        algo_stmt = (
            select(
                SysRating.algorithm_id,
                SysAlgorithm.name.label("algorithm_name"),
                func.count().label("total"),
                func.avg(SysRating.rating).label("avg_rating"),
                func.sum(case((SysRating.rating <= 2, 1), else_=0)).label("low_count"),
            )
            .outerjoin(SysAlgorithm, SysRating.algorithm_id == SysAlgorithm.id)
            .where(SysRating.deleted == 0)
            .group_by(SysRating.algorithm_id, SysAlgorithm.name)
        )
        if start_time:
            algo_stmt = algo_stmt.where(
                SysRating.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            algo_stmt = algo_stmt.where(
                SysRating.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        algo_rows = (await db.execute(algo_stmt)).all()

        algorithm_stats = []
        for row in algo_rows:
            total = int(row.total or 0)
            low = int(row.low_count or 0)
            algorithm_stats.append(
                {
                    "algorithmId": int(row.algorithm_id) if row.algorithm_id is not None else None,
                    "algorithmName": row.algorithm_name or "",
                    "averageRating": round(float(row.avg_rating or 0), 2),
                    "totalRatings": total,
                    "lowRatingRate": round(low * 100 / total, 2) if total > 0 else 0,
                }
            )

        return {
            "totalRatings": total_ratings,
            "averageRating": average_rating,
            "ratingDistribution": rating_distribution,
            "positiveTagRanking": positive_ranking,
            "negativeTagRanking": negative_ranking,
            "algorithmStats": algorithm_stats,
        }

    async def get_avg_rating(
        self,
        db: AsyncSession,
        algorithm_id: int,
    ) -> float:
        """算法平均评分（未评分为 0.0）"""
        from sqlalchemy import func

        stmt = select(func.avg(SysRating.rating)).where(
            SysRating.algorithm_id == algorithm_id,
            SysRating.deleted == 0,
        )
        avg = (await db.execute(stmt)).scalar()
        return float(avg) if avg else 0.0

    async def count_low_ratings_by_algorithm_24h(
        self,
        db: AsyncSession,
        algorithm_id: int,
    ) -> int:
        since = datetime.now() - timedelta(hours=24)
        stmt = select(func.count()).where(
            SysRating.deleted == 0,
            SysRating.algorithm_id == algorithm_id,
            SysRating.rating <= 2,
            SysRating.create_time >= since,
        )
        return (await db.execute(stmt)).scalar() or 0

    async def get_low_rating_stats_24h(self, db: AsyncSession) -> dict:
        since = datetime.now() - timedelta(hours=24)
        base = select(SysRating).where(
            SysRating.deleted == 0,
            SysRating.create_time >= since,
        )
        total_stmt = select(func.count()).select_from(base.subquery())
        total = (await db.execute(total_stmt)).scalar() or 0

        low_stmt = select(func.count()).where(
            SysRating.deleted == 0,
            SysRating.create_time >= since,
            SysRating.rating <= 2,
        )
        low_count = (await db.execute(low_stmt)).scalar() or 0
        return {"total": total, "lowCount": low_count}


class FeedbackRepository(BaseRepository[SysFeedback]):
    model = SysFeedback

    async def get_detail_with_users(self, db: AsyncSession, feedback_id: int) -> dict | None:
        Assignee = aliased(SysUser)
        stmt = (
            select(
                SysFeedback,
                SysUser.username.label("username"),
                Assignee.username.label("assignee_name"),
            )
            .outerjoin(SysUser, SysFeedback.user_id == SysUser.id)
            .outerjoin(Assignee, SysFeedback.assignee_id == Assignee.id)
            .where(SysFeedback.id == feedback_id, SysFeedback.deleted == 0)
        )
        result = await db.execute(stmt)
        row = result.first()
        if not row:
            return None
        return {
            "feedback": row[0],
            "username": row[1],
            "assignee_name": row[2],
        }

    async def get_my_page(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        page_size: int,
    ) -> tuple[list[dict], int]:
        Assignee = aliased(SysUser)
        stmt = (
            select(
                SysFeedback,
                SysUser.username.label("username"),
                Assignee.username.label("assignee_name"),
            )
            .outerjoin(SysUser, SysFeedback.user_id == SysUser.id)
            .outerjoin(Assignee, SysFeedback.assignee_id == Assignee.id)
            .where(
                SysFeedback.user_id == user_id,
                SysFeedback.deleted == 0,
            )
        )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysFeedback.create_time.desc(), SysFeedback.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        rows = result.all()

        items = [
            {
                "feedback": row[0],
                "username": row[1],
                "assignee_name": row[2],
            }
            for row in rows
        ]
        return items, total

    async def get_admin_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        *,
        keywords: str | None = None,
        feedback_type: str | None = None,
        status: str | None = None,
        related_module: str | None = None,
        priority: int | None = None,
        assignee_id: int | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[list[dict], int]:
        Assignee = aliased(SysUser)
        stmt = (
            select(
                SysFeedback,
                SysUser.username.label("username"),
                SysUser.nickname.label("nickname"),
                Assignee.username.label("assignee_name"),
            )
            .outerjoin(SysUser, SysFeedback.user_id == SysUser.id)
            .outerjoin(Assignee, SysFeedback.assignee_id == Assignee.id)
            .where(SysFeedback.deleted == 0)
        )

        if keywords:
            escaped = escape_like(keywords)
            like_pattern = f"%{escaped}%"
            stmt = stmt.where(
                or_(
                    SysFeedback.title.like(like_pattern, escape="\\"),
                    SysFeedback.content.like(like_pattern, escape="\\"),
                )
            )
        if feedback_type:
            stmt = stmt.where(SysFeedback.feedback_type == feedback_type)
        if status and status in FEEDBACK_STATUS_MAP:
            stmt = stmt.where(SysFeedback.status == FEEDBACK_STATUS_MAP[status])
        if related_module:
            stmt = stmt.where(SysFeedback.related_module == related_module)
        if priority is not None:
            stmt = stmt.where(SysFeedback.priority == priority)
        if assignee_id is not None:
            stmt = stmt.where(SysFeedback.assignee_id == assignee_id)
        if start_time:
            stmt = stmt.where(
                SysFeedback.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            stmt = stmt.where(
                SysFeedback.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysFeedback.create_time.desc(), SysFeedback.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        rows = result.all()

        items = [
            {
                "feedback": row[0],
                "username": row[1],
                "nickname": row[2],
                "assignee_name": row[3],
            }
            for row in rows
        ]
        return items, total

    async def get_stats(
        self,
        db: AsyncSession,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict:
        base_filter = [SysFeedback.deleted == 0]
        if start_time:
            base_filter.append(
                SysFeedback.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            base_filter.append(
                SysFeedback.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )

        total_stmt = select(func.count()).where(*base_filter)
        total_feedback = int((await db.execute(total_stmt)).scalar() or 0)

        type_stmt = (
            select(SysFeedback.feedback_type, func.count())
            .where(*base_filter)
            .group_by(SysFeedback.feedback_type)
        )
        type_rows = (await db.execute(type_stmt)).all()
        type_distribution = {t: 0 for t in ["suggestion", "bug", "experience", "complaint"]}
        for t, c in type_rows:
            type_distribution[t] = int(c)

        status_stmt = (
            select(SysFeedback.status, func.count())
            .where(*base_filter)
            .group_by(SysFeedback.status)
        )
        status_rows = (await db.execute(status_stmt)).all()
        status_distribution = {s: 0 for s in ["pending", "processing", "replied", "closed"]}
        for s, c in status_rows:
            status_name = FEEDBACK_STATUS_REVERSE_MAP.get(s, "pending")
            status_distribution[status_name] = int(c)

        module_stmt = (
            select(SysFeedback.related_module, func.count())
            .where(*base_filter, SysFeedback.related_module.isnot(None))
            .group_by(SysFeedback.related_module)
        )
        module_rows = (await db.execute(module_stmt)).all()
        module_distribution = [{"module": m, "count": int(c)} for m, c in module_rows]

        reply_subq = (
            select(
                SysFeedbackReply.feedback_id,
                func.min(SysFeedbackReply.create_time).label("first_reply_time"),
            )
            .where(SysFeedbackReply.replier_type == 2)
            .group_by(SysFeedbackReply.feedback_id)
            .subquery()
        )
        response_stmt = (
            select(
                func.avg(
                    func.unix_timestamp(reply_subq.c.first_reply_time)
                    - func.unix_timestamp(SysFeedback.create_time)
                )
            )
            .join(reply_subq, SysFeedback.id == reply_subq.c.feedback_id)
            .where(*base_filter, SysFeedback.status.in_([3, 4]))
        )
        avg_response_seconds = (await db.execute(response_stmt)).scalar()
        average_response_time = (
            round(float(avg_response_seconds) / 3600, 2) if avg_response_seconds else 0
        )

        close_stmt = select(
            func.avg(
                func.unix_timestamp(SysFeedback.update_time)
                - func.unix_timestamp(SysFeedback.create_time)
            )
        ).where(*base_filter, SysFeedback.status == 4)
        avg_close_seconds = (await db.execute(close_stmt)).scalar()
        average_close_time = round(float(avg_close_seconds) / 3600, 2) if avg_close_seconds else 0

        text_stmt = select(SysFeedback.title, SysFeedback.content).where(*base_filter)
        text_rows = (await db.execute(text_stmt)).all()
        word_counts: dict[str, int] = {}
        for title, content in text_rows:
            text = f"{title or ''} {content or ''}"
            for word in text.split():
                if len(word) >= 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
        top_keywords = sorted(
            [{"keyword": k, "count": v} for k, v in word_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20]

        return {
            "totalFeedback": total_feedback,
            "typeDistribution": type_distribution,
            "moduleDistribution": module_distribution,
            "statusDistribution": status_distribution,
            "averageResponseTime": average_response_time,
            "averageCloseTime": average_close_time,
            "topKeywords": top_keywords,
        }


class FeedbackReplyRepository(BaseRepository[SysFeedbackReply]):
    model = SysFeedbackReply

    async def list_by_feedback_id(
        self, db: AsyncSession, feedback_id: int
    ) -> tuple[list[dict], int]:
        stmt = (
            select(
                SysFeedbackReply,
                SysUser.username.label("username"),
            )
            .outerjoin(SysUser, SysFeedbackReply.replier_id == SysUser.id)
            .where(SysFeedbackReply.feedback_id == feedback_id)
            .order_by(SysFeedbackReply.create_time.asc(), SysFeedbackReply.id.asc())
        )
        result = await db.execute(stmt)
        rows = result.all()
        items = [{"reply": row[0], "username": row[1]} for row in rows]
        return items, len(items)


rating_repository = RatingRepository()
feedback_repository = FeedbackRepository()
feedback_reply_repository = FeedbackReplyRepository()
