from datetime import date, datetime

import pytest
from pydantic import ValidationError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.feedback import FeedbackCreateForm
from app.service import feedback_service as m
from app.service.feedback_service import feedback_service

pytestmark = pytest.mark.requires_db

USER_ID = 1006001


def _fake_rating(**overrides):
    base = {
        "id": 1,
        "pred_log_id": 100,
        "algorithm_id": 1,
        "user_id": 10,
        "rating": 4,
        "comment": "不错",
        "tags": None,
        "image_urls": None,
        "is_anonymous": 0,
        "is_hidden": 0,
        "admin_reply": None,
        "reply_time": None,
        "create_time": datetime(2025, 1, 1, 12, 0, 0),
    }
    base.update(overrides)
    return type("SysRating", (), base)()


class _FakeRatingRepo:

    def __init__(self, *, log_create_by=10, visible=None, detail=None, admin_page=None):
        self._log = type("Log", (), {"create_by": log_create_by})()
        self._visible = visible
        self._detail = detail
        self._admin_page = admin_page

    async def get_prediction_log(self, db, pred_log_id):
        return self._log

    async def get_visible_by_pred_log_id(self, db, pred_log_id):
        return self._visible

    async def get_detail_with_user(self, db, rating_id):
        return self._detail

    async def get_admin_page(self, db, page, page_size, **kwargs):
        return self._admin_page


class TestGetRatingByPrediction:

    @pytest.mark.parametrize(
        "is_anonymous,exp_user_id,exp_username,exp_avatar",
        [
            (1, None, None, None),
            (0, 10, "user10", "http://x/a.png"),
        ],
        ids=["anonymous_masked", "non_anonymous_keeps_user"],
    )
    async def test_visible_rating_vo(
        self, monkeypatch, is_anonymous, exp_user_id, exp_username, exp_avatar
    ):
        rating = _fake_rating(is_anonymous=is_anonymous, is_hidden=0)
        detail = {
            "rating": rating,
            "username": "user10",
            "nickname": "昵称",
            "avatar": "http://x/a.png",
            "algorithm_name": "去雾算法",
        }
        monkeypatch.setattr(feedback_service, "rating_repository", _FakeRatingRepo(visible=rating, detail=detail))

        result = await feedback_service.get_rating_by_prediction(None, 10, 100)

        assert result["userId"] == exp_user_id
        assert result["username"] == exp_username
        assert result["userAvatar"] == exp_avatar
        assert result["isAnonymous"] == is_anonymous
        assert result["algorithmId"] == 1
        assert result["rating"] == 4

    async def test_other_user_rating_rejected(self, monkeypatch):
        monkeypatch.setattr(feedback_service, "rating_repository", _FakeRatingRepo(log_create_by=99))

        with pytest.raises(BusinessException) as exc:
            await feedback_service.get_rating_by_prediction(None, 10, 100)

        assert exc.value.code == ResultCode.OPERATION_NOT_ALLOW


class TestListPagedRatingsAnonymize:

    async def test_anonymous_rating_masked(self, monkeypatch):
        anon = _fake_rating(id=1, user_id=1, is_anonymous=1)
        normal = _fake_rating(id=2, user_id=2, is_anonymous=0)
        items = [
            {
                "rating": anon,
                "username": "u1",
                "nickname": "n1",
                "avatar": "a1",
                "algorithm_name": "algo",
            },
            {
                "rating": normal,
                "username": "u2",
                "nickname": "n2",
                "avatar": "a2",
                "algorithm_name": "algo",
            },
        ]
        monkeypatch.setattr(feedback_service, "rating_repository", _FakeRatingRepo(admin_page=(items, 2)))

        data = await feedback_service.list_paged_ratings(None, {"pageNum": 1, "pageSize": 10})

        anon_vo, normal_vo = data["list"]
        assert data["total"] == 2
        assert anon_vo["userId"] is None
        assert anon_vo["username"] is None
        assert anon_vo["userAvatar"] is None
        assert normal_vo["userId"] == 2
        assert normal_vo["username"] == "u2"
        assert normal_vo["userAvatar"] == "a2"


class TestFeedbackTypeLiteral:

    def test_valid_types(self):
        for t in ["suggestion", "bug", "experience", "complaint"]:
            FeedbackCreateForm(
                feedbackType=t, title="这是一个标题", content="这是反馈内容的详细描述信息"
            )

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackCreateForm(
                feedbackType="invalid", title="这是一个标题", content="这是反馈内容的详细描述信息"
            )

    def test_missing_type_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackCreateForm(title="这是一个标题", content="这是反馈内容的详细描述信息")

    def test_title_too_short_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackCreateForm(feedbackType="bug", title="标题", content="这是反馈内容的详细描述信息")

    def test_content_too_short_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackCreateForm(feedbackType="bug", title="这是一个标题", content="太短")


class TestRatingGrowthDictDriven:
    """评价成长值（rating_growth_value / rating_growth_daily_limit）取自 sys_dict。"""

    async def _create_rating(self, db, redis, user_id):
        """构造一条可直接发放成长值的评价记录（member 已初始化）。"""
        from app.repository.member_repository import member_repository

        await member_repository.get_or_init_member(db, user_id)
        rating = type("R", (), {"id": 9001, "algorithm_id": 1, "user_id": user_id})()
        await feedback_service._award_rating_growth(db, redis, user_id, rating.id)
        return rating

    async def test_rating_growth_value_from_dict(self, db, mock_redis):
        from app.repository.dict_repository import dict_repository
        from app.repository.member_repository import member_repository

        item = await dict_repository.get_by_type_code_and_name(
            db, "member_growth_rules", "rating_growth_value"
        )
        item.value = "8"
        await db.flush()
        # 模拟生产：运营更新字典后失效 dict:value 缓存（测试绕过 DictService 直改 DB）
        from app.service.dict_service import _invalidate_dict_value_cache

        await _invalidate_dict_value_cache(mock_redis, "member_growth_rules")
        # 清空当日计数，保证测试独立
        from app.service.feedback_service import RATING_DAILY_COUNT_KEY

        await mock_redis.delete(RATING_DAILY_COUNT_KEY.format(user_id=USER_ID, date=date.today()))
        await self._create_rating(db, mock_redis, USER_ID)
        member = await member_repository.get_by_user_id(db, USER_ID)
        assert member.growth_value == 8

    async def test_rating_growth_daily_limit_from_dict(self, db, mock_redis):
        """每日评价成长值上限取自字典，超过后不再累计。"""
        from app.repository.dict_repository import dict_repository
        from app.repository.member_repository import member_repository

        limit_item = await dict_repository.get_by_type_code_and_name(
            db, "member_growth_rules", "rating_growth_daily_limit"
        )
        limit_item.value = "1"
        await db.flush()
        from app.service.dict_service import _invalidate_dict_value_cache
        from app.service.feedback_service import RATING_DAILY_COUNT_KEY

        await _invalidate_dict_value_cache(mock_redis, "member_growth_rules")
        # 清空当日计数，保证测试独立
        await mock_redis.delete(RATING_DAILY_COUNT_KEY.format(user_id=USER_ID, date=date.today()))

        await self._create_rating(db, mock_redis, USER_ID)
        member = await member_repository.get_by_user_id(db, USER_ID)
        assert member.growth_value == 5

        # 第二次评价，当日已达上限 1 次，不再累计
        await self._create_rating(db, mock_redis, USER_ID)
        member = await member_repository.get_by_user_id(db, USER_ID)
        assert member.growth_value == 5
