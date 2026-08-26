from datetime import datetime

import pytest
from pydantic import ValidationError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.feedback import FeedbackCreateForm
from app.service import feedback_service as m
from app.service.feedback_service import feedback_service


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
