import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.prediction_service import PredictionService
from tests.stubs import NullDBSession, run_coro


def test_batch_predict_empty_items_rejected():
    with pytest.raises(BusinessException) as ei:
        run_coro(PredictionService().batch_predict(1, [], user_id=1, skip_quota_check=True))
    assert ei.value.code == ResultCode.PARAM_ERROR


def test_batch_predict_exceed_limit_a0500():
    svc = PredictionService()
    items = [{"fileId": i} for i in range(1, 7)]

    from unittest.mock import AsyncMock, patch

    from app.repository.member_benefit_repository import member_benefit_repository
    from app.repository.member_repository import member_repository
    from app.service import prediction_service as pmod
    from tests.stubs import make_benefit, make_member

    db = NullDBSession()
    member = make_member()
    benefit = make_benefit(batch_limit=5)

    with patch.object(pmod, "get_db_session", return_value=db), \
         patch.object(member_repository, "get_by_user_id", AsyncMock(return_value=member)), \
         patch.object(member_benefit_repository, "get_by_level_code", AsyncMock(return_value=benefit)):
        with pytest.raises(BusinessException) as ei:
            run_coro(svc.batch_predict(1, items, user_id=1, skip_quota_check=False))
        assert ei.value.code == ResultCode.BUSINESS_ERROR
        assert "不能超过5张" in ei.value.message
