import pytest

pytestmark = pytest.mark.requires_db

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.prediction import BatchPredictionItem
from app.service.prediction.prediction_service import prediction_service


async def test_batch_predict_empty_items_rejected():
    with pytest.raises(BusinessException) as ei:
        await prediction_service.batch_predict(1, [], user_id=1, skip_quota_check=True)
    assert ei.value.code == ResultCode.PARAM_ERROR


async def test_batch_predict_exceed_limit_a0500(db):
    svc = prediction_service
    items = [BatchPredictionItem(fileId=i) for i in range(1, 7)]

    from unittest.mock import AsyncMock, patch

    from app.repository.member_benefit_repository import member_benefit_repository
    from app.repository.member_repository import member_repository
    from tests.stubs.factories import make_benefit, make_member

    member = make_member()
    benefit = make_benefit(batch_limit=5)

    with patch.object(member_repository, "get_by_user_id", AsyncMock(return_value=member)), \
         patch.object(member_benefit_repository, "get_by_level_code", AsyncMock(return_value=benefit)):
        with pytest.raises(BusinessException) as ei:
            await svc.batch_predict(1, items, user_id=1, skip_quota_check=False)
        assert ei.value.code == ResultCode.BUSINESS_ERROR
        assert "不能超过5张" in ei.value.message
