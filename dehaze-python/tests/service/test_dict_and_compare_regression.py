import pytest
from pydantic import ValidationError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.algorithm_select import CompareRequest
from app.models.schema.dict import DictForm
from app.service.dict_service import (
    SYSTEM_PRESET_DICT_TYPE_CODES,
    dict_service,
    dict_type_service,
)


class TestCompareRequestMin:
    def test_single_algorithm_rejected(self):
        with pytest.raises(ValidationError):
            CompareRequest(algorithmIds=[1])

    def test_two_algorithms_ok(self):
        CompareRequest(algorithmIds=[1, 2])

    def test_three_algorithms_ok(self):
        CompareRequest(algorithmIds=[1, 2, 3])

    def test_four_algorithms_rejected(self):
        with pytest.raises(ValidationError):
            CompareRequest(algorithmIds=[1, 2, 3, 4])


class TestDictTypeCodeReadonly:
    async def test_update_dict_type_code_change_rejected(self):
        class FakeRepo:
            @staticmethod
            async def get_by_id(db, type_id):
                return type("T", (), {"code": "old_code"})()

            @staticmethod
            async def update_by_id(db, type_id, data):
                return True

            @staticmethod
            async def get_by_code(db, code):
                return None

        from app.service import dict_service as m

        orig = m.dict_type_repository
        m.dict_type_repository = FakeRepo()
        try:
            with pytest.raises(BusinessException) as ei:
                await dict_type_service.update_dict_type(None, None, 1, {"code": "new_code"})
            assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        finally:
            m.dict_type_repository = orig


class TestDictPresetProtection:
    def test_preset_code_set_nonempty(self):
        assert SYSTEM_PRESET_DICT_TYPE_CODES

    async def test_delete_preset_type_rejected(self):
        class FakeRepo:
            @staticmethod
            async def count_by_ids(db, type_ids):
                return 1

            @staticmethod
            async def get_by_ids(db, type_ids):
                return [type("T", (), {"code": "gender"})()]

            @staticmethod
            async def delete_by_ids(db, type_ids):
                return 1

        from app.service import dict_service as m

        orig = m.dict_type_repository
        m.dict_type_repository = FakeRepo()
        try:
            with pytest.raises(BusinessException) as ei:
                await dict_type_service.delete_dict_types(None, None, [1], force=True)
            assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        finally:
            m.dict_type_repository = orig


class TestDictSortDefault:
    def test_dict_form_sort_default_one(self):
        form = DictForm(typeCode="gender", name="男", value="1")
        assert form.sort == 1
