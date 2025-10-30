"""
字典服务测试
"""
import pytest

from app.models import SysDict, SysDictType
from app.service.dict_service import DictService, DictTypeService


@pytest.mark.unit
@pytest.mark.requires_db
class TestDictService:
    """字典服务测试类"""

    def test_create_dict(self, db_session):
        """测试创建字典项"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1,
            remark='测试类型备注'
        )
        db_session.add(dict_type)
        db_session.commit()

        # 创建字典项
        dict_data = {
            'typeCode': 'TEST_TYPE',
            'name': '测试字典项',
            'value': 'test_value',
            'status': 1,
            'sort': 1,
            'remark': '测试字典项备注'
        }

        result = DictService.create_dict(dict_data)
        assert result is True

        # 验证字典项创建成功
        dict_item = SysDict.query.filter_by(name='测试字典项').first()
        assert dict_item is not None
        assert dict_item.type_code == 'TEST_TYPE'
        assert dict_item.value == 'test_value'
        assert dict_item.status == 1
        assert dict_item.sort == 1
        assert dict_item.remark == '测试字典项备注'

    def test_get_dict_form(self, db_session):
        """测试获取字典表单数据"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1
        )
        db_session.add(dict_type)
        db_session.commit()

        # 创建字典项
        dict_item = SysDict(
            type_code='TEST_TYPE',
            name='测试字典项',
            value='test_value',
            status=1,
            sort=1,
            remark='测试备注'
        )
        db_session.add(dict_item)
        db_session.commit()

        # 获取表单数据
        form_data = DictService.get_dict_form(dict_item.id)
        assert form_data is not None
        assert form_data['id'] == dict_item.id
        assert form_data['typeCode'] == 'TEST_TYPE'
        assert form_data['name'] == '测试字典项'
        assert form_data['value'] == 'test_value'
        assert form_data['status'] == 1
        assert form_data['sort'] == 1
        assert form_data['remark'] == '测试备注'

    def test_update_dict(self, db_session):
        """测试更新字典项"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1
        )
        db_session.add(dict_type)
        db_session.commit()

        # 创建字典项
        dict_item = SysDict(
            type_code='TEST_TYPE',
            name='测试字典项',
            value='test_value',
            status=1,
            sort=1
        )
        db_session.add(dict_item)
        db_session.commit()

        # 更新字典项
        update_data = {
            'typeCode': 'TEST_TYPE',
            'name': '更新后的字典项',
            'value': 'updated_value',
            'status': 0,
            'sort': 2,
            'remark': '更新后的备注'
        }

        result = DictService.update_dict(dict_item.id, update_data)
        assert result is True

        # 验证更新成功
        updated_item = SysDict.query.get(dict_item.id)
        assert updated_item.name == '更新后的字典项'
        assert updated_item.value == 'updated_value'
        assert updated_item.status == 0
        assert updated_item.sort == 2
        assert updated_item.remark == '更新后的备注'

    def test_delete_dict(self, db_session):
        """测试删除字典项"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1
        )
        db_session.add(dict_type)
        db_session.commit()

        # 创建多个字典项
        dict_item1 = SysDict(
            type_code='TEST_TYPE',
            name='测试字典项1',
            value='test_value1'
        )
        dict_item2 = SysDict(
            type_code='TEST_TYPE',
            name='测试字典项2',
            value='test_value2'
        )
        db_session.add(dict_item1)
        db_session.add(dict_item2)
        db_session.commit()

        # 记录ID
        id1, id2 = dict_item1.id, dict_item2.id

        # 删除字典项
        result = DictService.delete_dict([id1, id2])
        assert result is True

        # 验证删除成功
        deleted_items = SysDict.query.filter(SysDict.id.in_([id1, id2])).all()
        assert len(deleted_items) == 0

    def test_list_dict_options(self, db_session):
        """测试获取字典下拉列表"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1
        )
        db_session.add(dict_type)
        db_session.commit()

        # 创建字典项
        dict_item1 = SysDict(
            type_code='TEST_TYPE',
            name='选项1',
            value='option1',
            status=1
        )
        dict_item2 = SysDict(
            type_code='TEST_TYPE',
            name='选项2',
            value='option2',
            status=1
        )
        dict_item3 = SysDict(
            type_code='TEST_TYPE',
            name='禁用选项',
            value='disabled_option',
            status=0  # 禁用状态
        )
        db_session.add(dict_item1)
        db_session.add(dict_item2)
        db_session.add(dict_item3)
        db_session.commit()

        # 获取下拉列表
        options = DictService.list_dict_options('TEST_TYPE')
        assert len(options) == 3  # 包括禁用的选项
        assert {'value': 'option1', 'label': '选项1'} in options
        assert {'value': 'option2', 'label': '选项2'} in options
        assert {'value': 'disabled_option', 'label': '禁用选项'} in options

    def test_get_dict_page(self, db_session):
        """测试获取字典分页列表"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1
        )
        db_session.add(dict_type)
        db_session.commit()

        # 创建多个字典项
        for i in range(15):
            dict_item = SysDict(
                type_code='TEST_TYPE',
                name=f'测试字典项{i}',
                value=f'test_value{i}',
                status=1
            )
            db_session.add(dict_item)
        db_session.commit()

        # 获取第一页数据
        items, total = DictService.get_dict_page(page=1, page_size=10)
        assert len(items) == 10
        assert total == 15

        # 获取第二页数据
        items, total = DictService.get_dict_page(page=2, page_size=10)
        assert len(items) == 5
        assert total == 15

        # 按关键词搜索
        items, total = DictService.get_dict_page(page=1, page_size=10, keywords='测试字典项1')
        assert total > 0
        assert all('测试字典项1' in item.name for item in items)

        # 按类型编码搜索
        items, total = DictService.get_dict_page(page=1, page_size=10, type_code='TEST_TYPE')
        assert total == 15
        assert all(item.type_code == 'TEST_TYPE' for item in items)


@pytest.mark.unit
@pytest.mark.requires_db
class TestDictTypeService:
    """字典类型服务测试类"""

    def test_create_dict_type(self, db_session):
        """测试创建字典类型"""
        # 创建字典类型
        dict_type_data = {
            'name': '测试类型',
            'code': 'TEST_TYPE',
            'status': 1,
            'remark': '测试类型备注'
        }

        result = DictTypeService.create_dict_type(dict_type_data)
        assert result is True

        # 验证字典类型创建成功
        dict_type = SysDictType.query.filter_by(code='TEST_TYPE').first()
        assert dict_type is not None
        assert dict_type.name == '测试类型'
        assert dict_type.status == 1
        assert dict_type.remark == '测试类型备注'

    def test_get_dict_type_form(self, db_session):
        """测试获取字典类型表单数据"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1,
            remark='测试备注'
        )
        db_session.add(dict_type)
        db_session.commit()

        # 获取表单数据
        form_data = DictTypeService.get_dict_type_form(dict_type.id)
        assert form_data is not None
        assert form_data['id'] == dict_type.id
        assert form_data['name'] == '测试类型'
        assert form_data['code'] == 'TEST_TYPE'
        assert form_data['status'] == 1
        assert form_data['remark'] == '测试备注'

    def test_update_dict_type(self, db_session):
        """测试更新字典类型"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1
        )
        db_session.add(dict_type)
        db_session.commit()

        # 更新字典类型
        update_data = {
            'name': '更新后的类型',
            'code': 'UPDATED_TYPE',
            'status': 0,
            'remark': '更新后的备注'
        }

        result = DictTypeService.update_dict_type(dict_type.id, update_data)
        assert result is True

        # 验证更新成功
        updated_type = SysDictType.query.get(dict_type.id)
        assert updated_type.name == '更新后的类型'
        assert updated_type.code == 'UPDATED_TYPE'
        assert updated_type.status == 0
        assert updated_type.remark == '更新后的备注'

    def test_delete_dict_types(self, db_session):
        """测试删除字典类型"""
        # 创建多个字典类型
        dict_type1 = SysDictType(
            name='测试类型1',
            code='TEST_TYPE1'
        )
        dict_type2 = SysDictType(
            name='测试类型2',
            code='TEST_TYPE2'
        )
        db_session.add(dict_type1)
        db_session.add(dict_type2)
        db_session.commit()

        # 记录ID
        id1, id2 = dict_type1.id, dict_type2.id

        # 删除字典类型
        result = DictTypeService.delete_dict_types([id1, id2])
        assert result is True

        # 验证删除成功
        deleted_types = SysDictType.query.filter(SysDictType.id.in_([id1, id2])).all()
        assert len(deleted_types) == 0

    def test_list_dict_items_by_type_code(self, db_session):
        """测试根据字典类型编码获取字典项列表"""
        # 创建字典类型
        dict_type = SysDictType(
            name='测试类型',
            code='TEST_TYPE',
            status=1
        )
        db_session.add(dict_type)
        db_session.commit()

        # 创建字典项
        dict_item1 = SysDict(
            type_code='TEST_TYPE',
            name='启用选项1',
            value='enabled_option1',
            status=1  # 启用状态
        )
        dict_item2 = SysDict(
            type_code='TEST_TYPE',
            name='启用选项2',
            value='enabled_option2',
            status=1  # 启用状态
        )
        dict_item3 = SysDict(
            type_code='TEST_TYPE',
            name='禁用选项',
            value='disabled_option',
            status=0  # 禁用状态
        )
        db_session.add(dict_item1)
        db_session.add(dict_item2)
        db_session.add(dict_item3)
        db_session.commit()

        # 获取字典项列表
        items = DictTypeService.list_dict_items_by_type_code('TEST_TYPE')
        # 只应该返回启用的选项
        assert len(items) == 2
        assert {'value': 'enabled_option1', 'label': '启用选项1'} in items
        assert {'value': 'enabled_option2', 'label': '启用选项2'} in items
        assert {'value': 'disabled_option', 'label': '禁用选项'} not in items

    def test_get_dict_type_page(self, db_session):
        """测试获取字典类型分页列表"""
        # 创建多个字典类型
        for i in range(15):
            dict_type = SysDictType(
                name=f'测试类型{i}',
                code=f'TEST_TYPE{i}',
                status=1
            )
            db_session.add(dict_type)
        db_session.commit()

        # 获取第一页数据
        types, total = DictTypeService.get_dict_type_page(page=1, page_size=10)
        assert len(types) == 10
        assert total == 15

        # 获取第二页数据
        types, total = DictTypeService.get_dict_type_page(page=2, page_size=10)
        assert len(types) == 5
        assert total == 15

        # 按关键词搜索
        types, total = DictTypeService.get_dict_type_page(page=1, page_size=10, keywords='测试类型1')
        assert total > 0
        assert all('测试类型1' in t.name for t in types)
