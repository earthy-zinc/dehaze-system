from app.extensions import mysql
from app.models import SysDict, SysDictType
from typing import List, Dict, Optional, Any
from sqlalchemy import or_, and_


class DictService:
    """字典服务类，处理字典相关的业务逻辑"""

    @staticmethod
    def get_dict_page(page: int = 1, page_size: int = 10, keywords: str = None, type_code: str = None) -> tuple:
        """
        获取字典分页列表
        
        Args:
            page (int): 页码
            page_size (int): 每页数量
            keywords (str): 搜索关键词
            type_code (str): 字典类型编码
            
        Returns:
            tuple: (字典列表, 总数)
        """
        query = SysDict.query
        
        # 添加搜索条件
        if keywords:
            query = query.filter(
                or_(
                    SysDict.name.like(f'%{keywords}%'),
                    SysDict.value.like(f'%{keywords}%')
                )
            )
            
        if type_code:
            query = query.filter(SysDict.type_code == type_code)
            
        # 只选择需要的字段
        query = query.with_entities(
            SysDict.id,
            SysDict.type_code,
            SysDict.name,
            SysDict.value,
            SysDict.status,
            SysDict.sort
        )
        
        pagination = query.paginate(
            page=page,
            per_page=page_size,
            error_out=False
        )
        
        return pagination.items, pagination.total

    @staticmethod
    def get_dict_form(dict_id: int) -> Optional[Dict[str, Any]]:
        """
        获取字典表单数据
        
        Args:
            dict_id (int): 字典ID
            
        Returns:
            Optional[Dict[str, Any]]: 字典表单数据
        """
        dict_item = SysDict.query.with_entities(
            SysDict.id,
            SysDict.type_code,
            SysDict.name,
            SysDict.value,
            SysDict.status,
            SysDict.sort,
            SysDict.remark
        ).filter(SysDict.id == dict_id).first()
        
        if not dict_item:
            return None
            
        return {
            'id': dict_item.id,
            'typeCode': dict_item.type_code,
            'name': dict_item.name,
            'value': dict_item.value,
            'status': dict_item.status,
            'sort': dict_item.sort,
            'remark': dict_item.remark
        }

    @staticmethod
    def create_dict(data: Dict[str, Any]) -> bool:
        """
        创建字典项
        
        Args:
            data (Dict[str, Any]): 字典数据
            
        Returns:
            bool: 是否创建成功
        """
        dict_item = SysDict(
            type_code=data.get('typeCode'),
            name=data.get('name'),
            value=data.get('value'),
            status=data.get('status', 1),
            sort=data.get('sort', 0),
            remark=data.get('remark', '')
        )
        
        mysql.session.add(dict_item)
        mysql.session.commit()
        return True

    @staticmethod
    def update_dict(dict_id: int, data: Dict[str, Any]) -> bool:
        """
        更新字典项
        
        Args:
            dict_id (int): 字典ID
            data (Dict[str, Any]): 字典数据
            
        Returns:
            bool: 是否更新成功
        """
        dict_item = SysDict.query.get(dict_id)
        if not dict_item:
            return False
            
        dict_item.type_code = data.get('typeCode', dict_item.type_code)
        dict_item.name = data.get('name', dict_item.name)
        dict_item.value = data.get('value', dict_item.value)
        dict_item.status = data.get('status', dict_item.status)
        dict_item.sort = data.get('sort', dict_item.sort)
        dict_item.remark = data.get('remark', dict_item.remark)
        
        mysql.session.commit()
        return True

    @staticmethod
    def delete_dict(dict_ids: List[int]) -> bool:
        """
        删除字典项
        
        Args:
            dict_ids (List[int]): 字典ID列表
            
        Returns:
            bool: 是否删除成功
        """
        SysDict.query.filter(SysDict.id.in_(dict_ids)).delete(synchronize_session=False)
        mysql.session.commit()
        return True

    @staticmethod
    def list_dict_options(type_code: str) -> List[Dict[str, Any]]:
        """
        获取字典下拉列表
        
        Args:
            type_code (str): 字典类型编码
            
        Returns:
            List[Dict[str, Any]]: 字典下拉列表
        """
        dict_items = SysDict.query.with_entities(
            SysDict.value,
            SysDict.name
        ).filter(SysDict.type_code == type_code).all()
        
        return [{'value': item.value, 'label': item.name} for item in dict_items]


class DictTypeService:
    """字典类型服务类，处理字典类型相关的业务逻辑"""

    @staticmethod
    def get_dict_type_page(page: int = 1, page_size: int = 10, keywords: str = None) -> tuple:
        """
        获取字典类型分页列表
        
        Args:
            page (int): 页码
            page_size (int): 每页数量
            keywords (str): 搜索关键词
            
        Returns:
            tuple: (字典类型列表, 总数)
        """
        query = SysDictType.query
        
        # 添加搜索条件
        if keywords:
            query = query.filter(
                or_(
                    SysDictType.name.like(f'%{keywords}%'),
                    SysDictType.code.like(f'%{keywords}%')
                )
            )
            
        # 只选择需要的字段
        query = query.with_entities(
            SysDictType.id,
            SysDictType.name,
            SysDictType.code,
            SysDictType.status,
            SysDictType.remark
        )
        
        pagination = query.paginate(
            page=page,
            per_page=page_size,
            error_out=False
        )
        
        return pagination.items, pagination.total

    @staticmethod
    def get_dict_type_form(type_id: int) -> Optional[Dict[str, Any]]:
        """
        获取字典类型表单数据
        
        Args:
            type_id (int): 字典类型ID
            
        Returns:
            Optional[Dict[str, Any]]: 字典类型表单数据
        """
        dict_type = SysDictType.query.with_entities(
            SysDictType.id,
            SysDictType.name,
            SysDictType.code,
            SysDictType.status,
            SysDictType.remark
        ).filter(SysDictType.id == type_id).first()
        
        if not dict_type:
            return None
            
        return {
            'id': dict_type.id,
            'name': dict_type.name,
            'code': dict_type.code,
            'status': dict_type.status,
            'remark': dict_type.remark
        }

    @staticmethod
    def create_dict_type(data: Dict[str, Any]) -> bool:
        """
        创建字典类型
        
        Args:
            data (Dict[str, Any]): 字典类型数据
            
        Returns:
            bool: 是否创建成功
        """
        dict_type = SysDictType(
            name=data.get('name'),
            code=data.get('code'),
            status=data.get('status', 1),
            remark=data.get('remark', '')
        )
        
        mysql.session.add(dict_type)
        mysql.session.commit()
        return True

    @staticmethod
    def update_dict_type(type_id: int, data: Dict[str, Any]) -> bool:
        """
        更新字典类型
        
        Args:
            type_id (int): 字典类型ID
            data (Dict[str, Any]): 字典类型数据
            
        Returns:
            bool: 是否更新成功
        """
        dict_type = SysDictType.query.get(type_id)
        if not dict_type:
            return False
            
        dict_type.name = data.get('name', dict_type.name)
        dict_type.code = data.get('code', dict_type.code)
        dict_type.status = data.get('status', dict_type.status)
        dict_type.remark = data.get('remark', dict_type.remark)
        
        mysql.session.commit()
        return True

    @staticmethod
    def delete_dict_types(type_ids: List[int]) -> bool:
        """
        删除字典类型
        
        Args:
            type_ids (List[int]): 字典类型ID列表
            
        Returns:
            bool: 是否删除成功
        """
        SysDictType.query.filter(SysDictType.id.in_(type_ids)).delete(synchronize_session=False)
        mysql.session.commit()
        return True

    @staticmethod
    def list_dict_items_by_type_code(type_code: str) -> List[Dict[str, Any]]:
        """
        根据字典类型编码获取字典项列表
        
        Args:
            type_code (str): 字典类型编码
            
        Returns:
            List[Dict[str, Any]]: 字典项列表
        """
        dict_items = SysDict.query.with_entities(
            SysDict.value,
            SysDict.name
        ).filter(
            and_(
                SysDict.type_code == type_code,
                SysDict.status == 1
            )
        ).all()
        
        return [{'value': item.value, 'label': item.name} for item in dict_items]