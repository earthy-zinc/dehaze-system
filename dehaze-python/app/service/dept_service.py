from typing import List, Dict, Optional, Any
from app.models import SysDept
from app.extensions import mysql
from sqlalchemy import and_, or_
from app.utils.utils import format_time


class DeptService:
    @staticmethod
    def get_dept_list(keywords: str = None, status: int = None) -> List[Dict[str, Any]]:
        """
        获取部门列表
        :param keywords: 关键字(部门名称)
        :param status: 状态(1->正常；0->禁用)
        :return: 部门列表
        """
        # 构建查询条件
        query = SysDept.query
        
        if keywords:
            query = query.filter(SysDept.name.like(f'%{keywords}%'))
            
        if status is not None:
            query = query.filter(SysDept.status == status)
            
        query = query.order_by(SysDept.sort)
        
        dept_list = query.all()
        
        if not dept_list:
            return []
            
        # 构建部门字典
        dept_dict = {dept.id: {
            'id': dept.id,
            'name': dept.name,
            'parent_id': dept.parent_id,
            'tree_path': dept.tree_path,
            'sort': dept.sort,
            'status': dept.status,
            'deleted': dept.deleted,
            'create_time': format_time(dept.create_time),
            'update_time': format_time(dept.update_time),
            'children': []
        } for dept in dept_list}
        
        # 构建父子关系
        root_depts = []
        for dept in dept_dict.values():
            if dept['parent_id'] == 0:  # 根节点
                root_depts.append(dept)
            else:
                parent = dept_dict.get(dept['parent_id'])
                if parent:
                    parent['children'].append(dept)
                    
        return root_depts

    @staticmethod
    def get_dept_options() -> List[Dict[str, Any]]:
        """
        获取部门下拉选项
        :return: 部门下拉选项列表
        """
        # 查询启用状态的部门
        dept_list = SysDept.query.filter(
            SysDept.status == 1
        ).order_by(SysDept.sort).all()
        
        if not dept_list:
            return []
            
        # 构建部门字典
        dept_dict = {dept.id: {
            'value': dept.id,
            'label': dept.name,
            'children': []
        } for dept in dept_list}
        
        # 构建父子关系
        root_options = []
        for dept in dept_dict.values():
            dept_obj = next((d for d in dept_list if d.id == dept['value']), None)
            if dept_obj and dept_obj.parent_id == 0:
                root_options.append(dept)
            else:
                if dept_obj:
                    parent = dept_dict.get(dept_obj.parent_id)
                    if parent:
                        parent['children'].append(dept)
                        
        return root_options

    @staticmethod
    def get_dept_form(dept_id: int) -> Optional[Dict[str, Any]]:
        """
        获取部门表单数据
        :param dept_id: 部门ID
        :return: 部门表单数据
        """
        dept = SysDept.query.get(dept_id)
        if not dept:
            return None
            
        return {
            'id': dept.id,
            'name': dept.name,
            'parent_id': dept.parent_id,
            'status': dept.status,
            'sort': dept.sort
        }

    @staticmethod
    def create_dept(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        新增部门
        :param data: 部门数据
        :return: 创建结果
        """
        # 检查部门名称是否已存在
        name = data.get('name')
        existing_count = SysDept.query.filter(SysDept.name == name).count()
        if existing_count > 0:
            return {'success': False, 'message': '部门名称已存在', 'data': None}
            
        dept = SysDept()
        dept.name = data.get('name')
        dept.parent_id = data.get('parent_id', 0)
        dept.status = data.get('status', 1)
        dept.sort = data.get('sort', 0)
        
        # 生成部门路径(tree_path)
        tree_path = DeptService._generate_dept_tree_path(dept.parent_id)
        dept.tree_path = tree_path
        
        try:
            mysql.session.add(dept)
            mysql.session.commit()
            return {'success': True, 'message': '部门创建成功', 'data': dept.id}
        except Exception as e:
            mysql.session.rollback()
            return {'success': False, 'message': f'部门创建失败: {str(e)}', 'data': None}

    @staticmethod
    def update_dept(dept_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新部门
        :param dept_id: 部门ID
        :param data: 部门数据
        :return: 更新结果
        """
        dept = SysDept.query.get(dept_id)
        if not dept:
            return {'success': False, 'message': '部门不存在', 'data': None}
            
        # 检查部门名称是否已存在（排除当前部门）
        name = data.get('name')
        existing_count = SysDept.query.filter(
            and_(
                SysDept.name == name,
                SysDept.id != dept_id
            )
        ).count()
        if existing_count > 0:
            return {'success': False, 'message': '部门名称已存在', 'data': None}
            
        dept.name = data.get('name', dept.name)
        dept.parent_id = data.get('parent_id', dept.parent_id)
        dept.status = data.get('status', dept.status)
        dept.sort = data.get('sort', dept.sort)
        
        # 生成部门路径(tree_path)
        tree_path = DeptService._generate_dept_tree_path(dept.parent_id)
        dept.tree_path = tree_path
        
        try:
            mysql.session.commit()
            return {'success': True, 'message': '部门更新成功', 'data': dept.id}
        except Exception as e:
            mysql.session.rollback()
            return {'success': False, 'message': f'部门更新失败: {str(e)}', 'data': None}

    @staticmethod
    def delete_depts(dept_ids: List[int]) -> Dict[str, Any]:
        """
        删除部门
        :param dept_ids: 部门ID列表
        :return: 删除结果
        """
        try:
            for dept_id in dept_ids:
                # 删除部门及子部门
                SysDept.query.filter(
                    or_(
                        SysDept.id == dept_id,
                        SysDept.tree_path.like(f'%,{dept_id},%'),
                        SysDept.tree_path.like(f'{dept_id},%'),
                        SysDept.tree_path.like(f'%,{dept_id}')
                    )
                ).delete(synchronize_session=False)
                
            mysql.session.commit()
            return {'success': True, 'message': '部门删除成功', 'data': None}
        except Exception as e:
            mysql.session.rollback()
            return {'success': False, 'message': f'部门删除失败: {str(e)}', 'data': None}

    @staticmethod
    def _generate_dept_tree_path(parent_id: int) -> str:
        """
        生成部门路径
        :param parent_id: 父部门ID
        :return: 部门路径
        """
        if parent_id == 0:
            return "0"
        else:
            parent_dept = SysDept.query.get(parent_id)
            if parent_dept and parent_dept.tree_path:
                return f"{parent_dept.tree_path},{parent_dept.id}"
            elif parent_dept:
                return str(parent_dept.id)
            else:
                return "0"