from typing import Optional, List, Dict, Any
import json

from sqlalchemy.exc import SQLAlchemyError
from flask import current_app

from app.extensions import mysql
from app.models import SysRole, SysRoleMenu, SysUserRole, SysMenu


class RoleService:
    """角色服务类，处理角色相关的业务逻辑"""

    # 缓存常量
    ROLE_PERMS_PREFIX = "role:perms:"
    ROLE_OPTIONS_KEY = "role:options"
    CACHE_TTL_PERMS = 1800  # 30分钟
    CACHE_TTL_OPTIONS = 3600  # 1小时

    @staticmethod
    def get_role_list(page: int = 1, page_size: int = 10, keywords: str = None) -> tuple:
        """
        获取角色分页列表

        Args:
            page (int): 页码
            page_size (int): 每页数量
            keywords (str, optional): 搜索关键字（角色名称或编码）

        Returns:
            tuple: (角色列表, 总数)
        """
        query = SysRole.query.filter(SysRole.deleted == 0)

        if keywords:
            query = query.filter(
                (SysRole.name.like(f'%{keywords}%')) |
                (SysRole.code.like(f'%{keywords}%'))
            )

        pagination = query.paginate(
            page=page,
            per_page=page_size,
            error_out=False
        )

        return pagination.items, pagination.total

    @staticmethod
    def get_role_options() -> List[Dict[str, Any]]:
        """
        获取角色下拉选项列表

        Returns:
            List[Dict[str, Any]]: 角色下拉选项列表
        """
        # 尝试从缓存获取
        redis_client = current_app.extensions.get("redis_client")
        if redis_client:
            cached_data = redis_client.get(RoleService.ROLE_OPTIONS_KEY)
            if cached_data:
                return json.loads(cached_data)

        # 从数据库查询
        roles = SysRole.query.filter(
            SysRole.deleted == 0,
            SysRole.status == 1
        ).order_by(SysRole.sort).all()

        options = [{'value': role.id, 'label': role.name} for role in roles]

        # 缓存结果
        if redis_client:
            redis_client.setex(
                RoleService.ROLE_OPTIONS_KEY,
                RoleService.CACHE_TTL_OPTIONS,
                json.dumps(options)
            )

        return options

    @staticmethod
    def get_role_by_id(role_id: int) -> Optional[SysRole]:
        """
        根据ID获取角色信息

        Args:
            role_id (int): 角色ID

        Returns:
            Optional[SysRole]: 角色对象，如果未找到返回None
        """
        return SysRole.query.filter(
            SysRole.id == role_id,
            SysRole.deleted == 0
        ).first()

    @staticmethod
    def create_role(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建角色

        Args:
            data (Dict[str, Any]): 角色数据

        Returns:
            Dict[str, Any]: 创建结果
        """
        import re

        name = data.get('name')
        code = data.get('code')

        if not name or not code:
            return {'error': '角色名称和编码不能为空'}

        # 校验角色编码格式：大写字母、数字、下划线
        if not re.match(r'^[A-Z0-9_]+$', code):
            return {'error': '角色编码格式错误，只能包含大写字母、数字和下划线'}

        # 检查角色名称或编码是否已存在
        existing_role = SysRole.query.filter(
            (SysRole.name == name) | (SysRole.code == code),
            SysRole.deleted == 0
        ).first()

        if existing_role:
            return {'error': '角色名称或编码已存在'}

        role = SysRole(
            name=name,
            code=code,
            sort=data.get('sort', 0),
            status=data.get('status', 1),
            data_scope=data.get('dataScope', 1)
        )

        mysql.session.add(role)
        mysql.session.commit()

        # 清除角色选项缓存
        RoleService._clear_role_options_cache()

        return {'data': {'id': role.id}}

    @staticmethod
    def _clear_role_options_cache():
        """
        清除角色选项缓存
        """
        redis_client = current_app.extensions.get("redis_client")
        if redis_client:
            redis_client.delete(RoleService.ROLE_OPTIONS_KEY)

    @staticmethod
    def _clear_role_perms_cache(role_code: str):
        """
        清除角色权限缓存

        Args:
            role_code (str): 角色编码
        """
        redis_client = current_app.extensions.get("redis_client")
        if redis_client:
            cache_key = f"{RoleService.ROLE_PERMS_PREFIX}{role_code}"
            redis_client.delete(cache_key)

    @staticmethod
    def update_role(role_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新角色信息

        Args:
            role_id (int): 角色ID
            data (Dict[str, Any]): 角色数据

        Returns:
            Dict[str, Any]: 更新结果
        """
        role = RoleService.get_role_by_id(role_id)
        if not role:
            return {'error': '角色不存在'}

        name = data.get('name')

        if not name:
            return {'error': '角色名称不能为空'}

        # 检查角色名称是否已存在（排除自己）
        existing_role = SysRole.query.filter(
            SysRole.id != role_id,
            SysRole.name == name,
            SysRole.deleted == 0
        ).first()

        if existing_role:
            return {'error': '角色名称已存在'}

        # 更新角色信息（不更新 code，编码创建后不可修改）
        role.name = name
        role.sort = data.get('sort', role.sort)
        role.status = data.get('status', role.status)
        role.data_scope = data.get('dataScope', role.data_scope)

        mysql.session.commit()

        # 清除角色选项缓存和角色权限缓存
        RoleService._clear_role_options_cache()
        RoleService._clear_role_perms_cache(role.code)

        return {'data': '更新成功'}

    @staticmethod
    def delete_roles(ids: str) -> Dict[str, Any]:
        """
        删除角色（支持批量删除）

        Args:
            ids (str): 角色ID，多个以英文逗号分隔

        Returns:
            Dict[str, Any]: 删除结果
        """
        role_ids = [int(id) for id in ids.split(',')]

        for role_id in role_ids:
            role = RoleService.get_role_by_id(role_id)
            if not role:
                return {'error': f'角色ID {role_id} 不存在'}

            # 超级管理员角色保护：code='ROOT' 的角色不能删除
            if role.code == 'ROOT':
                return {'error': '超级管理员角色不可删除'}

            # 检查角色是否已分配给用户
            user_count = SysUserRole.query.filter(
                SysUserRole.role_id == role_id
            ).count()

            if user_count > 0:
                return {'error': f'角色【{role.name}】已分配给用户，请先解除关联后删除'}

            # 逻辑删除
            role.deleted = 1

            # 清除角色权限缓存
            RoleService._clear_role_perms_cache(role.code)

        mysql.session.commit()
        # 清除角色选项缓存
        RoleService._clear_role_options_cache()
        return {'data': '删除成功'}

    @staticmethod
    def update_role_status(role_id: int, status: int) -> Dict[str, Any]:
        """
        更新角色状态

        Args:
            role_id (int): 角色ID
            status (int): 状态（1-启用，0-禁用）

        Returns:
            Dict[str, Any]: 更新结果
        """
        if status not in [0, 1]:
            return {'error': '状态值只能为0或1'}

        role = RoleService.get_role_by_id(role_id)
        if not role:
            return {'error': '角色不存在'}

        # 超级管理员角色保护：code='ROOT' 的角色不能修改状态
        if role.code == 'ROOT':
            return {'error': '超级管理员角色不可禁用'}

        role.status = status
        mysql.session.commit()

        return {'data': '更新成功'}

    @staticmethod
    def get_role_menu_ids(role_id: int) -> List[int]:
        """
        获取角色的菜单ID集合

        Args:
            role_id (int): 角色ID

        Returns:
            List[int]: 菜单ID列表
        """
        role_menus = SysRoleMenu.query.filter(
            SysRoleMenu.role_id == role_id
        ).all()

        return [rm.menu_id for rm in role_menus]

    @staticmethod
    def get_maximum_data_scope(roles: List[str]) -> Optional[int]:
        """
        获取最大范围的数据权限

        Args:
            roles (List[str]): 角色编码集合

        Returns:
            Optional[int]: 数据权限范围
        """
        if not roles:
            return None

        # 查询角色并按data_scope排序，返回最小值（权限最大范围）
        role_list = SysRole.query.filter(
            SysRole.code.in_(roles),
            SysRole.deleted == 0
        ).order_by(SysRole.data_scope).all()

        if not role_list:
            return None

        # 返回最小的data_scope值，即权限最大的范围
        return min(role.data_scope for role in role_list)

    @staticmethod
    def assign_menus_to_role(role_id: int, menu_ids: List[int]) -> Dict[str, Any]:
        """
        分配菜单给角色

        Args:
            role_id (int): 角色ID
            menu_ids (List[int]): 菜单ID列表

        Returns:
            Dict[str, Any]: 分配结果
        """
        role = RoleService.get_role_by_id(role_id)
        if not role:
            return {'error': '角色不存在'}

        try:
            # 先删除原有角色菜单关联
            SysRoleMenu.query.filter(SysRoleMenu.role_id == role_id).delete()

            # 添加新的角色菜单关联
            for menu_id in menu_ids:
                role_menu = SysRoleMenu(
                    role_id=role_id,
                    menu_id=menu_id
                )
                mysql.session.add(role_menu)

            mysql.session.commit()

            # 清除角色权限缓存
            RoleService._clear_role_perms_cache(role.code)

            return {'data': '分配成功'}
        except SQLAlchemyError as e:
            mysql.session.rollback()
            return {'error': f'分配菜单失败: {str(e)}'}
