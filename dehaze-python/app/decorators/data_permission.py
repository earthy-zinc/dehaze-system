"""
数据权限装饰器模块

该模块提供了基于角色的数据权限控制功能，支持在 SQLAlchemy 查询中自动应用数据权限过滤。

数据权限类型：
    - ALL (0): 所有数据权限
    - DEPT_AND_SUB (1): 部门及子部门数据权限
    - DEPT (2): 本部门数据权限
    - SELF (3): 本人数据权限

使用示例：
    # 在 Service 层中使用
    from app.decorators.data_permission import apply_data_permission, DataScope
    from app.models import SysDataset

    # 获取用户信息（包含当前用户ID和部门ID）
    user_info = {
        'user_id': current_user.id,
        'dept_id': current_user.dept_id,
        'roles': ['ADMIN', 'USER']
    }

    # 应用数据权限过滤
    query = SysDataset.query
    filtered_query = apply_data_permission(
        query=query,
        user_info=user_info,
        model=SysDataset,
        dept_column='create_by',  # 用于部门及子部门权限的字段
        self_column='create_by'   # 用于本人权限的字段
    )

    results = filtered_query.all()
"""

from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Query
from app.models import SysUser, SysRole, SysUserRole, SysDept
from app.extensions import mysql


class DataScope:
    """数据权限枚举类型"""
    ALL = 0  # 所有数据权限
    DEPT_AND_SUB = 1  # 部门及子部门数据权限
    DEPT = 2  # 本部门数据权限
    SELF = 3  # 本人数据权限

    @classmethod
    def to_dict(cls) -> Dict[int, str]:
        """转换为字典，用于前端展示"""
        return {
            cls.ALL: '所有数据',
            cls.DEPT_AND_SUB: '部门及子部门数据',
            cls.DEPT: '本部门数据',
            cls.SELF: '本人数据'
        }

    @classmethod
    def get_name(cls, scope: int) -> Optional[str]:
        """根据权限范围获取名称"""
        return cls.to_dict().get(scope)


def _get_user_data_scope(user_info: Dict[str, Any]) -> Optional[int]:
    """
    获取用户的数据权限范围

    Args:
        user_info (Dict[str, Any]): 用户信息，包含 user_id、dept_id、roles 等字段

    Returns:
        Optional[int]: 数据权限范围，如果用户没有角色返回 None
    """
    # 如果用户信息中包含 data_scope 字段，直接使用
    if 'data_scope' in user_info:
        return user_info['data_scope']

    # 否则从数据库查询
    user_id = user_info.get('user_id')
    if not user_id:
        return None

    # 查询用户角色编码列表
    role_codes = _get_user_role_codes(user_id)
    if not role_codes:
        return None

    # 如果用户拥有 ROOT 角色，拥有所有数据权限
    if 'ROOT' in role_codes:
        return DataScope.ALL

    # 获取最大范围的数据权限（最小值代表最大权限范围）
    data_scope = _get_max_data_scope(role_codes)
    return data_scope


def _get_user_role_codes(user_id: int) -> List[str]:
    """
    获取用户角色编码列表

    Args:
        user_id (int): 用户ID

    Returns:
        List[str]: 角色编码列表
    """
    roles = SysRole.query.join(
        SysUserRole, SysRole.id == SysUserRole.role_id
    ).filter(
        SysUserRole.user_id == user_id,
        SysRole.deleted == 0,
        SysRole.status == 1
    ).all()

    return [role.code for role in roles]


def _get_max_data_scope(role_codes: List[str]) -> Optional[int]:
    """
    获取最大范围的数据权限

    Args:
        role_codes (List[str]): 角色编码列表

    Returns:
        Optional[int]: 数据权限范围
    """
    if not role_codes:
        return None

    role_list = SysRole.query.filter(
        SysRole.code.in_(role_codes),
        SysRole.deleted == 0
    ).order_by(SysRole.data_scope).all()

    if not role_list:
        return None

    # 返回最小的 data_scope 值，即权限最大的范围
    return min(role.data_scope for role in role_list)


def _get_dept_and_sub_dept_ids(dept_id: Optional[int]) -> List[int]:
    """
    获取部门及其所有子部门的ID列表

    Args:
        dept_id (Optional[int]): 部门ID

    Returns:
        List[int]: 部门ID列表（包含自身和所有子部门）
    """
    if not dept_id:
        return []

    # 获取当前部门信息
    dept = SysDept.query.filter_by(id=dept_id, deleted=0).first()
    if not dept:
        return [dept_id]

    # 如果部门有 tree_path，使用 tree_path LIKE 查询子部门
    if dept.tree_path:
        # 构建子部门查询：tree_path 以 dept.tree_path 开头
        child_dept_ids = SysDept.query.filter(
            SysDept.tree_path.like(f'{dept.tree_path}%'),
            SysDept.deleted == 0
        ).all()
        dept_ids = [dept.id] + [d.id for d in child_dept_ids]
    else:
        # 如果没有 tree_path，只查询当前部门
        dept_ids = [dept.id]

    return dept_ids


def _build_data_scope_filter(
    query: Query,
    model: Any,
    data_scope: int,
    user_id: int,
    dept_id: Optional[int],
    dept_column: Optional[str] = None,
    self_column: Optional[str] = None,
    table_alias: Optional[Any] = None
) -> Query:
    """
    根据数据权限范围构建查询过滤器

    Args:
        query (Query): SQLAlchemy 查询对象
        model: 数据模型类
        data_scope (int): 数据权限范围
        user_id (int): 用户ID
        dept_id (Optional[int]): 部门ID
        dept_column (Optional[str]): 用于部门权限过滤的字段名
        self_column (Optional[str]): 用于本人权限过滤的字段名
        table_alias (Optional[Any]): 表别名，用于联表查询场景

    Returns:
        Query: 应用权限过滤后的查询对象
    """
    # 如果是所有数据权限，不添加任何过滤条件
    if data_scope == DataScope.ALL:
        return query

    # 根据权限范围添加过滤条件
    if data_scope == DataScope.DEPT_AND_SUB:
        # 部门及子部门数据权限
        if dept_id and dept_column:
            # 获取部门及子部门ID列表
            dept_ids = _get_dept_and_sub_dept_ids(dept_id)

            # 获取模型属性
            if table_alias:
                column_attr = getattr(table_alias, dept_column, None)
            else:
                column_attr = getattr(model, dept_column, None)

            if column_attr:
                # 修复：使用子查询优化，避免 N+1 查询问题
                # 优化前：先查询所有用户对象再提取 ID（会触发 N+1 查询）
                # 优化后：直接使用子查询过滤，数据库层面完成优化
                # 修复 P1 级别 SQLAlchemy Subquery 警告：使用 scalar_subquery() 替代 subquery()
                user_ids_subquery = mysql.session.query(SysUser.id).filter(
                    SysUser.dept_id.in_(dept_ids),
                    SysUser.deleted == 0
                ).scalar_subquery()
                query = query.filter(column_attr.in_(user_ids_subquery))

    elif data_scope == DataScope.DEPT:
        # 本部门数据权限
        if dept_id and dept_column:
            # 获取模型属性
            if table_alias:
                column_attr = getattr(table_alias, dept_column, None)
            else:
                column_attr = getattr(model, dept_column, None)

            if column_attr:
                # 修复：使用子查询优化，避免 N+1 查询问题
                # 优化前：先查询所有用户对象再提取 ID（会触发 N+1 查询）
                # 优化后：直接使用子查询过滤，数据库层面完成优化
                # 修复 P1 级别 SQLAlchemy Subquery 警告：使用 scalar_subquery() 替代 subquery()
                user_ids_subquery = mysql.session.query(SysUser.id).filter(
                    SysUser.dept_id == dept_id,
                    SysUser.deleted == 0
                ).scalar_subquery()
                query = query.filter(column_attr.in_(user_ids_subquery))

    elif data_scope == DataScope.SELF:
        # 本人数据权限
        if self_column:
            # 获取模型属性
            if table_alias:
                column_attr = getattr(table_alias, self_column, None)
            else:
                column_attr = getattr(model, self_column, None)

            if column_attr:
                # 添加过滤条件：只查询本人创建的数据
                query = query.filter(column_attr == user_id)

    return query


def apply_data_permission(
    query: Query,
    user_info: Dict[str, Any],
    model: Any,
    dept_column: Optional[str] = None,
    self_column: Optional[str] = None,
    table_alias: Optional[Any] = None,
    skip_data_permission: bool = False
) -> Query:
    """
    在 SQLAlchemy 查询中应用数据权限过滤

    Args:
        query (Query): SQLAlchemy 查询对象
        user_info (Dict[str, Any]): 用户信息字典，必须包含以下字段：
            - user_id: 用户ID
            - dept_id: 部门ID（可选）
            - roles: 角色编码列表（可选）
            - data_scope: 数据权限范围（可选，如果不提供会自动查询）
        model: 数据模型类（用于获取模型属性）
        dept_column (Optional[str]): 用于部门权限过滤的字段名
            （例如：'create_by'，表示根据创建人的部门过滤）
        self_column (Optional[str]): 用于本人权限过滤的字段名
            （例如：'create_by'，表示只查询本人创建的数据）
        table_alias (Optional[Any]): 表别名对象，用于联表查询场景
            例如：在 query.join(SysDept, ...).filter(...) 场景中，可以使用表别名
        skip_data_permission (bool): 是否跳过数据权限过滤（默认为 False）
            设为 True 时，直接返回原查询，不应用任何数据权限过滤

    Returns:
        Query: 应用数据权限过滤后的查询对象

    Raises:
        ValueError: 如果用户信息缺少必要的字段

    示例1: 基本用法（查询本部门及子部门的数据）
        ```python
        from app.decorators.data_permission import apply_data_permission
        from app.models import SysDataset

        user_info = {
            'user_id': 1,
            'dept_id': 10,
            'roles': ['USER']
        }

        query = SysDataset.query
        filtered_query = apply_data_permission(
            query=query,
            user_info=user_info,
            model=SysDataset,
            dept_column='create_by',
            self_column='create_by'
        )
        results = filtered_query.all()
        ```

    示例2: 使用表别名（联表查询场景）
        ```python
        from app.decorators.data_permission import apply_data_permission
        from app.models import SysDataset, SysDept
        from sqlalchemy.orm import aliased

        user_info = {
            'user_id': 1,
            'dept_id': 10,
            'roles': ['USER']
        }

        # 创建表别名
        dataset_alias = aliased(SysDataset)

        # 联表查询
        query = mysql.session.query(dataset_alias, SysDept).join(
            SysDept, dataset_alias.parent_id == SysDept.id
        )

        # 应用数据权限过滤（使用表别名）
        filtered_query = apply_data_permission(
            query=query,
            user_info=user_info,
            model=SysDataset,
            dept_column='create_by',
            self_column='create_by',
            table_alias=dataset_alias
        )
        results = filtered_query.all()
        ```

    示例3: 跳过数据权限过滤（ROOT 用户）
        ```python
        from app.decorators.data_permission import apply_data_permission

        user_info = {
            'user_id': 1,
            'dept_id': 1,
            'roles': ['ROOT']
        }

        query = SysDataset.query
        filtered_query = apply_data_permission(
            query=query,
            user_info=user_info,
            model=SysDataset,
            dept_column='create_by',
            self_column='create_by'
        )
        # ROOT 用户会自动跳过数据权限过滤
        results = filtered_query.all()
        ```

    示例4: 在 Service 层中使用
        ```python
        from app.decorators.data_permission import apply_data_permission
        from app.models import SysDataset
        from flask import g

        class DatasetService:
            @staticmethod
            def get_dataset_list(page: int = 1, page_size: int = 10):
                # 获取当前用户信息
                user_info = {
                    'user_id': g.user.id,
                    'dept_id': g.user.dept_id,
                    'roles': g.user.roles
                }

                # 构建基础查询
                query = SysDataset.query.filter(SysDataset.deleted == 0)

                # 应用数据权限过滤
                filtered_query = apply_data_permission(
                    query=query,
                    user_info=user_info,
                    model=SysDataset,
                    dept_column='create_by',
                    self_column='create_by'
                )

                # 分页查询
                pagination = filtered_query.paginate(
                    page=page,
                    per_page=page_size,
                    error_out=False
                )

                return pagination.items, pagination.total
        ```
    """
    # 如果明确要求跳过数据权限，直接返回原查询
    if skip_data_permission:
        return query

    # 验证用户信息
    user_id = user_info.get('user_id')
    if not user_id:
        raise ValueError("用户信息必须包含 user_id 字段")

    # 获取用户的数据权限范围
    data_scope = _get_user_data_scope(user_info)

    # 如果没有数据权限配置（用户没有角色），默认返回空结果
    if data_scope is None:
        return query.filter(False)

    # 获取用户部门ID
    dept_id = user_info.get('dept_id')

    # 根据数据权限范围构建查询过滤器
    filtered_query = _build_data_scope_filter(
        query=query,
        model=model,
        data_scope=data_scope,
        user_id=user_id,
        dept_id=dept_id,
        dept_column=dept_column,
        self_column=self_column,
        table_alias=table_alias
    )

    return filtered_query


def apply_data_permission_to_dict_list(
    data_list: List[Dict[str, Any]],
    user_info: Dict[str, Any],
    dept_column: str = 'dept_id',
    self_column: str = 'user_id'
) -> List[Dict[str, Any]]:
    """
    对字典列表应用数据权限过滤

    适用于从缓存或其他非数据库来源获取数据后的过滤场景

    Args:
        data_list (List[Dict[str, Any]]): 数据字典列表
        user_info (Dict[str, Any]): 用户信息
        dept_column (str): 部门字段名（默认：'dept_id'）
        self_column (str): 用户ID字段名（默认：'user_id'）

    Returns:
        List[Dict[str, Any]]: 过滤后的数据列表

    示例：
        ```python
        from app.decorators.data_permission import apply_data_permission_to_dict_list

        # 从缓存获取数据
        cached_data = [
            {'id': 1, 'name': '数据1', 'dept_id': 10, 'user_id': 100},
            {'id': 2, 'name': '数据2', 'dept_id': 20, 'user_id': 101},
            {'id': 3, 'name': '数据3', 'dept_id': 10, 'user_id': 1},
        ]

        user_info = {
            'user_id': 1,
            'dept_id': 10,
            'roles': ['USER']
        }

        # 应用数据权限过滤
        filtered_data = apply_data_permission_to_dict_list(
            data_list=cached_data,
            user_info=user_info,
            dept_column='dept_id',
            self_column='user_id'
        )
        ```
    """
    # 获取用户的数据权限范围
    data_scope = _get_user_data_scope(user_info)

    # 如果是所有数据权限，不过滤
    if data_scope == DataScope.ALL:
        return data_list

    user_id = user_info.get('user_id')
    dept_id = user_info.get('dept_id')

    # 根据权限范围过滤数据
    filtered_data = []
    for item in data_list:
        if data_scope == DataScope.DEPT_AND_SUB:
            # 部门及子部门数据权限
            item_dept_id = item.get(dept_column)
            if item_dept_id:
                dept_ids = _get_dept_and_sub_dept_ids(dept_id)
                if item_dept_id in dept_ids:
                    filtered_data.append(item)

        elif data_scope == DataScope.DEPT:
            # 本部门数据权限
            item_dept_id = item.get(dept_column)
            if item_dept_id and item_dept_id == dept_id:
                filtered_data.append(item)

        elif data_scope == DataScope.SELF:
            # 本人数据权限
            item_user_id = item.get(self_column)
            if item_user_id and item_user_id == user_id:
                filtered_data.append(item)

    return filtered_data
