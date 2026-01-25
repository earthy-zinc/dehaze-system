from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import bcrypt
import jwt
from flask import current_app

from app.extensions import mysql
from app.models import SysUser, SysRole, SysUserRole


class UserService:
    """用户服务类，处理用户相关的业务逻辑"""

    @staticmethod
    def _hash_password(password: str) -> str:
        """
        使用BCrypt算法哈希密码，与Java版本保持一致

        Args:
            password (str): 明文密码

        Returns:
            str: 哈希后的密码
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    @staticmethod
    def _check_password(password: str, hashed: str) -> bool:
        """
        验证密码是否匹配

        Args:
            password (str): 明文密码
            hashed (str): 哈希后的密码

        Returns:
            bool: 是否匹配
        """
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    @staticmethod
    def get_user_by_username(username: str) -> Optional[SysUser]:
        """
        根据用户名获取用户信息

        Args:
            username (str): 用户名

        Returns:
            Optional[SysUser]: 用户对象，如果未找到返回None
        """
        return SysUser.query.filter_by(username=username, deleted=0).first()

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[SysUser]:
        """
        根据用户ID获取用户信息

        Args:
            user_id (int): 用户ID

        Returns:
            Optional[SysUser]: 用户对象，如果未找到返回None
        """
        return SysUser.query.filter_by(id=user_id, deleted=0).first()

    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[SysUser]:
        """
        验证用户身份

        Args:
            username (str): 用户名
            password (str): 密码（明文）

        Returns:
            Optional[SysUser]: 验证成功的用户对象，失败返回None
        """
        user = UserService.get_user_by_username(username)
        if user and UserService._check_password(password, user.password):
            return user
        return None

    @staticmethod
    def get_user_roles(user_id: int) -> List[SysRole]:
        """
        获取用户角色列表

        Args:
            user_id (int): 用户ID

        Returns:
            List[SysRole]: 用户角色列表
        """
        roles = mysql.session.query(SysRole).join(
            SysUserRole, SysRole.id == SysUserRole.role_id
        ).filter(
            SysUserRole.user_id == user_id,
            SysRole.deleted == 0,
            SysRole.status == 1
        ).all()
        return roles

    @staticmethod
    def get_user_permissions(user_id: int) -> List[str]:
        """
        获取用户权限列表（简化实现，实际项目中可能需要关联菜单权限表）

        Args:
            user_id (int): 用户ID

        Returns:
            List[str]: 用户权限标识列表
        """
        # 这里简化处理，实际项目中应该关联菜单权限表
        roles = UserService.get_user_roles(user_id)
        permissions = []
        for role in roles:
            permissions.append(f"role_{role.code}")
        return permissions

    @staticmethod
    def generate_token(user_id: int) -> str:
        """
        生成JWT令牌

        Args:
            user_id (int): 用户ID

        Returns:
            str: JWT令牌
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        token = jwt.encode(
            payload,
            current_app.config['SECRET_KEY'],
            algorithm='HS256'
        )
        return token

    @staticmethod
    def create_user(username: str, password: str, nickname: str = None) -> SysUser:
        """
        创建新用户

        Args:
            username (str): 用户名
            password (str): 密码
            nickname (str, optional): 昵称

        Returns:
            SysUser: 新创建的用户对象
        """
        hashed_password = UserService._hash_password(password)
        user = SysUser(
            username=username,
            password=hashed_password,
            nickname=nickname or username
        )
        mysql.session.add(user)
        mysql.session.commit()
        return user

    @staticmethod
    def create_user_with_roles(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建新用户并关联角色

        Args:
            data (Dict[str, Any]): 用户数据

        Returns:
            Dict[str, Any]: 创建结果
        """
        username = data.get('username')
        nickname = data.get('nickname', username)
        gender = data.get('gender')
        dept_id = data.get('deptId')
        mobile = data.get('mobile')
        email = data.get('email')
        role_ids = data.get('roleIds', [])

        if not username:
            return {'error': '用户名不能为空'}

        # 检查用户名是否已存在
        existing_user = UserService.get_user_by_username(username)
        if existing_user:
            return {'error': '用户名已存在'}

        # 创建用户
        user = SysUser(
            username=username,
            nickname=nickname,
            gender=gender,
            dept_id=dept_id,
            mobile=mobile,
            email=email,
            password=UserService._hash_password(current_app.config.get('DEFAULT_PASSWORD', '123456'))
        )
        mysql.session.add(user)
        mysql.session.flush()  # 获取用户ID但不提交事务

        # 关联角色
        if role_ids:
            for role_id in role_ids:
                user_role = SysUserRole(user_id=user.id, role_id=role_id)
                mysql.session.add(user_role)

        mysql.session.commit()

        return {
            'data': {
                'id': user.id,
                'username': user.username,
                'nickname': user.nickname
            }
        }

    @staticmethod
    def get_user_form_data(user_id: int) -> Optional[Dict[str, Any]]:
        """
        获取用户表单数据

        Args:
            user_id (int): 用户ID

        Returns:
            Optional[Dict[str, Any]]: 用户表单数据
        """
        user = UserService.get_user_by_id(user_id)
        if not user:
            return None

        # 获取用户角色ID列表
        role_ids = [role.id for role in UserService.get_user_roles(user_id)]

        return {
            'id': user.id,
            'username': user.username,
            'nickname': user.nickname,
            'gender': user.gender,
            'deptId': user.dept_id,
            'mobile': user.mobile,
            'email': user.email,
            'status': user.status,
            'roleIds': role_ids
        }

    @staticmethod
    def update_user_with_roles(user_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新用户信息并关联角色

        Args:
            user_id (int): 用户ID
            data (Dict[str, Any]): 用户数据

        Returns:
            Dict[str, Any]: 更新结果
        """
        user = UserService.get_user_by_id(user_id)
        if not user:
            return {'error': '用户不存在'}

        # 注意：用户名创建后不可修改，因此不处理 username 字段
        nickname = data.get('nickname')
        gender = data.get('gender')
        dept_id = data.get('deptId')
        mobile = data.get('mobile')
        email = data.get('email')
        role_ids = data.get('roleIds', [])
        status = data.get('status')

        # 更新用户信息（不包括 username）
        if nickname is not None:
            user.nickname = nickname
        if gender is not None:
            user.gender = gender
        if dept_id is not None:
            user.dept_id = dept_id
        if mobile is not None:
            user.mobile = mobile
        if email is not None:
            user.email = email
        if status is not None:
            user.status = status

        # 更新角色关联
        # 先删除原有角色关联
        mysql.session.query(SysUserRole).filter(SysUserRole.user_id == user_id).delete()

        # 添加新角色关联
        if role_ids:
            for role_id in role_ids:
                user_role = SysUserRole(user_id=user_id, role_id=role_id)
                mysql.session.add(user_role)

        mysql.session.commit()

        return {'data': '更新成功'}

    @staticmethod
    def update_password(user_id: int, new_password: str) -> bool:
        """
        更新用户密码

        Args:
            user_id (int): 用户ID
            new_password (str): 新密码

        Returns:
            bool: 是否更新成功
        """
        user = UserService.get_user_by_id(user_id)
        if user:
            user.password = UserService._hash_password(new_password)
            mysql.session.commit()
            return True
        return False

    @staticmethod
    def get_user_list(page: int = 1, page_size: int = 10, username: str = None,
                      keywords: str = None, status: int = None, dept_id: int = None,
                      create_time_start: str = None, create_time_end: str = None) -> tuple:
        """
        获取用户列表（分页）

        Args:
            page (int): 页码
            page_size (int): 每页数量
            username (str, optional): 用户名搜索条件
            keywords (str, optional): 关键词搜索（匹配用户名、昵称、手机号）
            status (int, optional): 状态筛选（1-正常，0-禁用）
            dept_id (int, optional): 部门ID筛选（包含下级部门）
            create_time_start (str, optional): 创建时间开始（格式：YYYY-MM-DD）
            create_time_end (str, optional): 创建时间结束（格式：YYYY-MM-DD）

        Returns:
            tuple: (用户列表, 总数)
        """
        query = SysUser.query.filter_by(deleted=0)

        if username:
            query = query.filter(SysUser.username.like(f'%{username}%'))

        if keywords:
            query = query.filter(
                (SysUser.username.like(f'%{keywords}%')) |
                (SysUser.nickname.like(f'%{keywords}%')) |
                (SysUser.mobile.like(f'%{keywords}%'))
            )

        if status is not None:
            query = query.filter(SysUser.status == status)

        # 部门树过滤：查询指定部门及其所有子部门的用户
        if dept_id:
            # 获取指定部门的 tree_path
            dept = SysDept.query.filter_by(id=dept_id, deleted=0).first()
            if dept:
                # 如果 dept 有 tree_path，使用 tree_path LIKE 查询子部门
                if dept.tree_path:
                    # 构建子部门查询：tree_path 以 dept.tree_path 开头
                    child_dept_ids = mysql.session.query(SysDept.id).filter(
                        SysDept.tree_path.like(f'{dept.tree_path}%'),
                        SysDept.deleted == 0
                    ).all()
                    dept_ids_to_query = [dept.id] + [d[0] for d in child_dept_ids]
                else:
                    # 如果没有 tree_path，只查询当前部门
                    dept_ids_to_query = [dept.id]
                query = query.filter(SysUser.dept_id.in_(dept_ids_to_query))

        # 创建时间范围过滤
        if create_time_start:
            try:
                start_datetime = datetime.strptime(create_time_start, '%Y-%m-%d')
                query = query.filter(SysUser.create_time >= start_datetime)
            except ValueError:
                pass  # 忽略格式错误的时间

        if create_time_end:
            try:
                end_datetime = datetime.strptime(create_time_end, '%Y-%m-%d') + timedelta(days=1)  # 包含结束日期当天
                query = query.filter(SysUser.create_time < end_datetime)
            except ValueError:
                pass  # 忽略格式错误的时间

        pagination = query.paginate(
            page=page,
            per_page=page_size,
            error_out=False
        )
        return pagination.items, pagination.total

    @staticmethod
    def update_user_status(user_id: int, status: int) -> bool:
        """
        更新用户状态

        Args:
            user_id (int): 用户ID
            status (int): 状态（1-正常，0-禁用）

        Returns:
            bool: 是否更新成功
        """
        user = UserService.get_user_by_id(user_id)
        if user:
            # Super admin 保护：username='root' 的用户不能修改状态
            if user.username == 'root':
                return False
            user.status = status
            mysql.session.commit()
            return True
        return False

    @staticmethod
    def delete_user(user_id: int) -> bool:
        """
        删除用户（逻辑删除）

        Args:
            user_id (int): 用户ID

        Returns:
            bool: 是否删除成功
        """
        user = UserService.get_user_by_id(user_id)
        if user:
            # Super admin 保护：username='root' 的用户不能删除
            if user.username == 'root':
                return False
            user.deleted = 1
            mysql.session.commit()
            return True
        return False

    @staticmethod
    def import_users(worksheet, dept_id: int) -> Dict[str, Any]:
        """
        从 Excel 工作表导入用户

        Args:
            worksheet: openpyxl 工作表对象
            dept_id (int): 目标部门ID

        Returns:
            Dict[str, Any]: 导入结果，包含成功数、失败数和失败明细
        """
        success_count = 0
        failed_count = 0
        failures = []

        # 从第二行开始读取（第一行是表头）
        for row_num, row in enumerate(worksheet.iter_rows(min_row=2, values_only=True), start=2):
            try:
                # 解析行数据（根据模板定义的列顺序）
                # 列顺序：用户名, 昵称, 密码, 邮箱, 手机号, 性别, 部门ID, 角色ID
                username = row[0]
                nickname = row[1]
                password = row[2]
                email = row[3]
                mobile = row[4]
                gender = row[5]
                user_dept_id = row[6] if row[6] else dept_id  # 如果指定了部门则使用指定部门
                role_ids_str = row[7]

                # 验证必填字段
                if not username or not nickname:
                    failures.append({
                        'row': row_num,
                        'username': username or '',
                        'reason': '用户名或昵称不能为空'
                    })
                    failed_count += 1
                    continue

                # 检查用户名是否已存在
                existing_user = UserService.get_user_by_username(username)
                if existing_user:
                    failures.append({
                        'row': row_num,
                        'username': username,
                        'reason': '用户名已存在'
                    })
                    failed_count += 1
                    continue

                # 解析性别
                if gender == '男':
                    gender_value = 1
                elif gender == '女':
                    gender_value = 0
                else:
                    gender_value = 1  # 默认为男

                # 解析角色ID列表
                role_ids = []
                if role_ids_str:
                    role_ids = [int(rid.strip()) for rid in str(role_ids_str).split(',') if rid.strip()]

                # 创建用户
                user = SysUser(
                    username=username,
                    nickname=nickname,
                    password=UserService._hash_password(password if password else current_app.config.get('DEFAULT_PASSWORD', '123456')),
                    email=email,
                    mobile=mobile,
                    gender=gender_value,
                    dept_id=user_dept_id,
                    status=1  # 默认启用
                )
                mysql.session.add(user)
                mysql.session.flush()  # 获取用户ID但不提交事务

                # 关联角色
                if role_ids:
                    for role_id in role_ids:
                        user_role = SysUserRole(user_id=user.id, role_id=role_id)
                        mysql.session.add(user_role)

                success_count += 1

            except Exception as e:
                failures.append({
                    'row': row_num,
                    'username': row[0] if row else '',
                    'reason': str(e)
                })
                failed_count += 1

        # 提交事务
        mysql.session.commit()

        return {
            'successCount': success_count,
            'failedCount': failed_count,
            'failures': failures
        }
