import fnmatch
from functools import wraps
from typing import List, Optional, Union

from app.utils.code import ResultCode
from app.utils.result import warning
from app.utils.logging import logger

try:
    from flask_jwt_extended import verify_jwt_in_request, get_jwt
except ImportError:
    logger.warning("Flask-JWT-Extended 未安装，权限装饰器将不可用")


def has_permission(
    required_permissions: Union[str, List[str]],
    all_required: bool = False,
    skip_on_missing: bool = False
):
    """
    权限校验装饰器

    检查当前用户是否具有指定的权限，支持通配符匹配。

    Args:
        required_permissions: 需要的权限，可以是单个权限字符串或权限列表
        all_required: 是否需要所有权限（True）还是任一权限（False），默认 False
        skip_on_missing: 如果缺少权限信息是否跳过校验（默认 False）

    Usage:
        # 基本使用：需要单一权限
        @app.route('/api/users', methods=['GET'])
        @has_permission('user:list')
        def list_users():
            return success()

        # 需要多个权限中的任一个
        @app.route('/api/users', methods=['POST'])
        @has_permission(['user:add', 'user:import'])
        def add_user():
            return success()

        # 需要所有权限
        @app.route('/api/users', methods=['DELETE'])
        @has_permission(['user:delete', 'audit:pass'], all_required=True)
        def delete_user():
            return success()

        # 支持通配符
        @app.route('/api/admin', methods=['GET'])
        @has_permission('admin:*')
        def admin_panel():
            return success()

        @app.route('/api/user', methods=['GET'])
        @has_permission('*:*')  # 所有权限
        def all_access():
            return success()

    注意:
        1. 用户权限列表从 JWT token 的 'permissions' 或 'perms' 字段获取
        2. 通配符支持: * 匹配任意多个字符，? 匹配单个字符
        3. 如果 JWT 中没有权限信息，会根据 skip_on_missing 参数决定行为
        4. 空权限列表 '*' 表示超级管理员（拥有所有权限）
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # 验证 JWT
                verify_jwt_in_request()
                token = get_jwt()

                # 获取用户权限列表
                user_permissions = _get_user_permissions(token)

                # 如果没有权限信息
                if not user_permissions:
                    if skip_on_missing:
                        logger.info("用户权限信息为空，跳过权限校验")
                        return f(*args, **kwargs)
                    else:
                        logger.warning("用户权限信息为空")
                        return warning(ResultCode.ACCESS_UNAUTHORIZED)

                # 标准化权限列表（如果是字符串，转为列表）
                if isinstance(required_permissions, str):
                    required_perms = [required_permissions]
                else:
                    required_perms = required_permissions

                # 检查权限
                has_perm = _check_permissions(user_permissions, required_perms, all_required)

                if not has_perm:
                    logger.warning(
                        f"权限校验失败，用户权限: {user_permissions}, "
                        f"需要权限: {required_permissions}"
                    )
                    return warning(ResultCode.FORBIDDEN_OPERATION)

                logger.debug(f"权限校验通过: {required_permissions}")
                return f(*args, **kwargs)

            except Exception as e:
                logger.error(f"权限校验装饰器异常: {str(e)}", exc_info=True)
                # 如果 JWT 认证失败，返回错误
                if skip_on_missing:
                    return f(*args, **kwargs)
                else:
                    return warning(ResultCode.ACCESS_UNAUTHORIZED)

        return decorated_function
    return decorator


def _get_user_permissions(token: dict) -> Optional[List[str]]:
    """
    从 JWT token 获取用户权限列表

    Args:
        token: JWT token payload

    Returns:
        用户权限列表
    """
    # 尝试从不同字段获取权限
    permissions = token.get('permissions') or token.get('perms') or token.get('perms_list')

    if not permissions:
        return None

    # 标准化为列表
    if isinstance(permissions, str):
        # 字符串格式: "user:list,user:add,role:edit"
        return [p.strip() for p in permissions.split(',') if p.strip()]
    elif isinstance(permissions, list):
        # 列表格式: ["user:list", "user:add", "role:edit"]
        return [str(p) for p in permissions if p]

    return None


def _check_permissions(
    user_permissions: List[str],
    required_permissions: List[str],
    all_required: bool
) -> bool:
    """
    检查用户是否具有所需的权限

    Args:
        user_permissions: 用户权限列表
        required_permissions: 需要的权限列表
        all_required: 是否需要所有权限

    Returns:
        是否具有权限
    """
    # 如果用户权限列表中包含 '*'，表示超级管理员
    if '*' in user_permissions or '*:*' in user_permissions:
        return True

    # 检查每个需要的权限
    matched_count = 0

    for required_perm in required_permissions:
        has_this_perm = _check_permission_with_wildcard(user_permissions, required_perm)
        if has_this_perm:
            matched_count += 1
            if not all_required:
                # 只要有一个匹配即可
                return True

    # 如果需要所有权限，检查是否全部匹配
    if all_required:
        return matched_count == len(required_permissions)

    return False


def _check_permission_with_wildcard(user_permissions: List[str], required_perm: str) -> bool:
    """
    使用通配符检查权限

    支持 * 和 ? 通配符，使用 fnmatch 进行匹配。

    Args:
        user_permissions: 用户权限列表
        required_perm: 需要的权限（可能包含通配符）

    Returns:
        是否具有权限

    Usage:
        user_permissions = ['user:list', 'user:add', 'admin:config']
        _check_permission_with_wildcard(user_permissions, 'user:list')
        # 结果: True

        _check_permission_with_wildcard(user_permissions, 'user:*')
        # 结果: True

        _check_permission_with_wildcard(user_permissions, 'user:de?ete')
        # 结果: True (匹配 user:delete)
    """
    # 精确匹配
    if required_perm in user_permissions:
        return True

    # 通配符匹配
    for user_perm in user_permissions:
        # 检查用户权限是否匹配需要的权限（支持通配符）
        if fnmatch.fnmatch(user_perm, required_perm):
            return True

        # 检查需要的权限是否匹配用户权限（支持通配符）
        if fnmatch.fnmatch(required_perm, user_perm):
            return True

    return False


# ==================== 辅助函数 ====================

def get_current_permissions() -> Optional[List[str]]:
    """
    获取当前用户的权限列表

    Returns:
        当前用户的权限列表，如果未登录则返回 None
    """
    try:
        from flask_jwt_extended import get_jwt
        token = get_jwt()
        return _get_user_permissions(token)
    except Exception:
        return None


def has_perm(permission: str) -> bool:
    """
    检查当前用户是否具有指定权限

    Args:
        permission: 需要的权限

    Returns:
        是否具有权限
    """
    user_permissions = get_current_permissions()
    if not user_permissions:
        return False

    return _check_permission_with_wildcard(user_permissions, permission)


def has_any_perms(permissions: List[str]) -> bool:
    """
    检查当前用户是否具有任一权限

    Args:
        permissions: 权限列表

    Returns:
        是否具有任一权限
    """
    user_permissions = get_current_permissions()
    if not user_permissions:
        return False

    for perm in permissions:
        if _check_permission_with_wildcard(user_permissions, perm):
            return True

    return False


def has_all_perms(permissions: List[str]) -> bool:
    """
    检查当前用户是否具有所有权限

    Args:
        permissions: 权限列表

    Returns:
        是否具有所有权限
    """
    user_permissions = get_current_permissions()
    if not user_permissions:
        return False

    matched_count = 0
    for perm in permissions:
        if _check_permission_with_wildcard(user_permissions, perm):
            matched_count += 1

    return matched_count == len(permissions)
