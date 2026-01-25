"""
装饰器模块

提供各种装饰器功能：
    - data_permission: 数据权限装饰器
    - anti_repeat: 防重复提交装饰器
    - captcha: 验证码校验装饰器
    - idempotent: 幂等性装饰器
    - permission: 权限校验装饰器

注意：限流功能请直接使用 flask-limiter，示例：
    from app.extensions import limiter
    
    @app.route('/api')
    @limiter.limit("10 per minute")
    def api():
        pass
"""

from app.decorators.data_permission import (
    DataScope,
    apply_data_permission,
    apply_data_permission_to_dict_list
)

from app.decorators.anti_repeat import anti_repeat
from app.decorators.captcha import verify_captcha
from app.decorators.idempotent import idempotent
from app.decorators.permission import (
    has_permission,
    has_perm,
    has_any_perms,
    has_all_perms,
    get_current_permissions
)

__all__ = [
    # Data permission
    'DataScope',
    'apply_data_permission',
    'apply_data_permission_to_dict_list',
    # Anti repeat
    'anti_repeat',
    # Captcha
    'verify_captcha',
    # Idempotent
    'idempotent',
    # Permission
    'has_permission',
    'has_perm',
    'has_any_perms',
    'has_all_perms',
    'get_current_permissions'
]
