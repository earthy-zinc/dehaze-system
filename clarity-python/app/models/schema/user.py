"""
用户模块 Schema 模型
"""

import re

from pydantic import BaseModel, Field, field_validator

from app.models.schema.common import validate_no_xss


class LoginForm(BaseModel):
    """登录表单"""

    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")
    captchaKey: str = Field(..., description="验证码Key")
    captchaCode: str = Field(..., description="验证码")
    rememberMe: bool | None = Field(default=None, description="记住我")
    deviceType: str | None = Field(
        default=None,
        pattern=r"^(web|android|flutter|miniprogram)$",
        description="设备类型(用于多端会话区分，默认 web)",
    )


class RegisterForm(BaseModel):
    """注册表单"""

    username: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_]{3,32}$",
        description="用户名(3-32位，仅字母、数字、下划线)",
    )
    password: str = Field(..., min_length=8, max_length=20, description="密码")
    nickname: str = Field(..., min_length=1, max_length=64, description="昵称")
    captchaKey: str = Field(..., description="验证码Key")
    captchaCode: str = Field(..., description="验证码")

    @field_validator("password")
    @classmethod
    def password_must_contain_letter_and_digit(cls, v: str) -> str:
        if not re.search(r"[A-Za-z]", v) or not re.search(r"\d", v):
            raise ValueError("密码必须包含字母和数字")
        return v

    nickname_no_xss_validator = field_validator("nickname")(validate_no_xss)


class UserForm(BaseModel):
    """用户表单"""

    id: int | None = Field(default=None, description="用户ID")
    username: str = Field(..., min_length=1, max_length=64, description="用户名")
    nickname: str = Field(..., min_length=1, max_length=64, description="昵称")
    mobile: str | None = Field(
        default=None,
        pattern=r"^$|^1(3\d|4[5-9]|5[0-35-9]|6[2567]|7[0-8]|8\d|9[0-35-9])\d{8}$",
        description="手机号码",
    )
    gender: int | None = Field(default=None, description="性别")
    avatar: str | None = Field(default=None, description="用户头像")
    email: str | None = Field(
        default=None,
        pattern=r"^$|^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$",
        description="邮箱",
    )
    status: int | None = Field(default=None, ge=0, le=1, description="用户状态(1:正常;0:禁用)")
    userType: str | None = Field(
        default=None,
        max_length=16,
        description=(
            "用户类型(personal:个人;enterprise:企业)，缺省 personal（三端统一宽松校验口径）"
        ),
    )
    deptId: int = Field(..., description="部门ID")
    roleIds: list[int] = Field(..., min_length=1, description="角色ID集合")

    nickname_no_xss_validator = field_validator("nickname")(validate_no_xss)


class PasswordForm(BaseModel):
    """密码表单"""

    password: str = Field(..., min_length=1, description="密码")


class PasswordChangeForm(BaseModel):
    """个人改密表单"""

    oldPassword: str = Field(..., min_length=1, description="旧密码")
    newPassword: str = Field(..., min_length=8, max_length=20, description="新密码")

    @field_validator("newPassword")
    @classmethod
    def new_password_must_contain_letter_and_digit(cls, v: str) -> str:
        if not re.search(r"[A-Za-z]", v) or not re.search(r"\d", v):
            raise ValueError("密码必须包含字母和数字")
        return v


class LoginData(BaseModel):
    """登录响应数据"""

    sessionId: str = Field(description="会话ID")
    user: dict = Field(description="用户信息")


class CaptchaData(BaseModel):
    """验证码响应数据"""

    captchaKey: str = Field(description="验证码 key")
    captchaBase64: str = Field(description="验证码图片 Base64")


class UserInfoVO(BaseModel):
    """用户信息响应"""

    userId: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="用户昵称")
    avatar: str | None = Field(default=None, description="头像地址")
    roles: set[str] = Field(description="用户角色编码集合")
    perms: set[str] = Field(description="用户权限标识集合")


class UserPageVO(BaseModel):
    """用户分页VO"""

    id: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="用户昵称")
    mobile: str | None = Field(default=None, description="手机号")
    genderLabel: str | None = Field(default=None, description="性别")
    avatar: str | None = Field(default=None, description="用户头像地址")
    status: int = Field(description="用户状态(1:启用;0:禁用)")
    email: str | None = Field(default=None, description="邮箱")
    deptName: str | None = Field(default=None, description="部门名称")
    roleNames: str | None = Field(default=None, description="角色名称，多个使用英文逗号(,)分割")
    userType: str | None = Field(
        default=None, description="用户类型(personal:个人;enterprise:企业)"
    )
    memberLevel: str | None = Field(
        default=None, description="会员等级(level_0~level_3)，无会员记录为 null"
    )
    memberExpireTime: str | None = Field(
        default=None, description="会员到期时间，null 表示成长值维持"
    )
    quotaUsage: str | None = Field(default=None, description="配额使用(已用/总量，如 30/100)")
    createTime: str | None = Field(default=None, description="创建时间")


class UserFormVO(BaseModel):
    """用户表单VO"""

    id: int | None = Field(default=None, description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="昵称")
    mobile: str | None = Field(default=None, description="手机号码")
    gender: int | None = Field(default=None, description="性别")
    avatar: str | None = Field(default=None, description="用户头像")
    email: str | None = Field(default=None, description="邮箱")
    status: int | None = Field(default=None, description="用户状态(1:正常;0:禁用)")
    deptId: int | None = Field(default=None, description="部门ID")
    roleIds: list[int] = Field(description="角色ID集合")


class UserDeleteVO(BaseModel):
    """用户删除结果VO"""

    deleted_count: int = Field(description="删除数量")


class CurrentUserVO(BaseModel):
    """当前用户信息VO"""

    userId: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str | None = Field(default=None, description="昵称")
    avatar: str | None = Field(default=None, description="头像地址")
    roles: list[str] = Field(description="角色列表")
    perms: list[str] = Field(description="权限列表")
