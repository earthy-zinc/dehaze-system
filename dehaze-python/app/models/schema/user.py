"""
用户模块 Schema 模型
"""

from pydantic import BaseModel, Field, field_validator

from app.models.schema.common import validate_no_xss


class LoginForm(BaseModel):
    """登录表单"""

    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")
    captchaKey: str = Field(..., description="验证码Key")
    captchaCode: str = Field(..., description="验证码")
    rememberMe: bool | None = Field(default=None, description="记住我")


class RegisterForm(BaseModel):
    """注册表单"""

    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")
    nickname: str = Field(..., min_length=1, max_length=64, description="昵称")
    captchaKey: str = Field(..., description="验证码Key")
    captchaCode: str = Field(..., description="验证码")

    nickname_no_xss_validator = field_validator("nickname")(validate_no_xss)


class UserForm(BaseModel):
    """用户表单"""

    id: int | None = Field(default=None, description="用户ID")
    username: str = Field(..., min_length=1, description="用户名")
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
    deptId: int = Field(..., description="部门ID")
    roleIds: list[int] = Field(..., min_length=1, description="角色ID集合")

    nickname_no_xss_validator = field_validator("nickname")(validate_no_xss)


class PasswordForm(BaseModel):
    """密码表单"""

    password: str = Field(..., min_length=1, description="密码")


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


class UserCreateVO(BaseModel):
    """创建用户响应VO"""

    id: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="昵称")


class UserDeleteVO(BaseModel):
    """用户删除结果VO"""

    deleted_count: int = Field(description="删除数量")


class CurrentUserVO(BaseModel):
    """当前用户信息VO"""

    userId: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str | None = Field(default=None, description="昵称")
    roles: list[str] = Field(description="角色列表")
    perms: list[str] = Field(description="权限列表")
