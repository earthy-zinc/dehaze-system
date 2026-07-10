"""
用户模块 Schema 模型
"""
from typing import List, Optional, Set

from app.models.schema.common import BasePageQuery
from pydantic import BaseModel, Field

# ==================== 查询参数模型 ====================


class UserPageQuery(BasePageQuery):
    """用户分页查询参数"""
    keywords: Optional[str] = Field(
        default=None, description="关键词(用户名/昵称/手机号)")
    status: Optional[int] = Field(
        default=None, ge=0, le=1, description="用户状态(1:启用;0:禁用)")
    deptId: Optional[int] = Field(default=None, description="部门ID")
    startTime: Optional[str] = Field(default=None, description="创建时间-开始时间")
    endTime: Optional[str] = Field(default=None, description="创建时间-结束时间")


class UserStatusQuery(BaseModel):
    """用户状态修改查询参数"""
    status: int = Field(..., ge=0, le=1, description="状态(1-启用；0-停用)")


class UserPasswordQuery(BaseModel):
    """用户密码修改查询参数"""
    pass


# ==================== 路径参数模型 ====================

class UserIdPath(BaseModel):
    """用户ID路径参数"""
    user_id: int = Field(..., description="用户ID")


class UserIdsPath(BaseModel):
    """批量删除路径参数"""
    ids: str = Field(..., description="用户ID，多个以英文逗号(,)分隔")


# ==================== 请求体模型 ====================

class LoginForm(BaseModel):
    """登录表单"""
    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")
    captchaKey: str = Field(..., description="验证码Key")
    captchaCode: str = Field(..., description="验证码")


class RegisterForm(BaseModel):
    """注册表单"""
    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")
    nickname: str = Field(..., min_length=1, description="昵称")


class UserForm(BaseModel):
    """用户表单"""
    id: Optional[int] = Field(default=None, description="用户ID")
    username: str = Field(..., min_length=1, description="用户名")
    nickname: str = Field(..., min_length=1, description="昵称")
    mobile: Optional[str] = Field(
        default=None,
        pattern=r"^$|^1(3\d|4[5-9]|5[0-35-9]|6[2567]|7[0-8]|8\d|9[0-35-9])\d{8}$",
        description="手机号码"
    )
    gender: Optional[int] = Field(default=None, description="性别")
    avatar: Optional[str] = Field(default=None, description="用户头像")
    email: Optional[str] = Field(default=None, description="邮箱")
    status: Optional[int] = Field(
        default=None, ge=0, le=1, description="用户状态(1:正常;0:禁用)")
    deptId: Optional[int] = Field(default=None, description="部门ID")
    roleIds: List[int] = Field(..., min_length=1, description="角色ID集合")


class PasswordForm(BaseModel):
    """密码表单"""
    password: str = Field(..., min_length=1, description="密码")


# ==================== 响应模型 ====================

class LoginData(BaseModel):
    """登录响应数据"""
    tokenType: str = Field(description="Token 类型")
    accessToken: str = Field(description="访问令牌")
    user: dict = Field(description="用户信息")


class CaptchaData(BaseModel):
    """验证码响应数据"""
    captchaKey: str = Field(description="验证码 key")
    captchaBase64: str = Field(description="验证码图片 Base64")


class LoginUserVO(BaseModel):
    """登录用户信息"""
    id: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="昵称")


class LoginVO(BaseModel):
    """登录响应"""
    token: str = Field(description="访问令牌")
    user: LoginUserVO = Field(description="用户信息")


class UserInfoVO(BaseModel):
    """用户信息响应"""
    userId: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="用户昵称")
    avatar: Optional[str] = Field(default=None, description="头像地址")
    roles: Set[str] = Field(description="用户角色编码集合")
    perms: Set[str] = Field(description="用户权限标识集合")


class UserPageVO(BaseModel):
    """用户分页VO"""
    id: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="用户昵称")
    mobile: Optional[str] = Field(default=None, description="手机号")
    genderLabel: Optional[str] = Field(default=None, description="性别")
    avatar: Optional[str] = Field(default=None, description="用户头像地址")
    status: int = Field(description="用户状态(1:启用;0:禁用)")
    email: Optional[str] = Field(default=None, description="邮箱")
    deptName: Optional[str] = Field(default=None, description="部门名称")
    roleNames: Optional[str] = Field(
        default=None, description="角色名称，多个使用英文逗号(,)分割")
    createTime: Optional[str] = Field(default=None, description="创建时间")


class UserFormVO(BaseModel):
    """用户表单VO"""
    id: Optional[int] = Field(default=None, description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="昵称")
    mobile: Optional[str] = Field(default=None, description="手机号码")
    gender: Optional[int] = Field(default=None, description="性别")
    avatar: Optional[str] = Field(default=None, description="用户头像")
    email: Optional[str] = Field(default=None, description="邮箱")
    status: Optional[int] = Field(default=None, description="用户状态(1:正常;0:禁用)")
    deptId: Optional[int] = Field(default=None, description="部门ID")
    roleIds: List[int] = Field(description="角色ID集合")


class UserCreateVO(BaseModel):
    """创建用户响应VO"""
    id: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="昵称")


class UserImportVO(BaseModel):
    """用户导入结果VO"""
    successCount: int = Field(description="成功数量")
    failedCount: int = Field(description="失败数量")


class UserDeleteVO(BaseModel):
    """用户删除结果VO"""
    deleted_count: int = Field(description="删除数量")


class CurrentUserVO(BaseModel):
    """当前用户信息VO"""
    userId: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: Optional[str] = Field(default=None, description="昵称")
    roles: List[str] = Field(description="角色列表")
    permissions: List[str] = Field(description="权限列表（最多显示10个）")
