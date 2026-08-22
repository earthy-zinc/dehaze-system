from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.decorators.permission import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.common import PageResult
from app.models.schema.user import (
    PasswordForm,
    UserCreateVO,
    UserDeleteVO,
    UserForm,
    UserFormVO,
    UserPageVO,
)
from app.service.user_service import user_service

router = APIRouter(prefix="/api/v1/users", tags=["用户管理"])


@router.get("/page", summary="获取用户分页列表", response_model=Result[PageResult[UserPageVO]])
async def get_user_page(
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页记录数"),
    keywords: str | None = Query(default=None, description="关键词(用户名/昵称/手机号)"),
    status: int | None = Query(default=None, ge=0, le=1, description="用户状态(1:启用;0:禁用)"),
    deptId: int | None = Query(default=None, description="部门ID"),
    startTime: str | None = Query(default=None, description="创建时间-开始时间"),
    endTime: str | None = Query(default=None, description="创建时间-结束时间"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    users, total = await user_service.get_user_list(
        db,
        page=pageNum,
        page_size=pageSize,
        keywords=keywords,
        status=status,
        dept_id=deptId,
        create_time_start=startTime,
        create_time_end=endTime,
        current_user=user,
    )

    user_list = []
    for u in users:
        gender = u.get("gender", 1)
        if gender == 1:
            gender_label = "男"
        elif gender == 2:
            gender_label = "女"
        else:
            gender_label = "未知"

        create_time = u.get("create_time")
        if create_time:
            create_time_str = (
                create_time.strftime("%Y-%m-%d")
                if not isinstance(create_time, str)
                else create_time[:10]
            )
        else:
            create_time_str = None

        user_list.append(
            {
                "id": u["id"],
                "username": u["username"],
                "nickname": u["nickname"],
                "mobile": u.get("mobile"),
                "genderLabel": gender_label,
                "avatar": u.get("avatar"),
                "status": u.get("status"),
                "email": u.get("email"),
                "deptName": u.get("deptName"),
                "roleNames": u.get("roleNames"),
                "createTime": create_time_str,
            }
        )

    return success(
        {
            "list": user_list,
            "total": total,
        }
    )


@router.post("", summary="新增用户", response_model=Result[UserCreateVO])
@require_permission("sys:user:add")
async def create_user(
    body: UserForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = body.model_dump(exclude_none=True)
    await user_service.create_user_with_roles(db, data)

    return success(msg="一切ok")


@router.get("/{user_id}/form", summary="获取用户表单数据", response_model=Result[UserFormVO])
async def get_user_form(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    user_data = await user_service.get_user_form_data(db, user_id)

    if not user_data:
        return success(None)

    return success(user_data)


@router.put("/{user_id}", summary="更新用户信息", response_model=Result[None])
@require_permission("sys:user:edit")
async def update_user(
    user_id: int,
    body: UserForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = body.model_dump(exclude_none=True)
    await user_service.update_user_with_roles(db, user_id, data)

    return success(msg="一切ok")


@router.patch("/{user_id}/status", summary="更新用户状态", response_model=Result[None])
async def update_user_status(
    user_id: int,
    status: int = Query(..., ge=0, le=1, description="状态(1-启用；0-停用)"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await user_service.update_user_status(db, user_id, status)

    return success(msg="一切ok")


@router.patch("/{user_id}/password", summary="修改用户密码", response_model=Result[None])
async def update_password(
    user_id: int,
    body: PasswordForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    # 只能修改自己的密码，或者有重置密码权限才能修改他人密码
    if user_id != user.id and not (
        user.is_root or "sys:user:password:reset" in user.permissions or "*" in user.permissions
    ):
        raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权修改其他用户的密码")

    await user_service.update_password(db, user_id, body.password)

    return success(msg="修改成功")


@router.delete("/{ids}", summary="删除用户", response_model=Result[UserDeleteVO])
@require_permission("sys:user:delete")
async def delete_users(
    ids: str,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await user_service.delete_users(db, ids, current_user=user)

    return success(result, msg=f"成功删除 {result['deleted_count']} 个用户")
