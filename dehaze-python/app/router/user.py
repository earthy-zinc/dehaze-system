from io import BytesIO
from typing import Optional

import openpyxl
from app.core.code import ResultCode
from app.core.result import Result, error, success
from app.database import get_db
from app.decorators.permission import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.common import PageResult
from app.models.schema.user import (CurrentUserVO, PasswordForm, UserCreateVO,
                                    UserDeleteVO, UserForm, UserFormVO,
                                    UserImportVO, UserPageVO)
from app.service.user_service import UserService
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/users", tags=["用户管理"])


@router.get("/template", summary="下载用户导入模板")
async def download_template(
    user: UserContext = Depends(get_current_user),
):
    output = UserService.generate_import_template()

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=user_import_template.xlsx",
        },
    )


@router.get("/page", summary="获取用户分页列表", response_model=Result[PageResult[UserPageVO]])
async def get_user_page(
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页记录数"),
    keywords: Optional[str] = Query(
        default=None, description="关键词(用户名/昵称/手机号)"),
    status: Optional[int] = Query(
        default=None, ge=0, le=1, description="用户状态(1:启用;0:禁用)"),
    deptId: Optional[int] = Query(default=None, description="部门ID"),
    startTime: Optional[str] = Query(default=None, description="创建时间-开始时间"),
    endTime: Optional[str] = Query(default=None, description="创建时间-结束时间"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    users, total = await UserService.get_user_list(
        db,
        page=pageNum,
        page_size=pageSize,
        keywords=keywords,
        status=status,
        dept_id=deptId,
        create_time_start=startTime,
        create_time_end=endTime,
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
            create_time_str = create_time.strftime("%Y-%m-%d") if not isinstance(create_time, str) else create_time[:10]
        else:
            create_time_str = None

        user_list.append({
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
        })

    return success({
        "list": user_list,
        "total": total,
    })


@router.get("/_export", summary="导出用户数据")
async def export_users(
    keywords: Optional[str] = Query(
        default=None, description="关键词(用户名/昵称/手机号)"),
    status: Optional[int] = Query(
        default=None, ge=0, le=1, description="用户状态(1:启用;0:禁用)"),
    deptId: Optional[int] = Query(default=None, description="部门ID"),
    startTime: Optional[str] = Query(default=None, description="创建时间-开始时间"),
    endTime: Optional[str] = Query(default=None, description="创建时间-结束时间"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    output = await UserService.export_users(
        db,
        keywords=keywords,
        status=status,
        dept_id=deptId,
        create_time_start=startTime,
        create_time_end=endTime,
    )

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=users_export.xlsx",
        },
    )


@router.post("/_import", summary="导入用户数据", response_model=Result[UserImportVO])
@require_permission("sys:user:add")
async def import_users(
    file: UploadFile = File(..., description="Excel 文件"),
    deptId: int = Form(..., description="目标部门ID"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    if not file.filename or not file.filename.endswith((".xls", ".xlsx")):
        return error("仅支持 .xls 和 .xlsx 格式的文件", code=ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH.code)

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        return error("文件大小超过限制（最大 10MB）", code=ResultCode.USER_UPLOAD_FILE_SIZE_EXCEEDS.code)

    try:
        wb = openpyxl.load_workbook(BytesIO(contents))
        ws = wb.active

        result = await UserService.import_users(db, ws, dept_id=deptId)

        return success(
            result,
            msg=f'导入完成，成功{result["successCount"]}条，失败{result["failedCount"]}条',
        )
    except Exception as e:
        return error(f"导入失败: {str(e)}")


@router.post("/", summary="新增用户", response_model=Result[UserCreateVO])
@require_permission("sys:user:add")
async def create_user(
    body: UserForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = body.model_dump(exclude_none=True)
    new_user = await UserService.create_user_with_roles(db, data)

    return success({
        "id": new_user.id,
        "username": new_user.username,
        "nickname": new_user.nickname,
    }, msg="新增成功")


@router.get("/{user_id}/form", summary="获取用户表单数据", response_model=Result[UserFormVO])
async def get_user_form(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    user_data = await UserService.get_user_form_data(db, user_id)

    if not user_data:
        return success(None)

    return success(user_data)


@router.put("/{user_id}", summary="更新用户信息", response_model=Result[None])
@require_permission("sys:user:edit")
async def update_user(
    user_id: int,
    body: UserForm,
    db: AsyncSession = Depends(get_db),
    current_user: UserContext = Depends(get_current_user),
):
    data = body.model_dump(exclude_none=True)
    await UserService.update_user_with_roles(db, user_id, data)

    return success(msg="更新成功")


@router.patch("/{user_id}/status", summary="更新用户状态", response_model=Result[None])
async def update_user_status(
    user_id: int,
    status: int = Query(..., ge=0, le=1, description="状态(1-启用；0-停用)"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await UserService.update_user_status(db, user_id, status)

    return success(msg="更新成功")


@router.patch("/{user_id}/password", summary="修改用户密码", response_model=Result[None])
async def update_password(
    user_id: int,
    body: PasswordForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    # 只能修改自己的密码，或者有重置密码权限才能修改他人密码
    if user_id != user.id and not (user.is_root or "sys:user:password:reset" in user.permissions or "*" in user.permissions):
        return error("无权修改其他用户的密码", code=ResultCode.ACCESS_UNAUTHORIZED.code)

    await UserService.update_password(db, user_id, body.password)

    return success(msg="修改成功")


@router.delete("/{ids}", summary="删除用户", response_model=Result[UserDeleteVO])
@require_permission("sys:user:delete")
async def delete_users(
    ids: str,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await UserService.delete_users(db, ids)

    return success(result, msg=f"成功删除 {result['deleted_count']} 个用户")


@router.get("/me", summary="获取当前用户信息", response_model=Result[CurrentUserVO], tags=["用户信息"])
async def get_current_user_info(
    user: UserContext = Depends(get_current_user),
):
    return success(
        {
            "userId": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "roles": user.roles,
            "permissions": user.permissions[:10] if user.permissions else [],
        }
    )
