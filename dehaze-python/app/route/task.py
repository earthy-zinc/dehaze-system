"""
导出任务路由模块 - 使用 flask-openapi3 自动生成 Swagger 文档
处理导出任务的创建、查询、下载和取消操作
"""

from flask import redirect
from flask_openapi3 import APIBlueprint, Tag

from app.models.schema.task import (
    TaskIdPath, ExportTaskCreateForm, TaskVO
)
from app.service.task_service import TaskService
from app.utils.jwt_util import jwt_required, get_current_user_id
from app.utils.result import success, error


# 定义 Tag
task_tag = Tag(name="导出任务管理", description="导出任务相关接口")

# 创建 APIBlueprint（自动携带安全配置）
task_blueprint = APIBlueprint(
    "task",
    __name__,
    url_prefix="/api/v1/export-tasks",
    abp_tags=[task_tag],
    abp_security=[{"BearerAuth": []}]
)


# ==================== 路由定义 ====================

@task_blueprint.post(
    "/",
    summary="创建导出任务",
    description="创建新的导出任务，支持批量导出数据集或数据项"
)
@jwt_required
def create_export_task(body: ExportTaskCreateForm):
    """创建导出任务"""
    # 获取当前用户ID
    user_id = get_current_user_id()
    if user_id is None:
        return error('用户未登录', 401)

    # 转换为服务层期望的格式
    from app.models.form.dataset_form import ExportTaskCreateForm as ServiceExportTaskCreateForm
    
    # 处理options嵌套对象
    options_dict = None
    if body.options:
        options_dict = body.options.model_dump()
    
    form = ServiceExportTaskCreateForm(
        type=body.type,
        target_id=body.targetId,
        target_ids=body.targetIds or [],
        options=options_dict or {}
    )

    # 创建任务
    try:
        task_vo = TaskService.create_export_task(form, user_id)
        return success(task_vo, '创建成功')
    except Exception as e:
        return error(str(e), 400)


@task_blueprint.get(
    "/<task_id>",
    summary="查询导出任务状态",
    description="根据任务ID查询任务执行状态和进度"
)
@jwt_required
def get_task_status(path: TaskIdPath):
    """查询导出任务状态"""
    if not path.task_id:
        return error('任务ID不能为空', 400)

    task_vo = TaskService.get_task_status(path.task_id)
    if task_vo is None:
        return error('任务不存在', 404)

    return success(task_vo)


@task_blueprint.get(
    "/<task_id>/download",
    summary="下载导出文件",
    description="下载已完成的导出任务文件（302重定向到文件存储）"
)
@jwt_required
def download_export_file(path: TaskIdPath):
    """下载导出文件"""
    if not path.task_id:
        return error('任务ID不能为空', 400)

    download_url = TaskService.download_export_file(path.task_id)
    if download_url is None:
        return error('任务未完成、已过期或下载链接不存在', 400)

    return redirect(download_url, code=302)


@task_blueprint.delete(
    "/<task_id>",
    summary="取消导出任务",
    description="取消正在执行或等待中的导出任务"
)
@jwt_required
def cancel_task(path: TaskIdPath):
    """取消导出任务"""
    if not path.task_id:
        return error('任务ID不能为空', 400)

    try:
        TaskService.cancel_task(path.task_id)
        return '', 204
    except Exception as e:
        return error(str(e), 400)
