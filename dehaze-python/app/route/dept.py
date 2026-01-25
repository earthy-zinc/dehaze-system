"""
部门管理路由 - 使用 flask-openapi3 自动生成 Swagger 文档
"""
from flask_openapi3 import APIBlueprint, Tag

from app.models.schema.dept import (
    DeptQuery,
    DeptIdPath,
    DeptIdsPath,
    DeptForm,
)
from app.service.dept_service import DeptService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error


# 定义 Tag
dept_tag = Tag(name="部门管理", description="部门相关接口")

# 创建 APIBlueprint（自动携带安全配置）
dept_blueprint = APIBlueprint(
    "dept",
    __name__,
    url_prefix="/api/v1/dept",
    abp_tags=[dept_tag],
    abp_security=[{"BearerAuth": []}]
)


# ==================== 路由定义 ====================

@dept_blueprint.get(
    "/",
    summary="获取部门列表",
    description="根据关键词和状态查询部门列表（树形结构）"
)
@jwt_required
def list_depts(query: DeptQuery):
    """获取部门列表"""
    depts = DeptService.get_dept_list(query.keywords, query.status)
    return success(depts)


@dept_blueprint.get(
    "/options",
    summary="获取部门下拉选项",
    description="获取所有部门的下拉选项列表（树形结构）"
)
@jwt_required
def list_dept_options():
    """获取部门下拉选项"""
    options = DeptService.get_dept_options()
    return success(options)


@dept_blueprint.get(
    "/<int:dept_id>/form",
    summary="获取部门表单数据",
    description="根据部门ID获取部门的表单数据"
)
@jwt_required
def get_dept_form(path: DeptIdPath):
    """获取部门表单数据"""
    dept_form = DeptService.get_dept_form(path.dept_id)

    if not dept_form:
        return error('部门不存在', 404)

    return success(dept_form)


@dept_blueprint.post(
    "/",
    summary="新增部门",
    description="创建一个新的部门"
)
@jwt_required
def add_dept(body: DeptForm):
    """新增部门"""
    data = body.model_dump(exclude_none=True)

    result = DeptService.create_dept(data)

    if result['success']:
        return success(result['data'], result['message'])
    else:
        return error(result['message'], 400)


@dept_blueprint.put(
    "/<int:dept_id>",
    summary="修改部门",
    description="根据部门ID修改部门信息"
)
@jwt_required
def update_dept(path: DeptIdPath, body: DeptForm):
    """修改部门"""
    data = body.model_dump(exclude_none=True)

    result = DeptService.update_dept(path.dept_id, data)

    if result['success']:
        return success(result['data'], result['message'])
    else:
        if '不存在' in result['message']:
            return error(result['message'], 404)
        else:
            return error(result['message'], 400)


@dept_blueprint.delete(
    "/<ids>",
    summary="删除部门",
    description="批量删除部门，多个ID以英文逗号分隔"
)
@jwt_required
def delete_depts(path: DeptIdsPath):
    """删除部门"""
    try:
        dept_ids = [int(i) for i in path.ids.split(',')]
        result = DeptService.delete_depts(dept_ids)

        if result['success']:
            return success(None, result['message'])
        else:
            return error(result['message'], 400)
    except ValueError:
        return error('参数格式错误', 400)
