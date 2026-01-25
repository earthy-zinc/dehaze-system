"""
字典管理路由 - 使用 flask-openapi3 自动生成 Swagger 文档
"""
from flask_openapi3 import APIBlueprint, Tag

from app.models.schema.dict import (
    DictPageQuery,
    DictTypePageQuery,
    DictIdPath,
    DictIdsPath,
    DictTypeIdPath,
    DictTypeIdsPath,
    DictTypeCodePath,
    DictForm,
    DictTypeForm,
)
from app.service.dict_service import DictService, DictTypeService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error


# 定义 Tag
dict_tag = Tag(name="字典管理", description="字典项相关接口")
dict_type_tag = Tag(name="字典类型管理", description="字典类型相关接口")

# 创建 APIBlueprint（自动携带安全配置）
dict_blueprint = APIBlueprint(
    "dict",
    __name__,
    url_prefix="/api/v1/dict",
    abp_tags=[dict_tag],
    abp_security=[{"BearerAuth": []}]
)


# ==================== 字典项接口 ====================

@dict_blueprint.get(
    "/page",
    summary="字典分页列表",
    description="获取字典分页列表"
)
@jwt_required
def get_dict_page(query: DictPageQuery):
    """字典分页列表"""
    dict_items, total = DictService.get_dict_page(
        query.pageNum, query.pageSize, query.keywords, query.typeCode
    )

    dict_list = []
    for item in dict_items:
        dict_list.append({
            'id': item.id,
            'typeCode': item.type_code,
            'name': item.name,
            'value': item.value,
            'status': item.status,
            'sort': item.sort
        })

    return success({
        'list': dict_list,
        'total': total,
        'pageNum': query.pageNum,
        'pageSize': query.pageSize
    })


@dict_blueprint.get(
    "/<int:dict_id>/form",
    summary="字典表单数据",
    description="获取字典表单数据"
)
@jwt_required
def get_dict_form(path: DictIdPath):
    """获取字典表单数据"""
    dict_data = DictService.get_dict_form(path.dict_id)

    if not dict_data:
        return error('字典不存在', 404)

    return success(dict_data)


@dict_blueprint.post(
    "/",
    summary="新增字典",
    description="新增字典项"
)
@jwt_required
def create_dict(body: DictForm):
    """新增字典"""
    data = body.model_dump(exclude_none=True)
    # 转换字段名 typeCode -> type_code
    if 'typeCode' in data:
        data['type_code'] = data.pop('typeCode')

    result = DictService.create_dict(data)

    if result:
        return success(None, '新增成功')
    else:
        return error('新增失败', 400)


@dict_blueprint.put(
    "/<int:dict_id>",
    summary="修改字典",
    description="修改字典项"
)
@jwt_required
def update_dict(path: DictIdPath, body: DictForm):
    """修改字典"""
    data = body.model_dump(exclude_none=True)
    # 转换字段名 typeCode -> type_code
    if 'typeCode' in data:
        data['type_code'] = data.pop('typeCode')

    result = DictService.update_dict(path.dict_id, data)

    if result:
        return success(None, '修改成功')
    else:
        return error('修改失败', 400)


@dict_blueprint.delete(
    "/<dict_ids>",
    summary="删除字典",
    description="删除字典项，多个ID以英文逗号分隔"
)
@jwt_required
def delete_dict(path: DictIdsPath):
    """删除字典"""
    try:
        id_list = [int(id) for id in path.dict_ids.split(',')]
        result = DictService.delete_dict(id_list)

        if result:
            return success(None, '删除成功')
        else:
            return error('删除失败', 400)
    except Exception as e:
        return error('参数错误', 400)


@dict_blueprint.get(
    "/<type_code>/options",
    summary="字典下拉列表",
    description="获取字典下拉列表",
    security=[]  # 无需认证
)
def list_dict_options(path: DictTypeCodePath):
    """字典下拉列表"""
    options = DictService.list_dict_options(path.type_code)
    return success(options)


# ==================== 字典类型接口 ====================

@dict_blueprint.get(
    "/types/page",
    tags=[dict_type_tag],
    summary="字典类型分页列表",
    description="获取字典类型分页列表"
)
@jwt_required
def get_dict_type_page(query: DictTypePageQuery):
    """字典类型分页列表"""
    dict_types, total = DictTypeService.get_dict_type_page(
        query.pageNum, query.pageSize, query.keywords
    )

    type_list = []
    for item in dict_types:
        type_list.append({
            'id': item.id,
            'name': item.name,
            'code': item.code,
            'status': item.status,
            'remark': item.remark
        })

    return success({
        'list': type_list,
        'total': total,
        'pageNum': query.pageNum,
        'pageSize': query.pageSize
    })


@dict_blueprint.get(
    "/types/<int:type_id>/form",
    tags=[dict_type_tag],
    summary="字典类型表单数据",
    description="获取字典类型表单数据"
)
@jwt_required
def get_dict_type_form(path: DictTypeIdPath):
    """获取字典类型表单数据"""
    dict_type_data = DictTypeService.get_dict_type_form(path.type_id)

    if not dict_type_data:
        return error('字典类型不存在', 404)

    return success(dict_type_data)


@dict_blueprint.post(
    "/types",
    tags=[dict_type_tag],
    summary="新增字典类型",
    description="新增字典类型"
)
@jwt_required
def create_dict_type(body: DictTypeForm):
    """新增字典类型"""
    data = body.model_dump(exclude_none=True)
    result = DictTypeService.create_dict_type(data)

    if result:
        return success(None, '新增成功')
    else:
        return error('新增失败', 400)


@dict_blueprint.put(
    "/types/<int:type_id>",
    tags=[dict_type_tag],
    summary="修改字典类型",
    description="修改字典类型"
)
@jwt_required
def update_dict_type(path: DictTypeIdPath, body: DictTypeForm):
    """修改字典类型"""
    data = body.model_dump(exclude_none=True)
    result = DictTypeService.update_dict_type(path.type_id, data)

    if result:
        return success(None, '修改成功')
    else:
        return error('修改失败', 400)


@dict_blueprint.delete(
    "/types/<type_ids>",
    tags=[dict_type_tag],
    summary="删除字典类型",
    description="删除字典类型，多个ID以英文逗号分隔"
)
@jwt_required
def delete_dict_types(path: DictTypeIdsPath):
    """删除字典类型"""
    try:
        id_list = [int(id) for id in path.type_ids.split(',')]
        result = DictTypeService.delete_dict_types(id_list)

        if result:
            return success(None, '删除成功')
        else:
            return error('删除失败', 400)
    except Exception as e:
        return error('参数错误', 400)
