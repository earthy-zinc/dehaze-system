"""
数据集管理路由 - 使用 flask-openapi3 自动生成 Swagger 文档
"""
from flask_openapi3 import APIBlueprint, Tag

from app.models import DatasetAddForm as FormDatasetAddForm, DatasetQuery as FormDatasetQuery
from app.models import DatasetItemCreateForm as FormDatasetItemCreateForm, DatasetItemUpdateForm as FormDatasetItemUpdateForm
from app.models.schema.dataset import (
    DatasetQuery,
    DatasetIdPath,
    DatasetIdsQuery,
    DatasetImagePageQuery,
    DatasetAddForm,
    DatasetUpdateForm,
    DatasetItemCreateForm,
    DatasetItemUpdateForm,
    DatasetItemDeleteForm,
)
from app.service.dataset_service import DatasetService, DatasetItemService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error


# 定义 Tag
dataset_tag = Tag(name="数据集管理", description="数据集相关接口")
dataset_item_tag = Tag(name="数据集项管理", description="数据集项相关接口")

# 创建 APIBlueprint（自动携带安全配置）
dataset_blueprint = APIBlueprint(
    "dataset",
    __name__,
    url_prefix="/api/v1/datasets",
    abp_tags=[dataset_tag],
    abp_security=[{"BearerAuth": []}]
)


# ==================== 数据集接口 ====================

@dataset_blueprint.get(
    "/",
    summary="获取数据集列表",
    description="获取数据集列表（树形结构）"
)
@jwt_required
def list_datasets(query: DatasetQuery):
    """获取数据集列表"""
    # 构建查询条件
    form_query = None
    if query.keywords:
        form_query = FormDatasetQuery(keyword=query.keywords)

    dataset_list = DatasetService.get_dataset_tree(form_query)
    return success(dataset_list)


@dataset_blueprint.get(
    "/options",
    summary="获取数据集下拉列表",
    description="获取数据集下拉选项列表"
)
@jwt_required
def list_dataset_options():
    """数据集下拉列表"""
    options = DatasetService.get_dataset_options()
    return success(options)


@dataset_blueprint.get(
    "/<int:dataset_id>",
    summary="获取数据集信息",
    description="根据ID获取数据集详情"
)
@jwt_required
def get_dataset_info(path: DatasetIdPath):
    """获取数据集信息"""
    dataset = DatasetService.get_dataset_by_id(path.dataset_id)

    if dataset is None:
        return error('数据集不存在', 404)

    return success(dataset)


@dataset_blueprint.post(
    "/",
    summary="新增数据集",
    description="创建新的数据集"
)
@jwt_required
def add_dataset(body: DatasetAddForm):
    """新增数据集"""
    try:
        # 验证必填字段
        if not body.name:
            return error("数据集名称不能为空", 400)

        # 转换为 Form 对象
        form = FormDatasetAddForm(
            parent_id=body.parentId,
            name=body.name,
            type=body.type or '',
            description=body.description or '',
            path=body.path or '',
            status=body.status
        )

        result = DatasetService.create_dataset(form)
        return success(result.to_dict() if result else None, '保存成功')
    except Exception as e:
        return error(f"创建数据集失败: {str(e)}", 500)


@dataset_blueprint.put(
    "/<int:dataset_id>",
    summary="修改数据集",
    description="根据ID修改数据集信息"
)
@jwt_required
def update_dataset(path: DatasetIdPath, body: DatasetUpdateForm):
    """修改数据集"""
    # 转换为字典供 service 使用
    data = {}
    if body.parentId is not None:
        data['parentId'] = body.parentId
    if body.name is not None:
        data['name'] = body.name
    if body.type is not None:
        data['type'] = body.type
    if body.description is not None:
        data['description'] = body.description
    if body.path is not None:
        data['path'] = body.path
    if body.status is not None:
        data['status'] = body.status

    result = DatasetService.update_dataset(path.dataset_id, data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '保存成功')


@dataset_blueprint.delete(
    "/",
    summary="删除数据集",
    description="批量删除数据集，多个ID以英文逗号分隔"
)
@jwt_required
def delete_datasets(query: DatasetIdsQuery):
    """删除数据集"""
    if not query.ids:
        return error('参数错误', 400)

    try:
        dataset_ids = [int(id_str) for id_str in query.ids.split(',')]
    except ValueError:
        return error('参数格式错误', 400)

    result = DatasetService.delete_datasets(dataset_ids)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '删除成功')


@dataset_blueprint.get(
    "/<int:dataset_id>/images",
    summary="获取数据集图片项",
    description="分页获取数据集下的图片项"
)
@jwt_required
def get_dataset_images(path: DatasetIdPath, query: DatasetImagePageQuery):
    """获取数据集图片项"""
    result = DatasetService.get_image_items(path.dataset_id, query.pageNum, query.pageSize)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'])


# ==================== 数据集项接口 ====================

@dataset_blueprint.post(
    "/item",
    tags=[dataset_item_tag],
    summary="新增数据项",
    description="在指定数据集下创建新的数据项"
)
@jwt_required
def add_dataset_item(body: DatasetItemCreateForm):
    """新增数据项"""
    if not body.datasetId:
        return error('缺少参数datasetId', 400)

    result = DatasetItemService.create_dataset_item(body.datasetId, body.name)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '创建成功')


@dataset_blueprint.put(
    "/item",
    tags=[dataset_item_tag],
    summary="修改数据项",
    description="根据数据项ID修改数据项信息"
)
@jwt_required
def update_dataset_item(body: DatasetItemUpdateForm):
    """修改数据项"""
    if not body.id:
        return error('缺少参数id', 400)

    if not body.name:
        return error('缺少参数name', 400)

    result = DatasetItemService.update_dataset_item(body.id, body.name)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '更新成功')


@dataset_blueprint.delete(
    "/item",
    tags=[dataset_item_tag],
    summary="删除数据项",
    description="根据数据项ID删除数据项"
)
@jwt_required
def delete_dataset_item(body: DatasetItemDeleteForm):
    """删除数据项"""
    if not body.datasetItemId:
        return error('缺少参数datasetItemId', 400)

    result = DatasetItemService.delete_dataset_item(body.datasetItemId)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '删除成功')
