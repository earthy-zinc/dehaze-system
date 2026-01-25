"""
算法管理路由 - 使用 flask-openapi3 自动生成 Swagger 文档
"""
from flask_openapi3 import APIBlueprint, Tag

from app.models.schema.algorithm import (
    AlgorithmQuery,
    AlgorithmIdPath,
    AlgorithmIdsQuery,
    AlgorithmForm,
)
from app.service.algorithm_service import AlgorithmService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error


# 定义 Tag
algorithm_tag = Tag(name="算法管理", description="算法相关接口")

# 创建 APIBlueprint（自动携带安全配置）
algorithm_blueprint = APIBlueprint(
    "algorithm",
    __name__,
    url_prefix="/api/v1/algorithm",
    abp_tags=[algorithm_tag],
    abp_security=[{"BearerAuth": []}]
)


# ==================== 路由定义 ====================

@algorithm_blueprint.get(
    "/",
    summary="获取算法树形表格",
    description="根据关键词查询算法树形列表"
)
@jwt_required
def list_algorithms(query: AlgorithmQuery):
    """获取算法树形表格"""
    algorithms = AlgorithmService.get_algorithm_list(query.keywords)
    return success(algorithms)


@algorithm_blueprint.get(
    "/options",
    summary="获取算法下拉选项列表",
    description="获取所有算法的下拉选项列表"
)
@jwt_required
def get_algorithm_options():
    """获取模型下拉选项列表"""
    options = AlgorithmService.get_algorithm_options()
    return success(options)


@algorithm_blueprint.get(
    "/<int:algorithm_id>",
    summary="获取算法信息",
    description="根据算法ID获取算法详细信息"
)
@jwt_required
def get_algorithm_by_id(path: AlgorithmIdPath):
    """根据ID获取算法信息"""
    algorithm = AlgorithmService.get_algorithm_by_id(path.algorithm_id)
    if algorithm:
        return success(algorithm)
    else:
        return error('算法不存在', 404)


@algorithm_blueprint.post(
    "/",
    summary="新增算法",
    description="创建一个新的算法"
)
@jwt_required
def add_algorithm(body: AlgorithmForm):
    """新增算法"""
    data = body.model_dump(exclude_none=True)
    result = AlgorithmService.create_algorithm(data)
    if result['success']:
        return success(None, result['message'])
    else:
        return error(result['message'], 500)


@algorithm_blueprint.put(
    "/<int:algorithm_id>",
    summary="修改算法",
    description="根据算法ID修改算法信息"
)
@jwt_required
def update_algorithm(path: AlgorithmIdPath, body: AlgorithmForm):
    """修改算法"""
    data = body.model_dump(exclude_none=True)
    result = AlgorithmService.update_algorithm(path.algorithm_id, data)
    if result['success']:
        return success(None, result['message'])
    else:
        if '不存在' in result['message']:
            return error(result['message'], 404)
        else:
            return error(result['message'], 500)


@algorithm_blueprint.delete(
    "/",
    summary="删除算法",
    description="批量删除算法，多个ID以英文逗号分隔"
)
@jwt_required
def delete_algorithms(query: AlgorithmIdsQuery):
    """删除算法"""
    try:
        algorithm_ids = [int(i) for i in query.ids.split(',')]
        result = AlgorithmService.delete_algorithms(algorithm_ids)
        if result['success']:
            return success(None, result['message'])
        else:
            return error(result['message'], 500)
    except ValueError:
        return error('参数格式错误', 400)
