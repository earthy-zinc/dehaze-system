import os
import time
import traceback
from importlib import import_module
from io import BytesIO
from uuid import uuid4
from flask_jwt_extended import get_jwt, jwt_required, verify_jwt_in_request
import torch
from flasgger import swag_from
from flask import Blueprint, request, current_app

from app.extensions import mysql
from app.models import SysAlgorithm, SysFile, SysPredLog, SysEvalLog
from app.service.file import read_file_from_url, upload_file
from app.service.model import get_flag
from app.utils.file import convert_size
from app.utils.metrics import calculate
from app.utils.result import success, error

model_blueprint = Blueprint("model", __name__, url_prefix="/model")


@jwt_required
@model_blueprint.route('/prediction', methods=['POST'])
@swag_from({
    "tags": ["10.模型接口"],
    "summary": "模型预测",
    "description": "基于指定模型对输入图像进行去雾处理，返回预测结果。",
    "consumes": ["application/json"],
    "produces": ["application/json"],
    "parameters": [
        {
            "in": "body",
            "name": "body",
            "required": True,
            "schema": {
                "type": "object",
                "required": ["modelId", "url"],
                "properties": {
                    "modelId": {
                        "type": "integer",
                        "description": "指定的模型 ID"
                    },
                    "url": {
                        "type": "string",
                        "format": "url",
                        "description": "输入图像的 URL"
                    }
                }
            }
        }
    ],
    "responses": {
        "200": {
            "description": "预测成功，返回预测图像的 URL。",
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "example": "00000"},
                    "msg": {"type": "string", "example": "success"},
                    "data": {"type": "string", "format": "url", "description": "预测图像的 URL"}
                }
            }
        },
        "400": {
            "description": "请求参数错误。",
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "example": "B00001"},
                    "msg": {"type": "string", "example": "Invalid input data"}
                }
            }
        },
        "500": {
            "description": "服务器内部错误。",
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "example": "B00001"},
                    "msg": {"type": "string", "example": "Internal Server Error"}
                }
            }
        }
    }
})
def predict():
    """模型预测"""
    data = request.get_json()
    id = int(data.get("modelId"))
    url: str = data.get("url")

    # 参数校验
    verify_jwt_in_request()
    algorithm = SysAlgorithm.query.get(id)
    if not algorithm:
        return error("模型不存在")

    # 读取上传文件
    input_img, input_file_info = read_file_from_url(url, flag=get_flag(algorithm))

    # 检查SysPredLog中是否有预测记录
    sys_pred_log = SysPredLog.query.filter_by(algorithm_id=id, origin_md5=input_file_info.md5).first()
    if sys_pred_log:
        return success({
            "predUrl": sys_pred_log.pred_url,
            "hazeUrl": sys_pred_log.origin_url
        })

    # 不存在则调用模型进行预测
    try:
        start = time.time()
        # 动态加载算法模块
        try:
            model = import_module(algorithm.import_path)
            if not hasattr(model, "dehaze"):
                return error(f"算法模块 {algorithm.import_path} 缺少 dehaze 函数", 404)
        except Exception as e:
            traceback.print_exc()
            return error(f"加载算法模块失败：{str(e)}", 404)

        model_path = os.path.join(current_app.config.get("MODEL_PATH", ""), algorithm.path)
        pred_img: BytesIO = model.dehaze(input_img, model_path)

        end = time.time()

        # 释放显存
        torch.cuda.empty_cache()

        # 保存预测图像
        pred_img_info: SysFile = upload_file("pred_" + uuid4().hex +".png", "image/png", pred_img)

        # 记录预测日志
        sys_pred_log = SysPredLog(
            algorithm_id=id,
            origin_file_id=input_file_info.id,
            origin_md5=input_file_info.md5,
            origin_url=input_file_info.url,
            pred_file_id=pred_img_info.id,
            pred_md5=pred_img_info.md5,
            pred_url=pred_img_info.url,
            time=end - start,
            create_by=get_jwt().get("userId", None),
            update_by=get_jwt().get("userId", None)
        )
        mysql.session.add(sys_pred_log)
        mysql.session.commit()

        return success({
            "predUrl": pred_img_info.url,
            "hazeUrl": input_file_info.url
        })

    except Exception as e:
        traceback.print_exc()
        return error(f"模型预测失败：{str(e)}")

@jwt_required
@model_blueprint.route('/evaluation', methods=['POST'])
@swag_from({
    "tags": ["10.模型接口"],
    "summary": "模型评估",
    "description": "根据预测图像和参考图像，计算图像质量指标。",
    "consumes": ["application/json"],
    "produces": ["application/json"],
    "parameters": [
        {
            "in": "body",
            "name": "body",
            "required": True,
            "schema": {
                "type": "object",
                "required": ["modelId", "predUrl", "gtUrl"],
                "properties": {
                    "modelId": {
                        "type": "integer",
                        "description": "指定的模型 ID"
                    },
                    "predUrl": {
                        "type": "string",
                        "format": "url",
                        "description": "预测图像的 URL"
                    },
                    "gtUrl": {
                        "type": "string",
                        "format": "url",
                        "description": "参考图像的 URL"
                    }
                }
            }
        }
    ],
    "responses": {
        "200": {
            "description": "评估成功，返回计算的质量指标。",
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "example": "00000"},
                    "msg": {"type": "string", "example": "success"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer", "description": "指标 ID"},
                                "label": {"type": "string", "description": "指标名称"},
                                "value": {"type": "number", "description": "计算的指标值"},
                                "better": {"type": "string", "description": "该指标的优化方向，'higher' 或 'lower'"},
                                "description": {"type": "string", "description": "指标的详细描述"}
                            }
                        }
                    }
                }
            }
        },
        "400": {
            "description": "请求参数错误。",
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "example": "B00001"},
                    "msg": {"type": "string", "example": "Invalid input data"}
                }
            }
        },
        "500": {
            "description": "服务器内部错误。",
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "example": "B00001"},
                    "msg": {"type": "string", "example": "Internal Server Error"}
                }
            }
        }
    }
})
def evaluate():
    """模型评估"""
    # 获取请求中的模型ID和图像的URL
    data = request.get_json()
    id: int = int(data.get("modelId"))
    pred: str = data.get("predUrl")
    gt: str = data.get("gtUrl")

    # 验证参数
    verify_jwt_in_request()
    algorithm = SysAlgorithm.query.get(id)
    if not algorithm:
        return error("模型不存在")
    flag = get_flag(algorithm)

    # 读取上传文件
    pred_bytes, pred_info = read_file_from_url(pred)
    gt_bytes, gt_info = read_file_from_url(gt, flag)

    # 检查SysEvalLog中是否有评估记录
    sys_eval_log = SysEvalLog.query.filter_by(pred_md5=pred_info.md5, gt_md5=gt_info.md5).first()
    if sys_eval_log:
        return success(sys_eval_log.result)

    # 评估预测效果
    try:
        start = time.time()
        result = calculate(pred_bytes, gt_bytes)
        end = time.time()
        # 释放显存
        torch.cuda.empty_cache()
        # 记录评估日志
        sys_eval_log = SysEvalLog(
            algorithm_id=id,
            pred_file_id=pred_info.id,
            pred_md5=pred_info.md5,
            pred_url=pred_info.url,
            gt_file_id=gt_info.id,
            gt_md5=gt_info.md5,
            gt_url=gt_info.url,
            time=end - start,
            result=result,
            create_by=get_jwt().get("userId", None),
            update_by=get_jwt().get("userId", None),
        )
        mysql.session.add(sys_eval_log)
        mysql.session.commit()

        return success(result)
    except Exception as e:
        traceback.print_exc()
        return error(f"模型评估失败：{str(e)}")

@model_blueprint.route("/init", methods=["GET"])
@swag_from({
    "tags": ["10.模型接口"],
    "summary": "模型初始化",
    "description": "计算每个模型的参数量和flops，存到数据库中。",
})
def init_algorithm():
    import torch
    from thop import profile
    test_input = torch.randn(1, 3, 256, 256).cuda()
    algorithms: list[SysAlgorithm] = SysAlgorithm.query.all()
    model_path = current_app.config.get("MODEL_PATH")
    for algorithm in algorithms:
        flops = algorithm.flops
        params = algorithm.params
        if flops is not None and params is not None:
            continue
        try:
            run = import_module(algorithm.import_path)
            model = run.get_model(os.path.join(model_path, algorithm.path))
            net_flops, net_params = profile(model, inputs=(test_input,))
            algorithm.flops = convert_size(net_flops)
            algorithm.params = convert_size(net_params)
            mysql.session.commit()
            del model
            torch.cuda.empty_cache()
        except Exception:
            print("模型" + algorithm.name + "计算参数及浮点数失败")
            traceback.print_exc()
            continue
    return success("模型初始化成功")


@jwt_required
@model_blueprint.route("/jwt", methods=["GET"])
@swag_from({
    "tags": ["10.模型接口"],
    "summary": "测试接口",
    "description": "测试接口。",
    "consumes": ["application/json"],
    "produces": ["application/json"],
})
def test():
    verify_jwt_in_request()
    return success(get_jwt().get("userId"))
