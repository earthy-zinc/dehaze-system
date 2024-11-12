import traceback
from importlib import import_module

from flasgger import swag_from
from flask import request
from common import success, generate_predict_img, generate_input_img, error
from base import base as app
from model import SysAlgorithm
from algorithm.metrics import calculate


@app.route('/model/prediction', methods=['GET'])
@swag_from({
    'tags': ['10.模型接口'],
    'summary': '模型预测',
    'parameters': [
        {
            'name': 'modelId',
            'in': 'query',
            'type': 'integer',
            'required': True,
            'description': '模型id'
        },
        {
            'name': 'input',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': '输入图片路径'
        }
    ],
    'responses': {
        '200': {
            'description': '模型预测结果',
        }
    }
})
def predict():
    id = request.args.get("modelId")
    inputs: list = request.args.getlist("input")
    algorithm = SysAlgorithm.query.get(id)
    assert algorithm is not None, "模型不存在"

    try:
        model = import_module(algorithm.import_path)
        result = []
        for input in inputs:
            input_img = generate_input_img(input, id)
            pred_img = generate_predict_img(id)
            model.dehaze(input_img.get("path"), pred_img.get("path"), algorithm.path)
            result.append({
                "model": algorithm.name,
                "input": input_img,
                "output": pred_img
            })
        return success(result)
    except ImportError as e:
        traceback.print_exc()
        return error("无法从" + algorithm.import_path + "中导入算法代码，错误信息：" + e.msg)
    except AttributeError as e:
        traceback.print_exc()
        return error("模块未定义" + algorithm.import_path + ".dehaze函数，错误信息：不存在属性 " + e.name)


@app.route('/model/evaluation', methods=['GET'])
@swag_from({
    'tags': ['10.模型接口'],
    'summary': '模型评估',
    'parameters': [
        {
            'name': 'input',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': '预测图片路径'
        },
        {
            'name': 'output',
            'in': 'query',
            'type': 'string',
            'required': False,
            'description': '真实图片路径'
        }
    ],
    'responses': {
        '200': {
            'description': '模型评估结果',
        }
    }
})
def evaluate():
    pred = request.args.get("input")
    gt = request.args.get("output")
    result = calculate(pred, gt)
    return success(result)
