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
    'description': '''
    1. 从请求参数中获取模型ID和输入图片路径。
    2. 根据模型ID从数据库中查询模型信息。
    3. 使用`import_module`动态导入模型的算法模块。
    4. 对每个输入图片路径进行处理：
        - 生成输入图片对象。
        - 生成预测图片对象。
        - 调用模型的`dehaze`函数进行去雾处理。
        - 将去雾结果添加到结果列表中。
    5. 返回去雾预测结果。
    ''',
    'parameters': [
        {
            'name': 'modelId',
            'in': 'query',
            'type': 'integer',
            'required': True,
            'description': '模型id，用于查询数据库中的模型信息。'
        },
        {
            'name': 'input',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': '输入图片路径，可以传入多个图片路径。'
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
    'description': '''
    1. 从请求参数中获取预测图片路径和真实图片路径。
    2. 默认的评估算法`calculate`。
    3. 返回评估结果。
    ''',
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

    if request.args.get("flag"):
        from algorithm.WPXNet.calculate import calculate as calculate_wpxnet
        return calculate_wpxnet(pred, gt)

    result = calculate(pred, gt)
    return success(result)
