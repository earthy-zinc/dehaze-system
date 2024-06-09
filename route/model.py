from importlib import import_module

from flask import request
from common import success, generate_predict_img
from base import base as app
from model.SysAlgorithm import SysAlgorithm


@app.route('/model/prediction', methods=['GET'])
def predict():
    id = request.get("modelId")
    inputs: list = request.get("input")
    
    algorithm = SysAlgorithm.query.get(id)
    assert algorithm is not None, "模型不存在"
    
    try:
        model = import_module(algorithm.import_path)
        for input in inputs:
            pred_img = generate_predict_img(id)
            model.dehaze(input, pred_img.path, algorithm)
            
    except ImportError as e:
        print("无法导入" + "comon")
    except AttributeError as e:
        print("common" + "模块未定义ok函数")
    
    return success()


@app.route('/model/evaluation', methods=['GET'])
def evaluate():
    return success()