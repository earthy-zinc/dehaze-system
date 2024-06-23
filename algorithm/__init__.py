import math

import torch
from thop import profile

from global_variable import DEVICE


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class ModelInfo:
    def __init__(self, name, model, flops, params):
        self.name = name
        self.model = model
        self.flops = flops
        self.params = params
        self.init_time = 0


class ModelContainer:

    def __init__(self):
        self.model_infos = []
        self.test_input = torch.randn(1, 3, 256, 256).to(DEVICE)

    def is_model_registered(self, model_name):
        for model_info in self.model_infos:
            if model_info.name == model_name:
                return True

    def register_model(self, model_name, model):
        # 模型已存在，则从现有中返回模型信息
        for _model in self.model_infos:
            if _model.name == model_name:
                return _model

        net_g_flops, net_g_params = profile(model, inputs=(self.test_input,))
        model_info = ModelInfo(model_name, model, convert_size(net_g_flops), convert_size(net_g_params))
        self.model_infos.append(model_info)
        return model_info
