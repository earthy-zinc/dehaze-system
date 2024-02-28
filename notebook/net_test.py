from copy import deepcopy
import torch
import platform

from basicsr import get_root_logger
from collections import OrderedDict
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.options import ordered_yaml
import yaml
from basicsr import build_network
from basicsr.archs.ridcp_new_arch import RIDCPNew

def print_different_keys_loading(crt_net, load_net, strict=True):
    """Print keys with different name or different size when loading models.

    1. Print keys with different names.
    2. If strict=False, print the same key but with different tensor size.
        It also ignore these keys with different sizes (not load).

    Args:
        crt_net (torch model): Current network.
        load_net (dict): Loaded network.
        strict (bool): Whether strictly loaded. Default: True.
    """
    crt_net = crt_net.state_dict()
    crt_net_keys = set(crt_net.keys())
    load_net_keys = set(load_net.keys())

    logger = get_root_logger()
    if crt_net_keys != load_net_keys:
        logger.warning('当前神经网络 相比 已加载的神经网络 多出的网络层:')
        for v in sorted(list(crt_net_keys - load_net_keys)):
            logger.warning(f'  {v}')
        logger.warning('已加载的神经网络 相比 当前神经网络 多出的网络参数层:')
        for v in sorted(list(load_net_keys - crt_net_keys)):
            logger.warning(f'  {v}')

    # check the size for the same keys
    if not strict:
        common_keys = crt_net_keys & load_net_keys
        for k in common_keys:
            if crt_net[k].size() != load_net[k].size():
                logger.warning(f'网络参数层[{k}]形状不同，以忽略: 当前网络参数层形状: '
                               f'{crt_net[k].shape}; 已加载网络参数层形状: {load_net[k].shape}')
                load_net[k + '.ignore'] = load_net.pop(k)

def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    logger = get_root_logger()
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            logger.info('加载中: params_ema（指数滑动平均）不存在，将使用 params')
        load_net = load_net[param_key]
    logger.info(f'从 {load_path} 加载模型 {net.__class__.__name__} 网络参数层为: [{param_key}].')
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    print_different_keys_loading(net, load_net, strict)
    net.load_state_dict(load_net, strict=strict)



pretrained_new_hqp_path = "../pretrained_models/pretrained_HQPs_renamed.pth"
pretrained_new_net_path = "../pretrained_models/pretrained_RIDCP_renamed.pth"
param_key = 'params'

with open("../options/RIDCPNew-pei-NH-HAZE.yml", mode='r') as f:
    opt = yaml.load(f, Loader=ordered_yaml()[0])
hq_opt = opt['network_g'].copy()
hq_opt['LQ_stage'] = False

print(opt["network_g"])
net = build_network(opt['network_g'])
hqp = build_network(hq_opt)

# def func(*, LQ_stage, **kwargs):
#     print(LQ_stage, kwargs)
# func(**hq_opt)
# opt_hqp = {
#     "LQ_stage": False,
#     "frozen_module_keywords": ['quantizer', 'decoder_group', 'after_quant', 'out_conv']
# }
# opt_net = {
#     #"LQ_stage": True,
#     "frozen_module_keywords": ['quantizer', 'decoder_group', 'after_quant', 'out_conv']
# }
# opts = OrderedDict()
# opts["LQ_stage"] = True
# opts["frozen_module_keywords"] = ['quantizer', 'decoder_group', 'after_quant', 'out_conv']
# print(opts)
# net = ARCH_REGISTRY.get(hq_opt_type)(**hq_opt)
# for key, param in net.state_dict().items():
#     print(key)


# hqp = ARCH_REGISTRY.get("RIDCPNew")(**opt_hqp)
load_network(net, pretrained_new_net_path, False, param_key)
load_network(hqp, pretrained_new_hqp_path, False, param_key)
