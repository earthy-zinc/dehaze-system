import sys

import torch

sys.path.append("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP")

from basicsr.archs.itb_arch import FusionRefine

itb = FusionRefine(opt={"LQ_stage": True})

for key, param in itb.state_dict().items():
    print(key)


pretrained_new_hqp_path = "../pretrained_models/pretrained_HQPs_renamed.pth"
param_key = 'params'
load_hqp = torch.load(pretrained_new_hqp_path)[param_key]

prefix = 'feature_extract.'
# 使用字典推导式为每个键添加前缀
load_net = {prefix + key: value for key, value in load_hqp.items()}
save_dict = {param_key: load_net}
torch.save(save_dict, "../pretrained_models/ITB_init_weight_renamed.pth")

itb.load_state_dict(load_net, strict=False)
