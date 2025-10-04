import torch

# sys.path.append("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP")
# sys.path.append("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP")

from basicsr.archs.itb_arch import FusionRefine

print("ok")
pretrained_ridcp_path = "/mnt/e/DeepLearningCopies/2023/RIDCP/pretrained_models/pretrained_RIDCP_renamed.pth"
param_key = 'params'
load_ridcp = torch.load(pretrained_ridcp_path)[param_key]
print("load ridcp ok")
prefix = 'feature_extract.'
# 使用字典推导式为每个键添加前缀
load_net = {prefix + key: value for key, value in load_ridcp.items()}
print("添加前缀 ok")
pretrained_rcan_path = "/mnt/e/DeepLearningCopies/2023/RIDCP/pretrained_models/rcan.pkl"
load_rcan = torch.load(pretrained_rcan_path)
for key, value in load_rcan.items():
    if key.startswith("pre_trained_rcan"):
        load_net[key] = value

save_dict = {param_key: load_net}

saved_new_net_path = "/mnt/e/DeepLearningCopies/2023/RIDCP/pretrained_models/pretrained_ITB_renamed.pth"
torch.save(save_dict, saved_new_net_path)
print("saved_new_net ok")
itb = FusionRefine(opt={"LQ_stage": True})
itb.load_state_dict(load_net, strict=False)
print("check ok")
