import torch
from basicsr.archs.AOD_arch import AOD
from basicsr.archs.FFA_arch import FFA
path = '/mnt/e/DeepLearningCopies/2023/RIDCP/pretrained_models/compare/FFA/its_train_ffa_3_19.pk'
new_path = '/mnt/e/DeepLearningCopies/2023/RIDCP/pretrained_models/compare/FFA/its.pth'
param_key = 'params'
load_hqp = torch.load(path)['model']
# ffa = FFA()
# ffa.load_state_dict(load_hqp)

save_dict = {param_key: load_hqp}
torch.save(save_dict, new_path)
