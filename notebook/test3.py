import torch

from basicsr.archs.ridcp_new_arch import RIDCPNew


load_hqp = torch.load("E://DeepLearningCopies//2023//RIDCP//pretrained_models//pretrained_RIDCP_New.pth")
param_key = 'params'
load_hqp = load_hqp[param_key]

new_hqp = {}
for key, value in load_hqp.items():
    if key.startswith('multiscale_encoder.blocks.2.swin_blks.'):
        new_key = key.replace('multiscale_encoder.blocks.2.swin_blks.', 'ridcp_encoder.swin_blks.', 1)
    elif key.startswith('multiscale_encoder.'):
        new_key = key.replace('multiscale_encoder.', 'vq_encoder.', 1)
    elif key.startswith('decoder_group.'):
        new_key = key.replace('decoder_group.', 'vq_decoder_group.', 1)
    elif key.startswith('multiscale_decoder.'):
        new_key = key.replace('multiscale_decoder.', 'ridcp_decoder.', 1)
    else:
        new_key = key
    new_hqp[new_key] = value

hqp = RIDCPNew(LQ_stage=True)
hqp_state = hqp.state_dict()
if set(hqp_state.keys()) == set(new_hqp.keys()):
    hqp_state.update(new_hqp)
    hqp.load_state_dict(hqp_state)
    save_dict = {param_key: hqp.state_dict()}
    torch.save(save_dict, "E://DeepLearningCopies//2023//RIDCP//pretrained_models//pretrained_RIDCP_renamed.pth")
else:
    print(set(hqp_state.keys()) - set(new_hqp.keys()))
    print("-------------------------------")
    print(set(new_hqp.keys()) - set(hqp_state.keys()))

# ridcp = RIDCPNew(LQ_stage=True)
# for key in ridcp.state_dict().keys():
#     print(key)
