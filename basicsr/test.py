import sys

sys.path.append("/mnt/workspace/ridcp")
sys.path.append("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("E://DeepLearningCopies//2023//RIDCP")
sys.path.append("/mnt/e/DeepLearningCopies/2023/RIDCP")
sys.path.append("/home/zhou/wpx/RIDCP")

import logging
import torch
from os import path as osp
from basicsr.archs import build_network
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    # logger.info(dict2str(opt))
    # create test dataset and dataloader
    test_loader = None
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'test':
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'测试集 {dataset_opt["name"]} 的图片数量: {len(test_set)}')

    # create model
    model = build_model(opt)
    if opt.get('val') is not None and test_loader is not None:
        test_net_g_opt = opt['network_g'].copy()
        if test_net_g_opt.get('type') == 'FusionRefine':
            test_net_g_opt['opt']["use_weight"] = None
            test_net_g_opt['opt']["weight_alpha"] = -21.25
        test_net_g = build_network(test_net_g_opt)
        model.model_to_device(test_net_g)
        # opt['name'] = ''
        latest_net_g_path = osp.join('experiments', opt['name'], 'models/net_g_latest.pth')
        best_net_g_path = osp.join('experiments', opt['name'], 'models/net_g_best_.pth')
        if osp.isfile(best_net_g_path):
            model.load_network(test_net_g, best_net_g_path, False)
            model.nondist_test(test_net_g, test_loader, 999999, None, opt['val']['save_img'])
        if osp.isfile(latest_net_g_path):
            model.load_network(test_net_g, latest_net_g_path, False)
            model.nondist_test(test_net_g, test_loader, opt['train']['total_iter'], None, opt['val']['save_img'])
        if opt.get("no_need_load") is not None:
            model.validation(test_loader, 0, None, opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
