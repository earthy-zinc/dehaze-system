import datetime
import logging
import re
import sys
import math
import time
import torch
import os
import shutil
from os import path as osp
sys.path.append("/mnt/workspace/ridcp")
sys.path.append("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("E://DeepLearningCopies//2023//RIDCP")
sys.path.append("/mnt/e/DeepLearningCopies/2023/RIDCP")
sys.path.append("/Crack_detection/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("/home/zhou/wpx/RIDCP")
from basicsr.archs import build_network
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_test_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders, test_loader = None, [], None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(f'训练集图片数量: {len(train_set)}; '
                        f'批数量(BatchSize): {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\t每轮(epoch)训练所需迭代量: {num_iter_per_epoch}; '
                        f'总轮数(Total Epoch): {total_epochs}; 迭代次数: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'验证集 {dataset_opt["name"]} 的图片数量: {len(val_set)}')
            val_loaders.append(val_loader)
        elif phase == 'test':
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed']
            )
            logger.info(f'测试集 {dataset_opt["name"]} 的图片数量: {len(test_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, test_loader, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join(opt['root_path'], 'experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            state_path = str(state_path)
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
            else:
                print("未在目录{}中找到可恢复的训练状态".format(state_path))
        else:
            print("当前训练状态文件夹{}不存在，请检查目录".format(state_path))
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # 解析配置文件，进行分布式训练设置，设置随机数种子
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    # cuda的设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # 恢复上次中断的训练状态（如果需要的话）
    resume_state = load_resume_state(opt)

    # 为本次实验创建文件夹
    if resume_state is None:
        make_exp_dirs(opt)

    # 复制yml格式的配置文件到本次实验文件夹的根目录
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.debug(get_env_info())
    logger.debug(dict2str(opt))
    # 初始化 wandb and tensorboard 日志记录器
    tb_logger = init_tb_loggers(opt)

    # 创建训练集、验证集、测试集的数据加载器
    result = create_train_val_test_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, test_loader, total_epochs, total_iters = result
    model_type = opt['model_type']
    # 创建模型
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.warning(f"从第 {resume_state['epoch']} 轮(epoch)开始恢复训练, " f"当前迭代次数: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # 创建一个用于格式化输出的日志记录器 message logger
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 数据预加载器设置
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.debug(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # 开始训练
    logger.info(f'从第 {start_epoch} 轮(epoch)开始训练, 迭代次数: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # 更新学习率
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # 开始本次迭代的训练
            model.feed_data(train_data)
            if model_type == 'GCAModel':
                model.optimize_parameters(current_iter, epoch)
            else:
                model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # 记录日志
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # 保存图片到tensorboard中
            if current_iter % (opt['logger']['show_tf_imgs_freq']) == 0:
                visual_imgs = model.get_current_visuals()
                if tb_logger:
                    for k, v in visual_imgs.items():
                        if k == 'gt':
                            translation = '无雾基准图像'
                        elif k == 'gt_rec':
                            translation = '无雾重建图像'
                        elif k == 'lq':
                            translation = '有雾图像'
                        elif k == 'result_codebook':
                            translation = '码本匹配图像'
                        elif k == 'result_residual':
                            translation = '去雾图像'
                        else:
                            translation = k
                        tb_logger.add_images(f'示例图像/{translation}', v.clamp(0, 1), current_iter)

            # 保存模型和当前训练状态
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('保存当前模型和训练状态')
                model.save(epoch, current_iter)

            if current_iter % opt['logger']['save_latest_freq'] == 0:
                logger.info('保存最新的模型和训练状态')
                model.save(epoch, -1)

            # 开始验证
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('存在多个验证集')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter
    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'训练结束，花费总时间为: {consumed_time} 秒')
    logger.info('保存最新模型')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

    # 开始测试最好的那个
    if opt.get('val') is not None and test_loader is not None:
        test_net_g_opt = opt['network_g'].copy()
        if test_net_g_opt.get('type') == 'FusionRefine':
            test_net_g_opt['opt']["use_weight"] = None
            test_net_g_opt['opt']["weight_alpha"] = -21.25
        test_net_g = build_network(test_net_g_opt)
        model.model_to_device(test_net_g)

        latest_net_g_path = osp.join('experiments', opt['name'], 'models/net_g_latest.pth')
        best_net_g_path = osp.join('experiments', opt['name'], 'models/net_g_best_.pth')
        if osp.isfile(best_net_g_path):
            model.load_network(test_net_g, best_net_g_path, False)
            model.nondist_test(test_net_g, test_loader, 999999, tb_logger, opt['val']['save_img'])
        if osp.isfile(latest_net_g_path):
            model.load_network(test_net_g, latest_net_g_path, False)
            model.nondist_test(test_net_g, test_loader, opt['train']['total_iter'], tb_logger, opt['val']['save_img'])

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
