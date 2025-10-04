import copy
import math
from collections import OrderedDict
import os
import pyiqa
import torch
import torchvision.transforms
from thop import profile
from torch import nn
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, tensor2img, img2tensor, imwrite
from basicsr.utils.img_util import resize_image
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.static_util import convert_size


@MODEL_REGISTRY.register()
class DehamerModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        logger = get_root_logger()
        # lq => low quality image => hazy image
        # gt => ground truth => clean image
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # test_input = torch.randn(1, 3, 256, 256).to(self.device)
        # net_g_flops, net_g_params = profile(self.net_g, inputs=(test_input,))
        # logger.info("去雾模型net_g的FLOPS量为{}，参数量为{}。"
        #             .format(convert_size(net_g_flops), convert_size(net_g_params)))

        self.lq = None
        self.gt = None
        self.output = None
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)
        if self.is_train:
            self._init_training_settings()
        self.net_g_best = copy.deepcopy(self.net_g)
        self.normalize = torchvision.transforms.Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))


    def _init_training_settings(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, self.net_g.parameters(),
                                            **train_opt['optim_g'])
        # TODO
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.opt, factor=0.5, verbose=True)
        self.cri_loss = nn.MSELoss().to(self.device)


    def feed_data(self, data):
        if 'lq' in data:
            self.lq = resize_image(data['lq'].to(self.device))
            self.lq = self.normalize(self.lq)
        else:
            self.lq = None
        if 'gt' in data:
            self.gt = resize_image(data['gt'].to(self.device))
        else:
            self.gt = None

    def optimize_parameters(self, current_iter):
        self.output = self.net_g(self.lq)
        loss_dict = OrderedDict()

        loss = self.cri_loss(self.output, self.gt)
        loss_dict['loss'] = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.optimizer.step()
        self.scheduler.step(loss)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None, update_net=True):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        pbar = tqdm(total=len(dataloader), unit='张')

        for idx, val_data in enumerate(dataloader):
            img_name = os.path.splitext(os.path.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'], 'image_results',
                                                 f'{current_iter}',
                                                 f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = os.path.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = os.path.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = os.path.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, str(save_img_path))

            if with_metrics:
                # calculate metrics
                with torch.no_grad():
                    for name, opt_ in self.opt['val']['metrics'].items():
                        if name == "niqe" or name == "brisque" or name == "nima":
                            tmp_result = self.metric_funcs[name](metric_data[0])
                        else:
                            tmp_result = self.metric_funcs[name](*metric_data)
                        self.metric_results[name] += tmp_result.item()

            pbar.update(1)
            pbar.set_description(f'测试图片 {img_name} 中')

        pbar.close()

        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            if update_net:
                if self.key_metric is not None:
                    # If the best metric is updated, update and save best model
                    to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                                self.metric_results[self.key_metric], current_iter)

                    if to_update:
                        for name, opt_ in self.opt['val']['metrics'].items():
                            self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                        self.copy_model(self.net_g, self.net_g_best)
                        self.save_network(self.net_g, 'net_g_best', '')
                else:
                    # update each metric separately
                    updated = []
                    for name, opt_ in self.opt['val']['metrics'].items():
                        tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                      current_iter)
                        updated.append(tmp_updated)
                    # save best model if any metric is updated
                    if sum(updated):
                        self.copy_model(self.net_g, self.net_g_best)
                        self.save_network(self.net_g, 'net_g_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'验证集 {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\t最佳结果: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'评估指标/{dataset_name}/{metric}', value, current_iter)

    def nondist_test(self, net, dataloader, current_iter, tb_logger, save_img):
        self.net_g = net
        self.nondist_validation(dataloader, current_iter, tb_logger,
                                save_img, None, False)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'output'):
            out_dict['output'] = self.output.detach().cpu()
        if hasattr(self, 'lq'):
            out_dict['lq'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
