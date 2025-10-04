import copy
import os
from collections import OrderedDict

import pyiqa
import torch
import math

import torchvision.transforms
from thop import profile
from timm.scheduler import CosineLRScheduler
from torch import nn
from torch.optim import lr_scheduler
from torchvision.models import vgg16
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from basicsr.archs import build_network
from basicsr.archs.swin_transformer_v2 import SwinTransformerV2
from basicsr.losses.FAA_loss import FAALossNetwork
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, tensor2img, img2tensor, imwrite
from basicsr.utils.img_util import resize_image
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.static_util import convert_size
import torch.nn.functional as F

@MODEL_REGISTRY.register()
class ITBDehazeModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        logger = get_root_logger()
        # lq => low quality image => hazy image
        # gt => ground truth => clean image
        net_g_opt = opt['network_g']
        self.net_g = build_network(net_g_opt)
        self.net_g = self.model_to_device(self.net_g)
        test_input = torch.randn(1, 3, 256, 256).to(self.device)
        net_g_flops, net_g_params = profile(self.net_g, inputs=(test_input,))
        logger.info("去雾模型net_g的FLOPS量为{}，参数量为{}。"
                    .format(convert_size(net_g_flops), convert_size(net_g_params)))
        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
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

    def _init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()
        self.net_d.train()
        vgg_model = vgg16(pretrained=True)
        # vgg_model.load_state_dict(torch.load(os.path.join(args.vgg_model , 'vgg16.pth')))
        vgg_model = vgg_model.features[:16].to(self.device)
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.cri_perceptual = FAALossNetwork(vgg_model)
        self.cri_perceptual.eval()
        self.cri_ms_ssim = ms_ssim
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_g_type = train_opt['optim_g'].pop('type')
        self.optimizer_G = self.get_optimizer(optim_g_type,
                                              filter(lambda x: x.requires_grad, self.net_g.parameters()),
                                              **train_opt['optim_g'])
        optim_d_type = train_opt['optim_d'].pop('type')
        self.optimizer_D = self.get_optimizer(optim_d_type,
                                              filter(lambda x: x.requires_grad, self.net_d.parameters()),
                                              **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def setup_schedulers(self):
        train_opt = self.opt['train']
        scheduler_g_type = train_opt['scheduler_g'].pop('type')
        scheduler_d_type = train_opt['scheduler_d'].pop('type')
        if scheduler_g_type in ['CosineLRScheduler']:
            self.schedulers.append(CosineLRScheduler(self.optimizer_G, **train_opt['scheduler_g']))
        elif scheduler_d_type == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer_D, **train_opt['scheduler_d']))

    def feed_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        else:
            self.lq = None
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = None

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        loss_dict = OrderedDict()
        self.output = self.net_g(self.lq)
        self.net_d.zero_grad()
        real_out = self.net_d(self.gt).mean()
        fake_out = self.net_d(self.output).mean()
        d_loss = 1 - real_out + fake_out
        loss_dict['d_loss'] = d_loss
        d_loss.backward(retain_graph=True)
        self.net_g.zero_grad()
        g_adversarial_loss = torch.mean(1 - fake_out)
        g_smooth_loss = F.smooth_l1_loss(self.output, self.gt)
        g_perceptual_loss = self.cri_perceptual(self.output, self.gt)
        g_ms_ssim_loss = - ms_ssim(self.output, self.gt)
        total_loss = g_smooth_loss + 0.01 * g_perceptual_loss + 0.0005 * g_adversarial_loss + 0.5 * g_ms_ssim_loss
        loss_dict['g_loss'] = total_loss
        self.log_dict = self.reduce_loss_dict(loss_dict)
        total_loss.backward()
        self.optimizer_D.step()
        self.optimizer_G.step()


    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                # upscale tile
                output_tile = self.resize(input_tile)

                # output tile area on total image
                output_start_x = input_start_x
                output_end_x = input_end_x
                output_start_y = input_start_y
                output_end_y = input_end_y

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad)
                output_end_x_tile = output_start_x_tile + input_tile_width
                output_start_y_tile = (input_start_y - input_start_y_pad)
                output_end_y_tile = output_start_y_tile + input_tile_height

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        # print(output.shape)
        return output

    def resize(self, input):
        _, _, h_old, w_old = input.shape
        h_pad = 1024 - h_old
        w_pad = 1024 - w_old
        while input.shape[2] != 1024:
            input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        while input.shape[3] != 1024:
            input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]
        # print(input.shape)
        output = self.net_g(input)
        output = output[..., :h_old, :w_old]
        return output


    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            # print(self.lq.shape)
            self.output = self.test_tile(self.lq, 900, 124)
            # self.output = self.net_g(self.lq)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
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

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    # self.copy_model(self.net_g, self.net_g_best)
                    # self.save_network(self.net_g, 'net_g_best', '')
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                # if sum(updated):
                #     self.copy_model(self.net_g, self.net_g_best)
                #     self.save_network(self.net_g, 'net_g_best', '')

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
                                save_img, None)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'output'):
            out_dict['output'] = self.output.detach().cpu()
        if hasattr(self, 'lq'):
            out_dict['lq'] = self.gt.detach().cpu()
        return out_dict
