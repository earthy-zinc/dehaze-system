from collections import OrderedDict
from os import path as osp

import math
from thop import profile
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa

from ..utils.static_util import convert_size
import torch.nn.functional as F

@MODEL_REGISTRY.register()
class ITBModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        logger = get_root_logger()
        # define network
        test_input = torch.randn(1, 3, 256, 256).to(self.device)
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        net_g_flops, net_g_params = profile(self.net_g, inputs=(test_input,))
        logger.info("去雾模型net_g的FLOPS量为{}，参数量为{}。"
                    .format(convert_size(net_g_flops), convert_size(net_g_params)))

    # load pre-trained HQ ckpt, frozen decoder and codebook
        self.LQ_stage = self.opt['network_g']["opt"].get('LQ_stage', False)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            hq_opt = self.opt['network_hq']
            self.net_hq = build_network(hq_opt)
            self.net_hq = self.model_to_device(self.net_hq)

            net_hq_flops, net_hq_params = profile(self.net_hq, inputs=(test_input,))
            logger.info("生成模型net_hq的FLOPS量为{}，参数量为{}。"
                        .format(convert_size(net_hq_flops), convert_size(net_hq_params)))
            logger.info("生成模型net_hq和去雾模型net_g的FLOPS量差距为{}，参数量差距为{}。".format(
                convert_size(net_g_flops - net_hq_flops),
                convert_size(net_g_params - net_hq_params)
            ))

            logger.info("加载预训练的HQP模型")
            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            logger.info("加载本次去雾模型")
            load_init_net_g_path = self.opt['path'].get('pretrain_network_init_g', None)
            self.load_network(self.net_g, load_init_net_g_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            logger.info(f'从 {load_path} 中加载去雾网络（network generator）')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        # define metric functions
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        if self.is_train:
            self._init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
            self.net_d_best = copy.deepcopy(self.net_d)

        self.net_g_best = copy.deepcopy(self.net_g)

    def _init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            logger.info(f'从 {load_path} 中加载判别网络（network discriminator）')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        self.net_d.train()

        # define losses (criterion)
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('ms_ssim_opt'):
            self.cri_ms_ssim = build_loss(train_opt['ms_ssim_opt']).to(self.device)
        else:
            self.cri_ms_ssim = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # Todo 这是用来干啥的？
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.info(f'神经网络参数 {k} 将不会被优化器调整。')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

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

        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        if self.LQ_stage:
            with torch.no_grad():
                self.gt_rec, _, _, _, quant_gt, gt_indices = self.net_hq(self.gt)
            self.lq.requires_grad = True
            self.output, self.output_residual, l_codebook, quant_g, _, _ = self.net_g(self.lq, gt_indices)
        else:
            self.output, self.output_residual, l_codebook, quant_g, _, _ = self.net_g(self.gt)

        l_g_total = torch.zeros((1,)).to(self.device)
        loss_dict = OrderedDict()

        if self.output_residual is None:
            self.output_residual = self.output
            quant_gt = quant_g

        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None):
            l_codebook *= train_opt['codebook_opt']['loss_weight']
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()

        # semantic cluster loss, only for HQ stage!
        l_semantic = None
        if not self.LQ_stage:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(self.gt)[self.vgg_feat_layer]
            semantic_z_quant = self.conv_semantic(quant_g)
            l_semantic = F.mse_loss(semantic_z_quant, vgg_feat)
        if train_opt.get('semantic_opt', None) and isinstance(l_semantic, torch.Tensor):
            l_semantic *= train_opt['semantic_opt']['loss_weight']
            l_semantic = l_semantic.mean()
            l_g_total += l_semantic
            loss_dict['l_semantic'] = l_semantic

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output_residual, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # ms ssim loss
        if self.cri_ms_ssim:
            l_ms_ssim = self.cri_ms_ssim(self.output_residual, self.gt)
            l_g_total += l_ms_ssim
            loss_dict["l_ms_ssim"] = l_ms_ssim

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output_residual, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style

        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_iters']:
            fake_g_pred = self.net_d(quant_g)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.mean().backward()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real real_d_pred清晰图像经判别器处理应该为正值，越大越好，则gan损失会越小
            real_d_pred = self.net_d(quant_gt)
            # 经gan损失函数处理的输入大于1，则损失为0，等于1，损失为0。等于0，损失为1。输入越小，损失越大
            # 简单说，输入越大，损失越小
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(quant_g.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()

            self.optimizer_d.step()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 512 * 512  # use smaller min_size with limited GPU memory
        max_size = 1024 * 1024
        if self.lq is not None:
            inputs = self.lq
        else:
            inputs = self.gt
        _, _, h, w = inputs.shape
        if h * w < min_size:
            self.output, _ = net_g.test(inputs)
        elif h * w > max_size:
            num = math.floor(math.sqrt(h * w) / 1000)
            down_img = torch.nn.UpsamplingBilinear2d((h//num, w//num))(inputs)
            output, _ = net_g.test(down_img)
            self.output = torch.nn.UpsamplingBilinear2d((h, w))(output)
        else:
            down_img = torch.nn.UpsamplingBilinear2d((h//2, w//2))(inputs)
            output, _ = net_g.test(down_img)
            self.output = torch.nn.UpsamplingBilinear2d((h, w))(output)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('神经网络验证仅支持单个GPU')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir):
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
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
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
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, str(save_img_path))

            if with_metrics:
                # calculate metrics
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
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')
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
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

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

    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']["opt"]["codebook_emb_num"]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            # TODO
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        if self.lq != None:
            out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        if self.output != None:
            out_dict['result_codebook'] = self.output.detach().cpu()[:vis_samples]
        if self.output_residual != None:
            out_dict['result_residual'] = self.output_residual.detach().cpu()[:vis_samples]
        if not self.LQ_stage:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
