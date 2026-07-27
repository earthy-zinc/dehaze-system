SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (1, 0, '图像去雾', 'AECR-NET', null, 'AECR-Net/NH_train.pk', '35.1 MB', null, null, 'algorithm.AECRNet.run',
        'AECRNet 是一种深度学习模型，专门用于图像去雾任务。该模型由清华大学和微软亚洲研究院的研究人员在2019年提出，旨在解决传统去雾方法中存在的边缘模糊和细节丢失问题。AECRNet 通过引入对抗生成网络（GAN）和边缘保持机制，实现了高质量的去雾效果。',
        3, '2024-11-11 20:00:28', '2024-11-11 20:00:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (2, 0, '图像去雾', 'AODNet', null, 'AODNet/dehazer.pth', '8.41 KB', '1.72 KB', '109.12 MB',
        'algorithm.AODNet.run',
        'AODNet (All-in-One Dehazing Network) 是一种用于图像去雾的深度学习模型，由Yuan et al. 在2018年提出。传统的图像去雾方法通常依赖于大气散射模型以及一些先验知识，如暗通道先验等，这些方法虽然在某些情况下能够取得较好的效果，但是往往计算复杂度较高，且对于不同的环境条件适应性较差。AODNet旨在解决这些问题，提供一个更加高效和鲁棒的解决方案。',
        3, '2024-11-11 23:52:37', '2024-12-03 21:21:36', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (3, 0, '图像去雾', 'C2PNet', null, 'C2PNet/ITS.pkl', '35.98 MB', '6.84 MB', '429.3 GB', 'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        3, '2024-11-12 22:51:50', '2024-12-03 21:21:37', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (4, 3, '图像去雾', '室内去雾（ITS）', null, 'C2PNet/ITS.pkl', '35.98 MB', '6.84 MB', '429.3 GB',
        'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        3, '2024-11-12 22:52:15', '2024-12-03 21:21:37', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (5, 3, '图像去雾', '室外去雾（OTS）', null, 'C2PNet/OTS.pkl', '39.51 MB', '6.84 MB', '429.3 GB',
        'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        3, '2024-11-12 22:52:25', '2024-12-03 21:21:38', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (6, 0, '图像去雾', 'CMFNet', null, 'CMFNet', null, null, null, 'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        3, '2024-11-13 11:30:00', '2024-11-13 11:30:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (7, 6, '图像去雾', '去雾模型', null, 'CMFNet/dehaze_I_OHaze_CMFNet.pth', '197.53 MB', '16.44 MB', '595.07 GB',
        'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        3, '2024-11-13 11:30:16', '2024-12-03 21:21:41', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (8, 6, '图像去模糊', '去模糊模型', null, 'CMFNet/deblur_GoPro_CMFNet.pth', '197.53 MB', '16.44 MB', '595.07 GB',
        'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        3, '2024-11-13 11:30:26', '2024-12-03 21:21:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (9, 6, '图像去雨', '去雨模型', null, 'CMFNet/deraindrop_DeRainDrop_CMFNet.pth', '197.53 MB', '16.44 MB',
        '595.07 GB', 'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        3, '2024-11-13 11:30:33', '2024-12-03 21:21:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (10, 0, '图像去雾', 'D4', null, 'D4/weights_reconstruct.pth', '88.16 MB', null, null, 'algorithm.D4.run', '', 3,
        '2024-11-13 12:39:45', '2024-11-13 12:39:45', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (11, 0, '图像去雾', 'DaclipUir', null, 'daclip-uir/daclip_ViT-B-32.pt', '1.62 GB', null, null,
        'algorithm.DaclipUir.run',
        'DaclipUir 是一种先进的图像去雾模型，它结合了深度学习与物理模型，通过优化图像的对比度和色彩，有效去除雾霾，提高图像的清晰度。该模型特别注重保留图像的细节和自然度，适用于多种场景下的图像去雾任务',
        3, '2024-11-13 12:39:54', '2024-11-13 12:39:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (12, 0, '图像去雾', 'DCPDN', null, 'DCPDN/netG_epoch_8.pth', '255.55 MB', null, null, 'algorithm.DCPDN.run',
        'DCPDN 是一种基于深度学习的图像去雾方法，通过大气散射模型和密集连接的编码器-解码器结构，估计透射率图并进行去雾。该模型利用多级金字塔池化模块，提高了透射率估计的准确性，从而改善了去雾效果',
        3, '2024-11-13 12:40:03', '2024-11-13 12:40:03', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (13, 0, '图像去雾', 'DCP', null, '', null, null, null, 'algorithm.DCP.run',
        'DCP 是由何凯明等人在2009年提出的经典去雾算法，基于暗原色先验理论。该算法假设无雾图像的局部区域中至少有一个颜色通道的亮度值非常低。通过估计大气光和透射率，DCP 能够有效地去除图像中的雾霾，恢复图像的清晰度',
        3, '2024-11-13 12:40:13', '2024-11-13 12:40:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (14, 0, '图像去雾', 'DEANet', null, 'DEA-Net', null, null, null, 'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        3, '2024-11-13 12:40:20', '2024-11-13 12:40:20', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (15, 14, '图像去雾', 'HAZE4k模型', null, 'DEA-Net/HAZE4K/PSNR3426_SSIM9885.pth', '14 MB', '3.48 MB', '31.7 GB',
        'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        3, '2024-11-13 12:40:55', '2024-12-03 21:22:04', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (16, 14, '图像去雾', 'ITS模型', null, 'DEA-Net/ITS/PSNR4131_SSIM9945.pth', '14 MB', '3.48 MB', '31.7 GB',
        'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        3, '2024-11-13 12:41:02', '2024-12-03 21:22:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (17, 14, '图像去雾', 'OTS模型', null, 'DEA-Net/OTS/PSNR3659_SSIM9897.pth', '14 MB', '3.48 MB', '31.7 GB',
        'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        3, '2024-11-13 12:41:09', '2024-12-03 21:22:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (18, 0, '图像去雾', 'Dehamer', null, 'Dehamer', null, null, null, 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        3, '2024-11-13 12:41:26', '2024-11-13 12:41:26', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (19, 18, '图像去雾', 'dense-haze模型', null, 'Dehamer/dense/PSNR1662_SSIM05602.pt', '511.68 MB', '28.08 MB',
        '55.58 GB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        3, '2024-11-13 12:41:56', '2024-12-03 21:22:10', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (20, 18, '图像去雾', 'indoor', null, 'Dehamer/indoor/PSNR3663_ssim09881.pt', '511.68 MB', '28.08 MB', '55.91 GB',
        'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        3, '2024-11-13 12:42:02', '2024-12-03 21:22:14', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (21, 18, '图像去雾', 'NH-HAZE-模型', null, 'Dehamer/NH/PSNR2066_SSIM06844.pt', '511.68 MB', '28.08 MB',
        '56.25 GB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        3, '2024-11-13 12:42:10', '2024-12-03 21:22:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (22, 18, '图像去雾', 'outdoor', null, 'Dehamer/outdoor/PSNR3518_SSIM09860.pt', '511.68 MB', '28.08 MB',
        '56.59 GB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        3, '2024-11-13 12:42:17', '2024-12-03 21:22:23', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (23, 0, '图像去雾', 'DehazeFormer', null, 'DehazeFormer', null, null, null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:42:35', '2024-11-13 12:42:35', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (24, 23, '图像去雾', 'indoor', null, 'DehazeFormer/indoor', null, null, null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:42:57', '2024-11-13 12:42:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (25, 24, '图像去雾', 'indoor-b', null, 'DehazeFormer/indoor/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:45:52', '2024-12-03 21:22:24', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (26, 24, '图像去雾', 'indoor-d', null, 'DehazeFormer/indoor/dehazeformer-d.pth', '21.22 MB', '4.75 MB',
        '43.57 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:45:59', '2024-12-03 21:22:26', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (27, 24, '图像去雾', 'indoor-l', null, 'DehazeFormer/indoor/dehazeformer-l.pth', '98.22 MB', '24.27 MB',
        '256.26 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:46:05', '2024-12-03 21:22:27', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (28, 24, '图像去雾', 'indoor-m', null, 'DehazeFormer/indoor/dehazeformer-m.pth', '18.51 MB', '4.42 MB',
        '43.79 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:46:12', '2024-12-03 21:22:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (29, 24, '图像去雾', 'indoor-s', null, 'DehazeFormer/indoor/dehazeformer-s.pth', '5.46 MB', '1.22 MB',
        '11.23 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:46:19', '2024-12-03 21:22:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (30, 24, '图像去雾', 'indoor-t', null, 'DehazeFormer/indoor/dehazeformer-t.pth', '2.9 MB', '670.36 KB',
        '5.82 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:46:25', '2024-12-03 21:22:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (31, 24, '图像去雾', 'indoor-w', null, 'DehazeFormer/indoor/dehazeformer-w.pth', '38.06 MB', '9.23 MB',
        '83.69 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:46:33', '2024-12-03 21:22:29', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (32, 23, '图像去雾', 'outdoor', null, 'DehazeFormer/outdoor', null, null, null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:46:44', '2024-11-13 12:46:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (33, 32, '图像去雾', 'outdoor-b', null, 'DehazeFormer/outdoor/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:51:32', '2024-12-03 21:22:30', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (34, 32, '图像去雾', 'outdoor-m', null, 'DehazeFormer/outdoor/dehazeformer-m.pth', '18.51 MB', '4.42 MB',
        '43.79 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:51:38', '2024-12-03 21:22:31', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (35, 32, '图像去雾', 'outdoor-s', null, 'DehazeFormer/outdoor/dehazeformer-s.pth', '5.46 MB', '1.22 MB',
        '11.23 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:51:44', '2024-12-03 21:22:31', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (36, 32, '图像去雾', 'outdoor-t', null, 'DehazeFormer/outdoor/dehazeformer-t.pth', '2.9 MB', '670.36 KB',
        '5.82 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:51:50', '2024-12-03 21:22:31', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (37, 23, '图像去雾', 'reside6k', null, 'DehazeFormer/reside6k', null, null, null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:52:00', '2024-11-13 12:52:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (38, 37, '图像去雾', 'reside6k-b', null, 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:52:13', '2024-12-03 21:22:32', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (39, 37, '图像去雾', 'reside6k-b', null, 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:52:19', '2024-12-03 21:22:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (40, 37, '图像去雾', 'reside6k-b', null, 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:52:26', '2024-12-03 21:22:34', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (41, 37, '图像去雾', 'reside6k-b', null, 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        3, '2024-11-13 12:52:34', '2024-12-03 21:22:35', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (42, 23, '图像去雾', 'rshaze', null, 'DehazeFormer/rshaze', null, null, null, 'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        3, '2024-11-13 12:52:50', '2024-11-13 12:52:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (43, 42, '图像去雾', 'rshaze-b', null, 'DehazeFormer/rshaze/dehazeformer-b.pth', '10.71 MB', null, null,
        'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        3, '2024-11-13 12:59:58', '2024-11-13 12:59:58', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (44, 42, '图像去雾', 'rshaze-m', null, 'DehazeFormer/rshaze/dehazeformer-m.pth', '18.51 MB', null, null,
        'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        3, '2024-11-13 13:00:05', '2024-11-13 13:00:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (45, 42, '图像去雾', 'rshaze-s', null, 'DehazeFormer/rshaze/dehazeformer-s.pth', '5.46 MB', null, null,
        'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        3, '2024-11-13 13:00:18', '2024-11-13 13:00:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (46, 42, '图像去雾', 'rshaze-t', null, 'DehazeFormer/rshaze/dehazeformer-t.pth', '2.9 MB', null, null,
        'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        3, '2024-11-13 13:00:25', '2024-11-13 13:00:25', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (47, 0, '图像去雾', 'FCD', null, 'FCD/framework_da_230221_121802_gen.pth', '592.75 MB', null, null,
        'algorithm.FCD.run',
        'FCD 是一种基于全卷积网络的图像去雾方法，通过密集连接的卷积层进行端到端的去雾处理。该模型简化了模型结构，提高了计算效率，适用于实时去雾应用',
        3, '2024-11-13 13:00:33', '2024-11-13 13:00:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (48, 0, '图像去雾', 'FFANet', null, 'FFA-Net', null, null, null, 'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        3, '2024-11-13 13:00:40', '2024-11-13 13:00:40', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (49, 48, '图像去雾', 'its', null, 'FFA-Net/its_train_ffa_3_19.pk', '21.26 MB', null, null,
        'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        3, '2024-11-13 13:12:54', '2024-11-13 13:12:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (50, 48, '图像去雾', 'ots', null, 'FFA-Net/ots_train_ffa_3_19.pk', '25.39 MB', null, null,
        'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        3, '2024-11-13 13:13:01', '2024-11-13 13:13:01', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (51, 0, '图像去雾', 'FogRemoval', null, 'FogRemoval/NH-HAZE_params_0100000.pt', '512.01 MB', null, null,
        'algorithm.FogRemoval.run',
        'FogRemoval 是一种多阶段的图像去雾方法，通过逐步优化图像质量，实现自然的去雾效果。该模型结合了物理模型和深度学习，能够在不同光照条件下有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:13:11', '2024-11-13 13:13:11', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (52, 0, '图像去雾', 'GCANet', null, 'GCANet/wacv_gcanet_dehaze.pth', '2.69 MB', null, null,
        'algorithm.GCANet.run',
        'GCANet 是一种利用全局上下文模块的图像去雾模型，通过增强模型对全局信息的理解，改善去雾结果。该模型在处理复杂场景时，能够有效保留图像的结构和细节，提高去雾效果',
        3, '2024-11-13 13:13:18', '2024-11-13 13:13:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (53, 0, '图像去雾', 'GridDehazeNet', null, 'GridDehazeNet', null, null, null, 'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        3, '2024-11-13 13:13:27', '2024-11-13 13:13:27', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (54, 53, '图像去雾', 'indoor', null, 'GridDehazeNet/indoor_haze_best_3_6', '3.71 MB', null, null,
        'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        3, '2024-11-13 13:13:38', '2024-11-13 13:13:38', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (55, 53, '图像去雾', 'outdoor', null, 'GridDehazeNet/outdoor_haze_best_3_6', '3.71 MB', null, null,
        'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        3, '2024-11-13 13:13:44', '2024-11-13 13:13:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (56, 0, '图像去雾', 'ImgRestorationSde', null, 'image-restoration-sde', null, null, null,
        'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:13:52', '2024-11-13 13:13:52', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (57, 56, '图像去模糊', 'deblurring', null, 'image-restoration-sde/deblurring/ir-sde-deblurring.pth', '523.23 MB',
        null, null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:22:32', '2024-11-13 13:22:32', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (58, 56, '图像去噪', 'denoising', null, 'image-restoration-sde/denoising/ir-sde-denoising.pth', '523.19 MB',
        null, null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:22:39', '2024-11-13 13:22:39', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (59, 56, '图像去雨', 'deraining', null, 'image-restoration-sde/deraining', null, null, null,
        'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:22:46', '2024-11-13 13:22:46', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (60, 59, '图像去雨', 'deraining-H100', null, 'image-restoration-sde/deraining/ir-sde-derainH100.pth',
        '523.23 MB', null, null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:22:55', '2024-11-13 13:22:55', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (61, 59, '图像去雨', 'deraining-L100', null, 'image-restoration-sde/deraining/ir-sde-derainL100.pth',
        '523.23 MB', null, null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:23:01', '2024-11-13 13:23:01', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (62, 0, '图像去雾', 'ITBDehaze', null, 'ITBdehaze/best.pkl', '423.84 MB', null, null, 'algorithm.ITBDehaze.run',
        'ITBDehaze (Image Texture and Boundary Dehazing)是一种利用图像的纹理和边界信息的图像去雾模型，通过多尺度处理增强去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度24',
        3, '2024-11-13 13:23:09', '2024-11-13 13:23:09', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (63, 0, '图像去雾', 'LightDehazeNet', null, 'LightDehazeNet/trained_LDNet.pth', '122.61 KB', '29.48 KB',
        '1.84 GB', 'algorithm.LightDehazeNet.run',
        'LightDehazeNet 是一种轻量级的图像去雾模型，适用于移动设备上的实时去雾应用。该模型通过优化网络结构，减少了计算复杂度，同时保持了较高的去雾效果',
        3, '2024-11-13 13:23:15', '2024-12-03 21:22:41', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (64, 0, '图像去雾', 'LKDNet', null, 'LKDNet', null, null, null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:23:22', '2024-11-13 13:23:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (65, 64, '图像去雾', 'ITS', null, 'LKDNet/ITS', null, null, null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:23:29', '2024-11-13 13:23:29', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (66, 65, '图像去雾', 'ITS-b', null, 'LKDNet/ITS/LKD-b/LKD-b.pth', '4.94 MB', '1.16 MB', '11.32 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:23:47', '2024-12-03 21:22:41', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (67, 65, '图像去雾', 'ITS-l', null, 'LKDNet/ITS/LKD-l/LKD-l.pth', '9.67 MB', '2.27 MB', '22.2 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:23:53', '2024-12-03 21:22:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (68, 65, '图像去雾', 'ITS-s', null, 'LKDNet/ITS/LKD-s/LKD-s.pth', '2.57 MB', '619.33 KB', '5.89 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:23:58', '2024-12-03 21:22:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (69, 65, '图像去雾', 'ITS-t', null, 'LKDNet/ITS/LKD-t/LKD-t.pth', '1.39 MB', '335.13 KB', '3.17 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:24:08', '2024-12-03 21:22:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (70, 64, '图像去雾', 'OTS', null, 'LKDNet/OTS', null, null, null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:24:15', '2024-11-13 13:24:15', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (71, 70, '图像去雾', 'OTS-b', null, 'LKDNet/OTS/LKD-b/LKD-b.pth', '4.94 MB', '1.16 MB', '11.32 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:24:21', '2024-12-03 21:22:43', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (72, 70, '图像去雾', 'OTS-l', null, 'LKDNet/OTS/LKD-l/LKD-l.pth', '9.67 MB', '2.27 MB', '22.2 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:24:27', '2024-12-03 21:22:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (73, 70, '图像去雾', 'OTS-s', null, 'LKDNet/OTS/LKD-s/LKD-s.pth', '2.57 MB', '619.33 KB', '5.89 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:24:35', '2024-12-03 21:22:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (74, 70, '图像去雾', 'OTS-t', null, 'LKDNet/OTS/LKD-t/LKD-t.pth', '1.39 MB', '335.13 KB', '3.17 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        3, '2024-11-13 13:24:41', '2024-12-03 21:22:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (75, 0, '图像去雾', 'MADN', null, 'MADN/model.pth', '2.13 MB', null, null, 'algorithm.MADN.run',
        'MADN (Multi-Adversarial Domain Network): MADN 是一种基于多对抗域网络的图像去雾模型，通过域适应技术，提高了模型对不同场景的适应能力。该模型在处理真实世界中的雾霾图像时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:24:50', '2024-11-13 13:24:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (76, 0, '图像去雾', 'MB-TaylorFormer', null, 'MB-TaylorFormer', null, null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:24:57', '2024-11-13 13:24:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (77, 76, '图像去雾', 'dense-haze', null, '', null, null, null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:28:22', '2024-11-13 13:28:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (78, 77, '图像去雾', 'dense-haze-b', null, 'MB-TaylorFormer/densehaze-MB-TaylorFormer-B.pth', '10.49 MB', null,
        null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:28:29', '2024-11-13 13:28:29', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (79, 77, '图像去雾', 'dense-haze-l', null, 'MB-TaylorFormer/densehaze-MB-TaylorFormer-L.pth', '29.04 MB', null,
        null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:28:36', '2024-11-13 13:28:36', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (80, 76, '图像去雾', 'its', null, 'MB-TaylorFormer/ITS-MB-TaylorFormer-L.pth', '29.04 MB', null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:28:43', '2024-11-13 13:28:43', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (81, 76, '图像去雾', 'ohaze', null, 'MB-TaylorFormer/ohaze-MB-TaylorFormer-B.pth', '10.49 MB', null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:28:49', '2024-11-13 13:28:49', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (82, 76, '图像去雾', 'ots', null, '', null, null, null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:29:00', '2024-11-13 13:29:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (83, 82, '图像去雾', 'ots-b', null, 'MB-TaylorFormer/OTS-MB-TaylorFormer-B.pth', '10.51 MB', null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:29:06', '2024-11-13 13:29:06', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (84, 82, '图像去雾', 'ots-l', null, 'MB-TaylorFormer/OTS-MB-TaylorFormer-L.pth', '29.04 MB', null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        3, '2024-11-13 13:29:12', '2024-11-13 13:29:12', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (85, 0, '图像去雾', 'MixDehazeNet', null, 'MixDehazeNet', null, null, null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 13:29:57', '2024-11-13 13:29:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (86, 85, '图像去雾', 'haze4k', null, 'MixDehazeNet/haze4k', null, null, null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:26:20', '2024-11-13 14:26:20', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (87, 86, '图像去雾', 'haze4k-l', null, 'MixDehazeNet/haze4k/MixDehazeNet-l.pth', '143.93 MB', '11.84 MB',
        '104.58 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:26:47', '2024-12-03 21:22:48', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (88, 85, '图像去雾', 'indoor', null, 'MixDehazeNet/indoor', null, null, null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:26:54', '2024-11-13 14:26:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (89, 88, '图像去雾', 'indoor-b', null, 'MixDehazeNet/indoor/MixDehazeNet-b.pth', '72.44 MB', '5.96 MB',
        '52.6 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:27:00', '2024-12-03 21:22:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (90, 88, '图像去雾', 'indoor-l', null, 'MixDehazeNet/indoor/MixDehazeNet-l.pth', '143.93 MB', '11.84 MB',
        '104.58 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:27:06', '2024-12-03 21:22:55', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (91, 85, '图像去雾', 'outdoor', null, 'MixDehazeNet/outdoor', null, null, null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:27:13', '2024-11-13 14:27:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (92, 91, '图像去雾', 'outdoor-b', null, 'MixDehazeNet/outdoor/MixDehazeNet-b.pth', '72.44 MB', '5.96 MB',
        '52.6 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:27:26', '2024-12-03 21:22:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (93, 91, '图像去雾', 'outdoor-b', null, 'MixDehazeNet/outdoor/MixDehazeNet-l.pth', '143.93 MB', '11.84 MB',
        '104.58 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:27:35', '2024-12-03 21:23:01', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (94, 91, '图像去雾', 'outdoor-b', null, 'MixDehazeNet/outdoor/MixDehazeNet-s.pth', '36.69 MB', '3.02 MB',
        '26.61 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:27:41', '2024-12-03 21:23:02', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (95, 0, '图像去雾', 'MSFNet', null, 'MSFNet', null, null, null, 'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:27:49', '2024-11-13 14:27:49', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (96, 95, '图像去雾', 'indoor', null, 'MSFNet/indoor.pth', '4.01 MB', '1003.22 KB', '17.01 GB',
        'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:27:58', '2024-12-03 21:23:03', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (97, 95, '图像去雾', 'outdoor', null, 'MSFNet/outdoor.pth', '3.98 MB', '1003.22 KB', '17.01 GB',
        'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:28:04', '2024-12-03 21:23:03', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (98, 0, '图像去雾', 'PSD', null, 'PSD', null, null, null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:28:10', '2024-11-13 14:28:10', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (99, 98, '图像去雾', 'PSD-MSBDN', null, 'PSD/PSB-MSBDN', '126.4 MB', null, null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:28:19', '2024-11-13 14:28:19', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (100, 98, '图像去雾', 'PSD-FFANET', null, 'PSD/PSD-FFANET', '23.84 MB', null, null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:28:33', '2024-11-13 14:28:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (101, 98, '图像去雾', 'PSD-GCANET', null, 'PSD/PSD-GCANET', '9.23 MB', null, null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:28:42', '2024-11-13 14:28:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (102, 0, '图像去雾', 'RIDCP', null, 'RIDCP/pretrained_RIDCP.pth', '116.41 MB', '27.39 MB', '175.69 GB',
        'algorithm.RIDCP.run',
        '目前图像去雾领域缺乏强大的先验知识，作者提出在 VQGAN1使用大规模高质量数据集，预训练出一个离散码本，封装高质量先验（HQPs）；并且引入了一种提取特征能力较强的编码器 E，以及设计了一个具有归一化特征对齐模块（NFA）的解码器 G ，共同构建出基于高质量码本先验的真实图像去雾网络（RIDCP）',
        3, '2024-11-13 14:28:49', '2024-12-03 21:23:09', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (103, 0, '图像去雾', 'SCANet', null, 'SCANet', null, null, null, 'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:28:56', '2024-11-13 14:28:56', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (104, 103, '图像去雾', 'SCANet-40', null, 'SCANet/Gmodel_40.tar', '27.7 MB', null, null, 'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:29:05', '2024-11-13 14:29:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (105, 103, '图像去雾', 'SCANet-105', null, 'SCANet/Gmodel_105.tar', '27.7 MB', null, null,
        'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:29:11', '2024-11-13 14:29:11', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (106, 103, '图像去雾', 'SCANet-120', null, 'SCANet/Gmodel_120.tar', '27.68 MB', null, null,
        'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:29:16', '2024-11-13 14:29:16', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (107, 0, '图像去雾', 'SGIDPFF', null, 'SGID-PFF', null, null, null, 'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:29:22', '2024-11-13 14:29:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (108, 107, '图像去雾', 'indoor', null, 'SGID-PFF/SOTS_indoor.pt', '52.94 MB', '13.22 MB', '145.66 GB',
        'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:29:28', '2024-12-03 21:23:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (109, 107, '图像去雾', 'SGIDPFF', null, 'SGID-PFF/SOTS_outdoor.pt', '52.94 MB', '13.22 MB', '145.66 GB',
        'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        3, '2024-11-13 14:29:33', '2024-12-03 21:23:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (110, 0, '图像去雾', 'TSDNet', null, 'TSDNet/GNet.tar', '13.94 MB', null, null, 'algorithm.TSDNet.run',
        'TSDNet (Temporal-Spatial-Depth Network): TSDNet 是一种时间-空间-深度联合建模的图像去雾模型，通过优化视频序列的去雾效果，提高视频去雾的连贯性和自然度。该模型在处理视频序列时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的视频去雾任务。',
        3, '2024-11-13 14:29:43', '2024-11-13 14:29:43', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (111, 0, '图像去雾', 'WPXNet', null, 'WPXNet', null, null, null, 'algorithm.WPXNet.run',
        '引入无雾图像训练得到离散码本，封装具有原有图像色彩和结构的高质量先验知识。随后构建一种双分支神经网络结构，即先验匹配分支和通道注意力分支，利用邻域注意力和通道注意力提取有雾图像全局特征并学习浓雾区域与底层场景之间复杂交互特征，通过特征融合模块对两个分支提取的特征进行融合。将高质量先验约束码本与有雾图像特征通过一种可控距离重计算操作进行匹配，从而替换图像中受到雾影响的区域。本发明对原有雾图像进行重建实现了端到端的图像去雾流程，在保留原图像细节和纹理结构的情况下，提高了有雾图像的清晰度和可识别度。',
        3, '2024-11-28 14:03:07', '2024-11-28 14:03:07', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (112, 111, '图像去雾', 'DENSE-HAZE', null, 'WPXNet/dense-haze.pth', '151.75 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 3, '2024-11-28 14:04:05', '2024-12-03 21:23:17', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (113, 111, '图像去雾', 'I-HAZE', null, 'WPXNet/i-haze.pth', '151.75 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 3, '2024-11-28 14:04:24', '2024-12-03 21:23:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (114, 111, '图像去雾', 'O-HAZE', null, 'WPXNet/o-haze.pth', '151.74 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 3, '2024-11-28 14:04:45', '2024-12-03 21:23:21', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (115, 111, '图像去雾', 'NH-HAZE-20', null, 'WPXNet/nh-haze-20.pth', '151.74 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 3, '2024-11-28 14:05:06', '2024-12-03 21:23:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (116, 111, '图像去雾', 'NH-HAZE-21', null, 'WPXNet/nh-haze-21.pth', '151.74 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 3, '2024-11-28 14:05:16', '2024-12-03 21:23:24', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (117, 111, '图像去雾', 'NH-HAZE-23', null, 'WPXNet/nh-haze-23.pth', '151.75 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 3, '2024-11-28 14:05:23', '2024-12-03 21:23:26', 2, 2);

SET FOREIGN_KEY_CHECKS = 1;
