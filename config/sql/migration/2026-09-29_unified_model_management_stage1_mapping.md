# 阶段1 迁移：sys_algorithm → sys_ai_model(image_edit) 完整映射表

> 由 `2026-09-29_unified_model_management_stage1.sql` 第⑤段生成；基于 `config/sql/data/sys_algorithm.sql`（117 行）。
> 迁移 89 行（path 非空且 type<>'TEST'）→ 85 个唯一 model_id；同 weights path 的重复行合并（保留 id 最小者），
> 被合并行在下表 `merge` 列标注为 `→ id N`。最长 model_id 46 字符。

| algo id | name            | type       | algo status | 迁移后 status | path                                                   | model_id                                             |
| ------- | --------------- | ---------- | ----------- | ------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| 1       | AECR-NET        | 图像去雾   | 4           | 1             | AECR-Net/NH_train.pk                                   | `aecrnet`                                            |
| 2       | AODNet          | 图像去雾   | 4           | 1             | AODNet/dehazer.pth                                     | `aodnet`                                             |
| 3       | C2PNet          | 图像去雾   | 4           | 1             | C2PNet/ITS.pkl                                         | `c2pnet-its`                                         |
| 4       | 室内去雾（ITS） | 图像去雾   | 4           | 1             | C2PNet/ITS.pkl                                         | `c2pnet-its` (merge→id 3)                            |
| 5       | 室外去雾（OTS） | 图像去雾   | 4           | 1             | C2PNet/OTS.pkl                                         | `c2pnet-ots`                                         |
| 7       | 去雾模型        | 图像去雾   | 4           | 1             | CMFNet/dehaze_I_OHaze_CMFNet.pth                       | `cmfnet-dehaze-i-ohaze-cmfnet`                       |
| 8       | 去模糊模型      | 图像去模糊 | 4           | 1             | CMFNet/deblur_GoPro_CMFNet.pth                         | `cmfnet-deblur-gopro-cmfnet`                         |
| 9       | 去雨模型        | 图像去雨   | 4           | 1             | CMFNet/deraindrop_DeRainDrop_CMFNet.pth                | `cmfnet-deraindrop-deraindrop-cmfnet`                |
| 10      | D4              | 图像去雾   | 4           | 1             | D4/weights_reconstruct.pth                             | `d4`                                                 |
| 11      | DaclipUir       | 图像去雾   | 5           | 0             | daclip-uir/daclip_ViT-B-32.pt                          | `daclipuir`                                          |
| 12      | DCPDN           | 图像去雾   | 5           | 0             | DCPDN/netG_epoch_8.pth                                 | `dcpdn`                                              |
| 15      | HAZE4k模型      | 图像去雾   | 4           | 1             | DEA-Net/HAZE4K/PSNR3426_SSIM9885.pth                   | `deanet-haze4k-psnr3426-ssim9885`                    |
| 16      | ITS模型         | 图像去雾   | 4           | 1             | DEA-Net/ITS/PSNR4131_SSIM9945.pth                      | `deanet-its-psnr4131-ssim9945`                       |
| 17      | OTS模型         | 图像去雾   | 4           | 1             | DEA-Net/OTS/PSNR3659_SSIM9897.pth                      | `deanet-ots-psnr3659-ssim9897`                       |
| 19      | dense-haze模型  | 图像去雾   | 4           | 1             | Dehamer/dense/PSNR1662_SSIM05602.pt                    | `dehamer-dense-psnr1662-ssim05602`                   |
| 20      | indoor          | 图像去雾   | 4           | 1             | Dehamer/indoor/PSNR3663_ssim09881.pt                   | `dehamer-indoor-psnr3663-ssim09881`                  |
| 21      | NH-HAZE-模型    | 图像去雾   | 4           | 1             | Dehamer/NH/PSNR2066_SSIM06844.pt                       | `dehamer-nh-psnr2066-ssim06844`                      |
| 22      | outdoor         | 图像去雾   | 4           | 1             | Dehamer/outdoor/PSNR3518_SSIM09860.pt                  | `dehamer-outdoor-psnr3518-ssim09860`                 |
| 25      | indoor-b        | 图像去雾   | 4           | 1             | DehazeFormer/indoor/dehazeformer-b.pth                 | `dehazeformer-indoor-dehazeformer-b`                 |
| 26      | indoor-d        | 图像去雾   | 4           | 1             | DehazeFormer/indoor/dehazeformer-d.pth                 | `dehazeformer-indoor-dehazeformer-d`                 |
| 27      | indoor-l        | 图像去雾   | 4           | 1             | DehazeFormer/indoor/dehazeformer-l.pth                 | `dehazeformer-indoor-dehazeformer-l`                 |
| 28      | indoor-m        | 图像去雾   | 4           | 1             | DehazeFormer/indoor/dehazeformer-m.pth                 | `dehazeformer-indoor-dehazeformer-m`                 |
| 29      | indoor-s        | 图像去雾   | 4           | 1             | DehazeFormer/indoor/dehazeformer-s.pth                 | `dehazeformer-indoor-dehazeformer-s`                 |
| 30      | indoor-t        | 图像去雾   | 4           | 1             | DehazeFormer/indoor/dehazeformer-t.pth                 | `dehazeformer-indoor-dehazeformer-t`                 |
| 31      | indoor-w        | 图像去雾   | 4           | 1             | DehazeFormer/indoor/dehazeformer-w.pth                 | `dehazeformer-indoor-dehazeformer-w`                 |
| 33      | outdoor-b       | 图像去雾   | 4           | 1             | DehazeFormer/outdoor/dehazeformer-b.pth                | `dehazeformer-outdoor-dehazeformer-b`                |
| 34      | outdoor-m       | 图像去雾   | 4           | 1             | DehazeFormer/outdoor/dehazeformer-m.pth                | `dehazeformer-outdoor-dehazeformer-m`                |
| 35      | outdoor-s       | 图像去雾   | 4           | 1             | DehazeFormer/outdoor/dehazeformer-s.pth                | `dehazeformer-outdoor-dehazeformer-s`                |
| 36      | outdoor-t       | 图像去雾   | 4           | 1             | DehazeFormer/outdoor/dehazeformer-t.pth                | `dehazeformer-outdoor-dehazeformer-t`                |
| 38      | reside6k-b      | 图像去雾   | 4           | 1             | DehazeFormer/reside6k/dehazeformer-b.pth               | `dehazeformer-reside6k-dehazeformer-b`               |
| 39      | reside6k-b      | 图像去雾   | 4           | 1             | DehazeFormer/reside6k/dehazeformer-b.pth               | `dehazeformer-reside6k-dehazeformer-b` (merge→id 38) |
| 40      | reside6k-b      | 图像去雾   | 4           | 1             | DehazeFormer/reside6k/dehazeformer-b.pth               | `dehazeformer-reside6k-dehazeformer-b` (merge→id 38) |
| 41      | reside6k-b      | 图像去雾   | 4           | 1             | DehazeFormer/reside6k/dehazeformer-b.pth               | `dehazeformer-reside6k-dehazeformer-b` (merge→id 38) |
| 43      | rshaze-b        | 图像去雾   | 4           | 1             | DehazeFormer/rshaze/dehazeformer-b.pth                 | `dehazeformer-rshaze-dehazeformer-b`                 |
| 44      | rshaze-m        | 图像去雾   | 4           | 1             | DehazeFormer/rshaze/dehazeformer-m.pth                 | `dehazeformer-rshaze-dehazeformer-m`                 |
| 45      | rshaze-s        | 图像去雾   | 4           | 1             | DehazeFormer/rshaze/dehazeformer-s.pth                 | `dehazeformer-rshaze-dehazeformer-s`                 |
| 46      | rshaze-t        | 图像去雾   | 4           | 1             | DehazeFormer/rshaze/dehazeformer-t.pth                 | `dehazeformer-rshaze-dehazeformer-t`                 |
| 47      | FCD             | 图像去雾   | 5           | 0             | FCD/framework_da_230221_121802_gen.pth                 | `fcd`                                                |
| 49      | its             | 图像去雾   | 4           | 1             | FFA-Net/its_train_ffa_3_19.pk                          | `ffanet-its-train-ffa-3-19`                          |
| 50      | ots             | 图像去雾   | 4           | 1             | FFA-Net/ots_train_ffa_3_19.pk                          | `ffanet-ots-train-ffa-3-19`                          |
| 51      | FogRemoval      | 图像去雾   | 4           | 1             | FogRemoval/NH-HAZE_params_0100000.pt                   | `fogremoval`                                         |
| 52      | GCANet          | 图像去雾   | 4           | 1             | GCANet/wacv_gcanet_dehaze.pth                          | `gcanet`                                             |
| 54      | indoor          | 图像去雾   | 4           | 1             | GridDehazeNet/indoor_haze_best_3_6                     | `griddehazenet-indoor-haze-best-3-6`                 |
| 55      | outdoor         | 图像去雾   | 4           | 1             | GridDehazeNet/outdoor_haze_best_3_6                    | `griddehazenet-outdoor-haze-best-3-6`                |
| 57      | deblurring      | 图像去模糊 | 5           | 0             | image-restoration-sde/deblurring/ir-sde-deblurring.pth | `imgrestorationsde-deblurring-ir-sde-deblurring`     |
| 58      | denoising       | 图像去噪   | 5           | 0             | image-restoration-sde/denoising/ir-sde-denoising.pth   | `imgrestorationsde-denoising-ir-sde-denoising`       |
| 60      | deraining-H100  | 图像去雨   | 5           | 0             | image-restoration-sde/deraining/ir-sde-derainH100.pth  | `imgrestorationsde-deraining-ir-sde-derainh100`      |
| 61      | deraining-L100  | 图像去雨   | 5           | 0             | image-restoration-sde/deraining/ir-sde-derainL100.pth  | `imgrestorationsde-deraining-ir-sde-derainl100`      |
| 62      | ITBDehaze       | 图像去雾   | 4           | 1             | ITBdehaze/best.pkl                                     | `itbdehaze`                                          |
| 63      | LightDehazeNet  | 图像去雾   | 4           | 1             | LightDehazeNet/trained_LDNet.pth                       | `lightdehazenet`                                     |
| 66      | ITS-b           | 图像去雾   | 4           | 1             | LKDNet/ITS/LKD-b/LKD-b.pth                             | `lkdnet-its-lkd-b-lkd-b`                             |
| 67      | ITS-l           | 图像去雾   | 4           | 1             | LKDNet/ITS/LKD-l/LKD-l.pth                             | `lkdnet-its-lkd-l-lkd-l`                             |
| 68      | ITS-s           | 图像去雾   | 4           | 1             | LKDNet/ITS/LKD-s/LKD-s.pth                             | `lkdnet-its-lkd-s-lkd-s`                             |
| 69      | ITS-t           | 图像去雾   | 4           | 1             | LKDNet/ITS/LKD-t/LKD-t.pth                             | `lkdnet-its-lkd-t-lkd-t`                             |
| 71      | OTS-b           | 图像去雾   | 4           | 1             | LKDNet/OTS/LKD-b/LKD-b.pth                             | `lkdnet-ots-lkd-b-lkd-b`                             |
| 72      | OTS-l           | 图像去雾   | 4           | 1             | LKDNet/OTS/LKD-l/LKD-l.pth                             | `lkdnet-ots-lkd-l-lkd-l`                             |
| 73      | OTS-s           | 图像去雾   | 4           | 1             | LKDNet/OTS/LKD-s/LKD-s.pth                             | `lkdnet-ots-lkd-s-lkd-s`                             |
| 74      | OTS-t           | 图像去雾   | 4           | 1             | LKDNet/OTS/LKD-t/LKD-t.pth                             | `lkdnet-ots-lkd-t-lkd-t`                             |
| 75      | MADN            | 图像去雾   | 4           | 1             | MADN/model.pth                                         | `madn`                                               |
| 78      | dense-haze-b    | 图像去雾   | 4           | 1             | MB-TaylorFormer/densehaze-MB-TaylorFormer-B.pth        | `mb-taylorformer-densehaze-mb-taylorformer-b`        |
| 79      | dense-haze-l    | 图像去雾   | 4           | 1             | MB-TaylorFormer/densehaze-MB-TaylorFormer-L.pth        | `mb-taylorformer-densehaze-mb-taylorformer-l`        |
| 80      | its             | 图像去雾   | 4           | 1             | MB-TaylorFormer/ITS-MB-TaylorFormer-L.pth              | `mb-taylorformer-its-mb-taylorformer-l`              |
| 81      | ohaze           | 图像去雾   | 4           | 1             | MB-TaylorFormer/ohaze-MB-TaylorFormer-B.pth            | `mb-taylorformer-ohaze-mb-taylorformer-b`            |
| 83      | ots-b           | 图像去雾   | 4           | 1             | MB-TaylorFormer/OTS-MB-TaylorFormer-B.pth              | `mb-taylorformer-ots-mb-taylorformer-b`              |
| 84      | ots-l           | 图像去雾   | 4           | 1             | MB-TaylorFormer/OTS-MB-TaylorFormer-L.pth              | `mb-taylorformer-ots-mb-taylorformer-l`              |
| 87      | haze4k-l        | 图像去雾   | 4           | 1             | MixDehazeNet/haze4k/MixDehazeNet-l.pth                 | `mixdehazenet-haze4k-mixdehazenet-l`                 |
| 89      | indoor-b        | 图像去雾   | 4           | 1             | MixDehazeNet/indoor/MixDehazeNet-b.pth                 | `mixdehazenet-indoor-mixdehazenet-b`                 |
| 90      | indoor-l        | 图像去雾   | 4           | 1             | MixDehazeNet/indoor/MixDehazeNet-l.pth                 | `mixdehazenet-indoor-mixdehazenet-l`                 |
| 92      | outdoor-b       | 图像去雾   | 4           | 1             | MixDehazeNet/outdoor/MixDehazeNet-b.pth                | `mixdehazenet-outdoor-mixdehazenet-b`                |
| 93      | outdoor-b       | 图像去雾   | 4           | 1             | MixDehazeNet/outdoor/MixDehazeNet-l.pth                | `mixdehazenet-outdoor-mixdehazenet-l`                |
| 94      | outdoor-b       | 图像去雾   | 4           | 1             | MixDehazeNet/outdoor/MixDehazeNet-s.pth                | `mixdehazenet-outdoor-mixdehazenet-s`                |
| 96      | indoor          | 图像去雾   | 4           | 1             | MSFNet/indoor.pth                                      | `msfnet-indoor`                                      |
| 97      | outdoor         | 图像去雾   | 4           | 1             | MSFNet/outdoor.pth                                     | `msfnet-outdoor`                                     |
| 99      | PSD-MSBDN       | 图像去雾   | 4           | 1             | PSD/PSB-MSBDN                                          | `psd-psb-msbdn`                                      |
| 100     | PSD-FFANET      | 图像去雾   | 4           | 1             | PSD/PSD-FFANET                                         | `psd-psd-ffanet`                                     |
| 101     | PSD-GCANET      | 图像去雾   | 4           | 1             | PSD/PSD-GCANET                                         | `psd-psd-gcanet`                                     |
| 102     | RIDCP           | 图像去雾   | 4           | 1             | RIDCP/pretrained_RIDCP.pth                             | `ridcp`                                              |
| 104     | SCANet-40       | 图像去雾   | 4           | 1             | SCANet/Gmodel_40.tar                                   | `scanet-gmodel-40`                                   |
| 105     | SCANet-105      | 图像去雾   | 4           | 1             | SCANet/Gmodel_105.tar                                  | `scanet-gmodel-105`                                  |
| 106     | SCANet-120      | 图像去雾   | 4           | 1             | SCANet/Gmodel_120.tar                                  | `scanet-gmodel-120`                                  |
| 108     | indoor          | 图像去雾   | 4           | 1             | SGID-PFF/SOTS_indoor.pt                                | `sgidpff-sots-indoor`                                |
| 109     | SGIDPFF         | 图像去雾   | 4           | 1             | SGID-PFF/SOTS_outdoor.pt                               | `sgidpff-sots-outdoor`                               |
| 110     | TSDNet          | 图像去雾   | 4           | 1             | TSDNet/GNet.tar                                        | `tsdnet`                                             |
| 112     | DENSE-HAZE      | 图像去雾   | 4           | 1             | WPXNet/dense-haze.pth                                  | `wpxnet-dense-haze`                                  |
| 113     | I-HAZE          | 图像去雾   | 4           | 1             | WPXNet/i-haze.pth                                      | `wpxnet-i-haze`                                      |
| 114     | O-HAZE          | 图像去雾   | 4           | 1             | WPXNet/o-haze.pth                                      | `wpxnet-o-haze`                                      |
| 115     | NH-HAZE-20      | 图像去雾   | 4           | 1             | WPXNet/nh-haze-20.pth                                  | `wpxnet-nh-haze-20`                                  |
| 116     | NH-HAZE-21      | 图像去雾   | 4           | 1             | WPXNet/nh-haze-21.pth                                  | `wpxnet-nh-haze-21`                                  |
| 117     | NH-HAZE-23      | 图像去雾   | 4           | 1             | WPXNet/nh-haze-23.pth                                  | `wpxnet-nh-haze-23`                                  |
