import os


def list_files_in_directory(directory):
    # 获取当前目录下所有项
    items = os.listdir(directory)
    result = []

    for item in items:
        item_path = os.path.join(directory, item)
        # 检查是否为目录
        if os.path.isdir(item_path):
            # 创建一个字典来存储子文件夹的信息
            subdir_info = {"name": item, "path": []}
            # 遍历子文件夹中的所有文件
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 将文件的绝对路径添加到path数组中
                    subdir_info["path"].append(file_path)
            result.append(subdir_info)

    return result


# 调用函数并打印结果
subdir_files = list_files_in_directory("E://ProgramProject//new-dehaze//dehaze-python//trained_model")
print(subdir_files)

result = [{'name': 'AECR-Net',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\AECR-Net\\DH_train.pk',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\AECR-Net\\ITS_train.pk',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\AECR-Net\\NH_train.pk']},
          {'name': 'AODNet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\AODNet\\dehazer.pth']},
          {'name': 'C2PNet', 'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\C2PNet\\ITS.pkl',
                                      'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\C2PNet\\OTS.pkl']},
          {'name': 'CFEN-ViT-Dehazing', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_daytime_realworld\\latest_net_G.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_nhhaze\\20_net_G.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_nighttime\\latest_net_G.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_ohaze\\20_net_G.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_reside\\32_net_G.pth']},
          {'name': 'CMFNet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CMFNet\\deblur_GoPro_CMFNet.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CMFNet\\dehaze_I_OHaze_CMFNet.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CMFNet\\deraindrop_DeRainDrop_CMFNet.pth']},
          {'name': 'D4', 'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\D4\\weights_reconstruct.pth']},
          {'name': 'daclip-uir',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\daclip-uir\\daclip_ViT-B-32.pt',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\daclip-uir\\universal-ir.pth']},
          {'name': 'DCPDN',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DCPDN\\netG_epoch_8.pth']},
          {'name': 'DEA-Net', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DEA-Net\\HAZE4K\\PSNR3426_SSIM9885.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DEA-Net\\ITS\\PSNR4131_SSIM9945.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DEA-Net\\OTS\\PSNR3659_SSIM9897.pth']},
          {'name': 'Dehamer', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\Dehamer\\dense\\PSNR1662_SSIM05602.pt',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\Dehamer\\indoor\\PSNR3663_ssim09881.pt',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\Dehamer\\NH\\PSNR2066_SSIM06844.pt',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\Dehamer\\outdoor\\PSNR3518_SSIM09860.pt']},
          {'name': 'DehazeFormer', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\indoor\\dehazeformer-b.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\indoor\\dehazeformer-d.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\indoor\\dehazeformer-l.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\indoor\\dehazeformer-m.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\indoor\\dehazeformer-s.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\indoor\\dehazeformer-t.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\indoor\\dehazeformer-w.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\outdoor\\dehazeformer-b.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\outdoor\\dehazeformer-m.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\outdoor\\dehazeformer-s.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\outdoor\\dehazeformer-t.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\reside6k\\dehazeformer-b.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\reside6k\\dehazeformer-m.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\reside6k\\dehazeformer-s.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\reside6k\\dehazeformer-t.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\rshaze\\dehazeformer-b.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\rshaze\\dehazeformer-m.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\rshaze\\dehazeformer-s.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeFormer\\rshaze\\dehazeformer-t.pth']},
          {'name': 'DehazeNet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeNet\\defog4_noaug.pth']},
          {'name': 'FCD', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\FCD\\framework_da_230221_121802_gen.pth']},
          {'name': 'FFA-Net',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\FFA-Net\\its_train_ffa_3_19.pk',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\FFA-Net\\ots_train_ffa_3_19.pk']},
          {'name': 'FogRemoval', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\FogRemoval\\NH-HAZE_params_0100000.pt']},
          {'name': 'GCANet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\GCANet\\wacv_gcanet_dehaze.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\GCANet\\wacv_gcanet_derain.pth']},
          {'name': 'GridDehazeNet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\GridDehazeNet\\indoor_haze_best_3_6',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\GridDehazeNet\\outdoor_haze_best_3_6']},
          {'name': 'image-restoration-sde', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\rain100h_sde.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\deblurring\\ir-sde-deblurring.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\denoising\\ir-sde-denoising.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\denoising\\refusion-denoising.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\deraining\\ir-sde-derainH100.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\deraining\\ir-sde-derainL100.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\sisr\\ir-sde-srx4.pth']},
          {'name': 'ITBdehaze',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\ITBdehaze\\best.pkl',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\ITBdehaze\\swinv2_base_patch4_window8_256.pth']},
          {'name': 'LightDehazeNet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LightDehazeNet\\Epoch59.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LightDehazeNet\\trained_LDNet.pth']},
          {'name': 'LKDNet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\ITS\\LKD-b\\LKD-b.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\ITS\\LKD-l\\LKD-l.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\ITS\\LKD-s\\LKD-s.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\ITS\\LKD-t\\LKD-t.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\OTS\\LKD-b\\LKD-b.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\OTS\\LKD-l\\LKD-l.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\OTS\\LKD-s\\LKD-s.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\OTS\\LKD-t\\LKD-t.pth']},
          {'name': 'MADN',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MADN\\dehaze_80_state_dict.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MADN\\model.pth']},
          {'name': 'MB-TaylorFormer', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\densehaze-MB-TaylorFormer-B.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\densehaze-MB-TaylorFormer-L.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\ITS-MB-TaylorFormer-L.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\ohaze-MB-TaylorFormer-B.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\OTS-MB-TaylorFormer-B.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\OTS-MB-TaylorFormer-L.pth']},
          {'name': 'MixDehazeNet', 'path': [
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\haze4k\\MixDehazeNet-l.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\indoor\\MixDehazeNet-b.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\indoor\\MixDehazeNet-l.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\outdoor\\MixDehazeNet-b.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\outdoor\\MixDehazeNet-l.pth',
              'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\outdoor\\MixDehazeNet-s.pth']},
          {'name': 'MSFNet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MSFNet\\indoor.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MSFNet\\outdoor.pth']},
          {'name': 'PSD', 'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\PSD\\PSB-MSBDN',
                                   'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\PSD\\PSD-FFANET',
                                   'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\PSD\\PSD-GCANET']},
          {'name': 'RIDCP',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\RIDCP\\pretrained_HQPs.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\RIDCP\\pretrained_RIDCP.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\RIDCP\\weight_for_matching_dehazing_Flickr.pth']},
          {'name': 'SCANet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SCANet\\Gmodel_105.tar',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SCANet\\Gmodel_120.tar',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SCANet\\Gmodel_40.tar']},
          {'name': 'SGID-PFF',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SGID-PFF\\SOTS_indoor.pt',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SGID-PFF\\SOTS_outdoor.pt']},
          {'name': 'TSDNet',
           'path': ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\TSDNet\\GNet.tar']}]
