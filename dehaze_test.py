from importlib import import_module

import torch

modules_to_test = [
    {"import_path": "algorithm.AECRNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\AECR-Net\\DH_train.pk',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\AECR-Net\\ITS_train.pk',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\AECR-Net\\NH_train.pk']},
    {"import_path": "algorithm.AODNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\AODNet\\dehazer.pth']},
    {"import_path": "algorithm.C2PNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\C2PNet\\ITS.pkl',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\C2PNet\\OTS.pkl']},
    {"import_path": "algorithm.CFENViTDehazing.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_daytime_realworld\\latest_net_G.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_nhhaze\\20_net_G.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_nighttime\\latest_net_G.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_ohaze\\20_net_G.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_reside\\32_net_G.pth']},
    {"import_path": "algorithm.CMFNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CMFNet\\deblur_GoPro_CMFNet.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CMFNet\\dehaze_I_OHaze_CMFNet.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CMFNet\\deraindrop_DeRainDrop_CMFNet.pth']},
    {"import_path": "algorithm.D4.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\D4\\weights_reconstruct.pth']},
    {"import_path": "algorithm.DaclipUir.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\daclip-uir\\daclip_ViT-B-32.pt']},
    {"import_path": "algorithm.DCPDN.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DCPDN\\netG_epoch_8.pth']},
    {"import_path": "algorithm.DEANet.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DEA-Net\\HAZE4K\\PSNR3426_SSIM9885.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DEA-Net\\ITS\\PSNR4131_SSIM9945.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DEA-Net\\OTS\\PSNR3659_SSIM9897.pth']},
    {"import_path": "algorithm.Dehamer.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\Dehamer\\dense\\PSNR1662_SSIM05602.pt',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\Dehamer\\indoor\\PSNR3663_ssim09881.pt',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\Dehamer\\NH\\PSNR2066_SSIM06844.pt',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\Dehamer\\outdoor\\PSNR3518_SSIM09860.pt']},
    {"import_path": "algorithm.DehazeFormer.run", "model_path": [
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
    {"import_path": "algorithm.DehazeNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\DehazeNet\\defog4_noaug.pth']},
    {"import_path": "algorithm.FCD.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\FCD\\framework_da_230221_121802_gen.pth']},
    {"import_path": "algorithm.FFANet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\FFA-Net\\its_train_ffa_3_19.pk',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\FFA-Net\\ots_train_ffa_3_19.pk']},
    {"import_path": "algorithm.FogRemoval.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\FogRemoval\\NH-HAZE_params_0100000.pt']},
    {"import_path": "algorithm.GCANet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\GCANet\\wacv_gcanet_dehaze.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\GCANet\\wacv_gcanet_derain.pth']},
    {"import_path": "algorithm.GridDehazeNet.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\GridDehazeNet\\indoor_haze_best_3_6',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\GridDehazeNet\\outdoor_haze_best_3_6']},
    {"import_path": "algorithm.ImgRestorationSde.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\rain100h_sde.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\deblurring\\ir-sde-deblurring.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\denoising\\ir-sde-denoising.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\denoising\\refusion-denoising.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\deraining\\ir-sde-derainH100.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\deraining\\ir-sde-derainL100.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\image-restoration-sde\\sisr\\ir-sde-srx4.pth']},
    {"import_path": "algorithm.ITBdehaze.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\ITBdehaze\\best.pkl']},
    {"import_path": "algorithm.LightDehazeNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LightDehazeNet\\trained_LDNet.pth']},
    {"import_path": "algorithm.LKDNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\ITS\\LKD-b\\LKD-b.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\ITS\\LKD-l\\LKD-l.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\ITS\\LKD-s\\LKD-s.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\ITS\\LKD-t\\LKD-t.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\OTS\\LKD-b\\LKD-b.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\OTS\\LKD-l\\LKD-l.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\OTS\\LKD-s\\LKD-s.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\LKDNet\\OTS\\LKD-t\\LKD-t.pth']},
    {"import_path": "algorithm.MADN.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MADN\\model.pth']},
    {"import_path": "algorithm.MB-TaylorFormer.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\densehaze-MB-TaylorFormer-B.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\densehaze-MB-TaylorFormer-L.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\ITS-MB-TaylorFormer-L.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\ohaze-MB-TaylorFormer-B.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\OTS-MB-TaylorFormer-B.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MB-TaylorFormer\\OTS-MB-TaylorFormer-L.pth']},
    {"import_path": "algorithm.MixDehazeNet.run", "model_path": [
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\haze4k\\MixDehazeNet-l.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\indoor\\MixDehazeNet-b.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\indoor\\MixDehazeNet-l.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\outdoor\\MixDehazeNet-b.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\outdoor\\MixDehazeNet-l.pth',
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MixDehazeNet\\outdoor\\MixDehazeNet-s.pth']},
    {"import_path": "algorithm.MSFNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MSFNet\\indoor.pth',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\MSFNet\\outdoor.pth']},
    {"import_path": "algorithm.PSD.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\PSD\\PSB-MSBDN',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\PSD\\PSD-FFANET',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\PSD\\PSD-GCANET']},
    {"import_path": "algorithm.RIDCP.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\RIDCP\\pretrained_RIDCP.pth']},
    {"import_path": "algorithm.SCANet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SCANet\\Gmodel_105.tar',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SCANet\\Gmodel_120.tar',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SCANet\\Gmodel_40.tar']},
    {"import_path": "algorithm.SGIDPFF.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SGID-PFF\\SOTS_indoor.pt',
                    'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\SGID-PFF\\SOTS_outdoor.pt']},
    {"import_path": "algorithm.TSDNet.run",
     "model_path": ['E://ProgramProject//new-dehaze//dehaze-python//trained_model\\TSDNet\\GNet.tar']},
]
def one():
    # 定义待测试的模块列表
    count = 0
    import_error_count = 0
    attribute_error_count = 0
    other_error_count = 0
    hazy_img = "D:\DeepLearning\dataset\RESIDE\ITS\hazy//1_1_0.90179.png"
    clean_img = ".//output.png"
    for module in modules_to_test:
        import_path = module['import_path']
        model_paths = module['model_path']
        for model_path in model_paths:
            try:
                model = import_module(import_path)
                model.dehaze(hazy_img, clean_img, model_path)
            except ImportError as e:
                print("----------{}-------------".format(import_path))
                print("{} 不存在".format(import_path), e)
                import_error_count += 1
            except AttributeError as e:
                print("----------{}-------------".format(import_path))
                print("{} 的dehaze方法不存在".format(import_path), e)
                attribute_error_count += 1
            except Exception as e:
                print("----------{}-------------".format(import_path))
                print(e)
                other_error_count += 1
            finally:
                count += 1
    print("已测试{}个模型".format(count))
    print("导入错误{}个".format(import_error_count))
    print("属性错误{}个".format(attribute_error_count))
    print("其他错误{}个".format(other_error_count))

if __name__ == '__main__':
    hazy_img = "D:\DeepLearning\dataset\RESIDE\ITS\hazy//1_1_0.90179.png"
    clean_img = ".//output.png"
    for module in modules_to_test:
        import_path = module['import_path']
        model_paths = module['model_path']
        if import_path != "algorithm.CFENViTDehazing.run":
            continue
        for model_path in model_paths:
            model = import_module(import_path)
            model.dehaze(hazy_img, clean_img, model_path)
    print(torch.load(
        'E://ProgramProject//new-dehaze//dehaze-python//trained_model\\CFEN-ViT-Dehazing\\iid_hlgvit_crs_gd4_cfs_v3_daytime_realworld\\latest_net_G.pth').keys())
